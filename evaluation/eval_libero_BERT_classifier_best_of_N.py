"""
Best-of-N evaluation for BERT success classifier: sample N actions from the actor,
score with classifier, take the best. Supports splitting n_vals across multiple GPUs.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # hack to be able to "see" evaluation_libero easily

import argparse
import copy
import json
import multiprocessing as mp
import pickle
import secrets
import time
from pathlib import Path
from typing import List

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import wandb

from agents.acbcflowactor import ACBCFlowActorAgent, get_config as get_actor_config
from envs.env_utils import make_env_and_datasets
from evaluation_libero import evaluate
from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState
from utils.log_utils import get_wandb_video
from utils.networks import MLP

# set the egl_device_id to be same as cuda_visible_devices
os.environ['MUJOCO_GL'] = 'egl'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_triton_gemm_any=True'

# from classifier.train_BERT_success_classifier import TransClassifier_BERT, load_classifier
# # -----------------------------------------------------------------------------
# # Classifier model (must match train_BERT_success_classifier)
# # -----------------------------------------------------------------------------

class TransClassifier_BERT(nn.Module):
    vision_encoder: nn.Module = None
    layer_norm: bool = True
    p_drop_state: float = 0.5
    embed_dim: int = 128

    def setup(self):
        self.action_encoder = MLP((self.embed_dim, self.embed_dim), activate_final=True, layer_norm=self.layer_norm)
        self.lang_encoder = MLP((self.embed_dim, self.embed_dim), activate_final=True, layer_norm=self.layer_norm)
        self.classifier = MLP((self.embed_dim, self.embed_dim, 1), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, actions, language_embedding, train=True, rng=None):
        assert actions is not None, "Actions must be provided to the classifier"
        assert self.vision_encoder is not None, "Encoder must be provided to the classifier"

        obs_encoded = self.vision_encoder(observations)
        if rng is None:
            rng = jax.random.PRNGKey(0)
        mask = jax.random.bernoulli(rng, 1 - self.p_drop_state, shape=(obs_encoded.shape[0], 1))
        obs_encoded = jax.lax.cond(
            train,
            lambda x: mask * x,
            lambda x: x,
            obs_encoded
        )
        action_encoded = self.action_encoder(actions)
        lang_encoded = self.lang_encoder(language_embedding)
        inputs = jnp.concatenate([obs_encoded, action_encoded, lang_encoded], axis=-1)
        return self.classifier(inputs)


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

class LoggingHelper:
    def __init__(self, wandb_logger, to_log=True):
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()
        self.to_log = to_log

    def log(self, data, prefix, step):
        if not self.to_log:
            return
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)


# -----------------------------------------------------------------------------
# Load classifier from checkpoint
## TODO(YY): eventually just import this and transclassifier from file
# -----------------------------------------------------------------------------

def load_classifier(save_dir, ckpt_num):
    save_dir = Path(save_dir)
    with open(save_dir / f'hparams_{ckpt_num}.json', 'r') as f:
        hparams = json.load(f)
    with open(save_dir / f'params_{ckpt_num}.pkl', 'rb') as f:
        params = pickle.load(f)
    classifier_def = TransClassifier_BERT(
        vision_encoder=encoder_modules[hparams['encoder']](),
        embed_dim=hparams['embed_dim'],
        layer_norm=hparams['layer_norm'],
        p_drop_state=hparams['p_drop_state'],
    )
    networks = {'classifier': classifier_def}
    network_def = ModuleDict(networks)
    network_tx = optax.adam(learning_rate=hparams['lr'])
    network = TrainState.create(network_def, params, tx=network_tx)
    return network, hparams


# -----------------------------------------------------------------------------
# ClassifierAgent: actor + classifier for best-of-N
# -----------------------------------------------------------------------------

class ClassifierAgent:
    def __init__(self, classifier_network_restore_dir, classifier_ckpt_num, actor_restore_path, example_batch, horizon_length, actor_encoder, num_samples, actor_seed):
        class_net, class_hparams = load_classifier(classifier_network_restore_dir, classifier_ckpt_num)
        self.classifier_network = class_net
        self.actor_network = self._restore_actor_network(actor_restore_path, copy.deepcopy(example_batch), horizon_length, actor_encoder, actor_seed)
        self.config = {
            "action_dim": example_batch["actions"].shape[-1],
            "horizon_length": horizon_length,
            "action_chunking": True,
            "actor_encoder": actor_encoder,
            "flow_steps": 10,
            "num_samples": num_samples,
        }

    def _restore_actor_network(self, actor_restore_path, example_batch, horizon_length, actor_encoder, actor_seed):
        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        with open(actor_restore_path, 'rb') as f:
            load_dict = pickle.load(f)
        actor_agent_class = ACBCFlowActorAgent
        actor_config = get_actor_config()
        actor_config['encoder'] = actor_encoder
        actor_config['horizon_length'] = horizon_length
        actor_agent = actor_agent_class.create(
            seed=actor_seed,
            ex_observations=copy.deepcopy(ex_observations),
            ex_actions=copy.deepcopy(ex_actions),
            config=actor_config,
        )
        loaded_actor_params = load_dict['agent']['network']['params']
        actor_agent_state_dict = flax.serialization.to_state_dict(actor_agent)
        actor_agent_params = actor_agent_state_dict['network']['params']
        for key in actor_agent_params:
            if 'actor' in key:
                actor_agent_params[key] = loaded_actor_params[key]
        actor_agent = flax.serialization.from_state_dict(actor_agent, actor_agent_state_dict)
        return actor_agent.network

    def sample_actions(self, observations, rng=None, temperature=1.0, print_debug=False):
        if self.actor_network is None:
            raise ValueError("Actor network not found")
        full_action_dim = self.config["action_dim"] * (self.config["horizon_length"] if self.config["action_chunking"] else 1)
        k = sorted(observations.keys())[0]
        batch_size = observations[k].shape[0]
        lang_embedding = observations.get('language')

        noises = jax.random.normal(
            rng,
            (batch_size, self.config['num_samples'], full_action_dim),
        )
        observations = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x[:, None, ...], self.config["num_samples"], axis=1),
            observations
        )
        actions = self._compute_flow_actions(observations, noises)
        actions = jnp.clip(actions, -1, 1)

        observations.pop('proprio')
        flat_obs = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]),
            observations
        )
        flat_actions = jnp.reshape(actions, (-1, full_action_dim))
        flat_lang = jnp.repeat(lang_embedding, self.config['num_samples'], axis=0)

        logits = self.classifier_network.select('classifier')(
            flat_obs, flat_actions, flat_lang, train=False
        )
        logits = jnp.reshape(logits, (batch_size, self.config['num_samples']))
        indices = jnp.argmax(logits, axis=-1)
        actions = actions[jnp.arange(batch_size), indices, :]
        return actions

    def _compute_flow_actions(self, observations, noises):
        if self.config['actor_encoder'] is not None:
            observations = self.actor_network.select('actor_flow_encoder')(observations)
        actions = noises
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.actor_network.select('actor_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions


# -----------------------------------------------------------------------------
# Evaluation loop
# -----------------------------------------------------------------------------

def eval_agent(agent, eval_envs_iterator, example_batch, num_eval_episodes, num_video_episodes, num_parallel_envs, video_frame_skip, logger, n):
    all_eval_info = []
    for (eval_env_j, eval_env_j_name) in tqdm.tqdm(eval_envs_iterator, desc="Evaluating multi-task", position=0, leave=False):
        eval_info, trajs, renders = evaluate(
            agent=agent,
            env=eval_env_j,
            action_dim=example_batch["actions"].shape[-1],
            num_eval_episodes=num_eval_episodes,
            num_video_episodes=num_video_episodes,
            num_parallel_envs=num_parallel_envs,
            video_frame_skip=video_frame_skip,
        )
        eval_env_j.close()
        all_eval_info.append(eval_info)
        if len(renders) > 0:
            eval_info['video'] = get_wandb_video(renders)
        logger.log(eval_info, f"eval_{eval_env_j_name}", step=n)
    eval_info_agg = {k: np.mean([e[k] for e in all_eval_info]) for k in all_eval_info[0].keys() if k != 'video'}
    logger.log(eval_info_agg, "eval", step=n)
    return eval_info_agg, all_eval_info


# -----------------------------------------------------------------------------
# Main (runs on a single process; n_vals and gpu_id set by caller)
# -----------------------------------------------------------------------------

def main(args):
    classifier_restore_dir = args.classifier_restore_dir
    classifier_ckpt_num = args.classifier_ckpt_num
    actor_restore_path = args.actor_restore_path
    results_save_path = args.results_save_path + f'/{args.wandb_run_name}'
    env_name = args.env_name
    task_name = args.task_name
    n_vals = args.n_vals
    wandb_group_name = args.wandb_group_name
    wandb_run_name = args.wandb_run_name
    horizon_length = args.horizon_length
    num_eval_episodes = args.num_eval_episodes
    num_video_episodes = args.num_video_episodes
    num_parallel_envs = args.num_parallel_envs
    video_frame_skip = args.video_frame_skip
    language_embedder = args.language_embedder
    keys_to_load = args.keys_to_load
    actor_encoder = args.actor_encoder
    actor_seed = args.actor_seed
    use_wandb = args.use_wandb
    classifier_type = args.classifier_type
    assert n_vals, "n_vals must be non-empty"
    assert env_name, "env_name must be provided"
    assert classifier_type, "classifier_type must be provided"
    assert classifier_type in ['language', 'success'], "classifier_type must be either 'language' or 'success'"
    os.makedirs(results_save_path, exist_ok=True)

    if use_wandb:
        # Single GPU: this process creates the run (resume="allow").
        # Multi GPU: run was already created by a helper subprocess; we only resume (resume="must").
        wandb_run_id = getattr(args, 'wandb_run_id', None)
        init_timeout = getattr(args, 'wandb_init_timeout', 300)
        resume_mode = "must" if getattr(args, "wandb_run_created_by_helper", False) else "allow"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=wandb_group_name,
            name=wandb_run_name,
            id=wandb_run_id,
            resume=resume_mode,
            settings=wandb.Settings(init_timeout=init_timeout),
        )
        logger = LoggingHelper(wandb_logger=wandb)
    else:
        logger = LoggingHelper(wandb_logger=None, to_log=False)

    augmentation_type = 'none'
    augmentation_reward = False
    batch_level_sampling = True
    discount = 0.99

    _, eval_envs_iterator, dataset, _ = make_env_and_datasets(
        env_name,
        task_name,
        language_embedder,
        augmentation_type,
        augmentation_reward,
        num_parallel_envs=num_parallel_envs,
        keys_to_load=keys_to_load,
        batch_level_sampling=batch_level_sampling,
        demo_nums_to_use_per_task=args.demo_nums_to_use_per_task,
        augmentation_dict=None,
        is_notebook=False,
    )
    # eval_envs_list = list(eval_envs_iterator)

    example_batch = dataset.sample_sequence(1, sequence_length=horizon_length, discount=discount)

    for N in n_vals:
        classifier_agent = ClassifierAgent(
            classifier_restore_dir,
            classifier_ckpt_num,
            actor_restore_path,
            example_batch,
            horizon_length,
            actor_encoder,
            N,
            actor_seed,
        )
        agg_eval_info, all_eval_info = eval_agent(
            classifier_agent,
            eval_envs_iterator,
            example_batch,
            num_eval_episodes,
            num_video_episodes,
            num_parallel_envs,
            video_frame_skip,
            logger,
            n=N,
        )
        out_path = os.path.join(results_save_path, f'{wandb_run_name}_N_{N}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump({'agg_eval_info': agg_eval_info, 'all_eval_info': all_eval_info}, f)
        print(f"Saved eval info for N={N} to {out_path}")


def _create_wandb_run_subprocess(project, entity, group, name, run_id, init_timeout):
    """Run in a separate process to create the wandb run once. Avoids 409 Duplicate when
    multiple GPU processes all try to create the same run_id with resume='allow'."""
    wandb.init(
        project=project,
        entity=entity,
        group=group,
        name=name,
        id=run_id,
        resume="allow",
        settings=wandb.Settings(init_timeout=init_timeout),
    )
    # Run exists now; exit so parent can spawn GPU workers with resume="must"
    wandb.finish()


def run_on_gpu(args, gpu_id, n_vals_for_this_gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['EGL_DEVICE_ID'] = str(gpu_id)
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(gpu_id)
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    # set openblas threads to 1, mkl
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    xla_flags = os.environ.get('XLA_FLAGS', '')
    xla_flags += ' --xla_gpu_triton_gemm_any=True'
    os.environ['XLA_FLAGS'] = xla_flags
    args_copy = copy.deepcopy(args)
    args_copy.n_vals = n_vals_for_this_gpu
    args_copy.gpu_id = gpu_id
    main(args_copy)


# -----------------------------------------------------------------------------
# CLI: argparse + split n_vals across GPUs
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Best-of-N evaluation with BERT success classifier.')

    parser.add_argument('--classifier_type', choices=['language', 'success'], required=True, help='Classifier type, helpful for logging')
    parser.add_argument('--classifier_restore_dir', type=str, required=True, help='Directory with classifier params_*.pkl and hparams_*.json')
    parser.add_argument('--classifier_ckpt_num', type=int, required=True, help='Classifier checkpoint number (e.g. 204 for params_204.pkl)')
    
    parser.add_argument('--actor_restore_path', type=str, required=True, help='Path to actor checkpoint .pkl')
    parser.add_argument('--actor_encoder', type=str, default='combined_encoder_small', help='Actor encoder name used to train the actor network')
    
    parser.add_argument('--results_save_path', type=str, default='exp/multitask_RL/evaluations/', help='Directory to save per-N eval pkl files')


    parser.add_argument('--env_name', type=str, required=True, help='Environment name (e.g. libero_90)')
    parser.add_argument('--task_name', type=str, required=True, help='Task name (used with env_name for task subset)')
    parser.add_argument('--n_vals', nargs='+', type=int, required=True, help='Best-of-N values to evaluate (e.g. 1 2 4 8 16 32 64 128). Split across --gpus.')
    parser.add_argument('--gpus', nargs='+', type=int, required=True, help='GPU ids to use; n_vals are split across these (e.g. --gpus 0 1 2 3)')

    parser.add_argument('--wandb_group_name', type=str, required=True, help='W&B run group')
    parser.add_argument('--wandb_run_name', type=str, required=True, help='W&B run name')
    parser.add_argument('--wandb_project', type=str, default='multitask_RL', help='W&B project')
    parser.add_argument('--wandb_entity', type=str, default='yajatyadav', help='W&B entity')
    
    parser.add_argument('--horizon_length', type=int, default=5, help='Action chunk length')
    parser.add_argument('--num_eval_episodes', type=int, default=55, help='Evaluation episodes per env')
    parser.add_argument('--num_video_episodes', type=int, default=5, help='Video episodes per env')
    parser.add_argument('--num_parallel_envs', type=int, default=10, help='Parallel envs during eval')
    parser.add_argument('--video_frame_skip', type=int, default=3, help='Frame skip for videos')
    parser.add_argument('--language_embedder', type=str, default='bert', help='Language embedder name')
    parser.add_argument('--keys_to_load', nargs='+', type=str,
                        default=['agentview_rgb', 'eye_in_hand_rgb', 'language', 'proprio'],
                        help='Observation keys to load')
 
    
    parser.add_argument('--actor_seed', type=int, default=0, help='Actor network seed')
    parser.add_argument('--demo_nums_to_use_per_task', nargs='+', type=int, default=[0], help='Demo indices for dataset (for example_batch only)')
    parser.add_argument('--use_wandb', action='store_true', default=True, help='Log to W&B')
    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb', help='Disable W&B logging')
    parser.add_argument('--wandb_init_timeout', type=float, default=300,
                        help='Seconds to wait for wandb.init() before timing out (default 300; increase if you see TimeoutError)')
    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    args = parse_args()
    gpus = args.gpus
    print(f"Using GPUs: {gpus} with n_vals: {args.n_vals}")
    n_vals = args.n_vals
    n_per_gpu = len(n_vals) // len(gpus)
    remainder = len(n_vals) % len(gpus)

    n_vals_per_gpu = []
    start = 0
    for i in range(len(gpus)):
        count = n_per_gpu + (1 if i < remainder else 0)
        n_vals_per_gpu.append(n_vals[start:start + count])
        start += count

    # Pre-generate a single run_id so all processes log to the same run.
    if args.use_wandb:
        args.wandb_run_id = secrets.token_hex(12)
    else:
        args.wandb_run_id = None

    if len(gpus) == 1:
        args.n_vals = n_vals_per_gpu[0]
        args.gpu_id = gpus[0]
        args.wandb_run_created_by_helper = False
        main(args)
    else:
        # Create the wandb run once in a short-lived subprocess to avoid HTTP 409
        # (multiple GPU processes all trying to create the same run_id).
        if args.use_wandb:
            init_timeout = getattr(args, 'wandb_init_timeout', 300)
            creator = mp.Process(
                target=_create_wandb_run_subprocess,
                args=(
                    args.wandb_project,
                    args.wandb_entity,
                    args.wandb_group_name,
                    args.wandb_run_name,
                    args.wandb_run_id,
                    init_timeout,
                ),
            )
            creator.start()
            creator.join(timeout=init_timeout + 30)
            if creator.is_alive():
                creator.terminate()
                creator.join(timeout=5)
                print("Warning: wandb run creator timed out; GPU workers may see resume errors.")
            args.wandb_run_created_by_helper = True

        # Scale down num_parallel_envs per process so total LIBERO workers (n_gpus * num_parallel_envs)
        # stays bounded and we avoid BrokenPipeError from too many nested subprocesses.
        # Each GPU process uses num_parallel_envs // n_gpus (min 1).
        # num_parallel_envs_orig = args.num_parallel_envs
        # args.num_parallel_envs = max(1, args.num_parallel_envs // len(gpus))
        # if args.num_parallel_envs != num_parallel_envs_orig:
            # print(f"Multi-GPU: using num_parallel_envs={args.num_parallel_envs} per process (was {num_parallel_envs_orig}) to avoid pipe contention.")
        # evaluation_libero.py requires (num_eval_episodes - num_video_episodes) % num_parallel_envs == 0.
        # Round down so the assertion holds.
        # num_eval_episodes_orig = args.num_eval_episodes
        # n_eval = args.num_eval_episodes - args.num_video_episodes
        # n_eval = (n_eval // args.num_parallel_envs) * args.num_parallel_envs
        # args.num_eval_episodes = n_eval + args.num_video_episodes
        # if args.num_eval_episodes != num_eval_episodes_orig:
            # print(f"Multi-GPU: adjusted num_eval_episodes to {args.num_eval_episodes} (was {num_eval_episodes_orig}; must be divisible by num_parallel_envs={args.num_parallel_envs}).")

        processes = []
        for gpu_id, n_vals_this_gpu in zip(gpus, n_vals_per_gpu):
            if not n_vals_this_gpu:
                continue
            p = mp.Process(target=run_on_gpu, args=(args, gpu_id, n_vals_this_gpu))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        print("All GPU processes finished.")
