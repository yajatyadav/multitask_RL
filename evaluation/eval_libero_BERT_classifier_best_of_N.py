import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation_libero import evaluate
os.chdir('/home/yajatyadav/multitask_reinforcement_learning/multitask_RL')
os.environ['MUJOCO_GL'] = 'egl'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

import copy
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
import tqdm
import numpy as np
import math
import random

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field, restore_agent_with_file
import pickle
from utils.networks import MLP
from typing import Sequence


from envs.libero_utils import get_dataset as get_libero_dataset, make_env as make_libero_env

from utils.log_utils import get_wandb_video



class TransClassifier_BERT(nn.Module):
    vision_encoder: nn.Module = None    
    layer_norm: bool = True
    p_drop_state: float = 0.5
    embed_dim: int = 128

    def setup(self):
        # Action encoder: action_dim -> 128
        self.action_encoder = MLP((self.embed_dim, self.embed_dim), activate_final=True, layer_norm=self.layer_norm)
        
        # Language encoder: 768 (BERT) -> 128
        self.lang_encoder = MLP((self.embed_dim, self.embed_dim), activate_final=True, layer_norm=self.layer_norm)
        
        # Final classifier: 384 (128*3) -> 128 -> 128 -> 1
        self.classifier = MLP((self.embed_dim, self.embed_dim, 1), activate_final=False, layer_norm=self.layer_norm)
    
    def __call__(self, observations, actions, language_embedding, train=True, rng=None):
        """
        Args:
            observations: (batch, obs_dim)
            actions: (batch, action_dim)
            language_embedding: (batch, 768) - BERT embedding
            train: whether to apply dropout
            rng: random key for dropout
            
        Returns:
            logits: (batch, 1) - scalar logit for P(lang | obs, action)
        """
        assert actions is not None, "Actions must be provided to the classifier"
        assert self.vision_encoder is not None, "Encoder must be provided to the classifier"
        
        # Encode observations -> (batch, 128)
        obs_encoded = self.vision_encoder(observations)

        # Dropout on obs encoding
        if rng is None:
            rng = jax.random.PRNGKey(0)
        mask = jax.random.bernoulli(rng, 1 - self.p_drop_state, shape=(obs_encoded.shape[0], 1))
        obs_encoded = jax.lax.cond(
            train,
            lambda x: mask * x,
            lambda x: x,
            obs_encoded
        )
        
        # Encode actions -> (batch, 128)
        action_encoded = self.action_encoder(actions)
        
        # Encode language -> (batch, 128)
        lang_encoded = self.lang_encoder(language_embedding)
        
        # Concatenate all three -> (batch, 384)
        inputs = jnp.concatenate([obs_encoded, action_encoded, lang_encoded], axis=-1)
        
        # Final classification -> (batch, 1)
        return self.classifier(inputs)


import pickle
import json
from pathlib import Path
import time
from agents.acbcflowactor import ACBCFlowActorAgent, get_config as get_actor_config

ACTOR_SEED = 0
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

def load_classifier(save_dir):
    """Load classifier from saved directory."""
    save_dir = Path(save_dir)
    
    # Load hyperparameters
    with open(save_dir / 'hparams.json', 'r') as f:
        hparams = json.load(f)
    
    # Load parameters
    with open(save_dir / 'params.pkl', 'rb') as f:
        params = pickle.load(f)
    
    # Recreate model definition
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

class ClassifierAgent:
    def __init__(self, classifier_network_restore_path, actor_restore_path, example_batch, horizon_length, actor_encoder, num_samples):
        class_net, class_hparams = load_classifier(classifier_network_restore_path)
        self.classifier_network = class_net
        self.actor_network = self.restore_actor_network(actor_restore_path, copy.deepcopy(example_batch), horizon_length, actor_encoder)
        self.config = {"action_dim": example_batch["actions"].shape[-1], "horizon_length": horizon_length, "action_chunking": True, "actor_encoder": actor_encoder, "flow_steps": 10, "num_samples": num_samples}

    def restore_actor_network(self, actor_restore_path, example_batch, horizon_length, actor_encoder):
        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']

        with open(actor_restore_path, 'rb') as f:
            load_dict = pickle.load(f)
        actor_agent_class = ACBCFlowActorAgent
        actor_config = get_actor_config()
        actor_config['encoder'] = actor_encoder
        actor_config['horizon_length'] = horizon_length
        actor_agent = actor_agent_class.create(
            seed=ACTOR_SEED,
            ex_observations=copy.deepcopy(ex_observations),
            ex_actions=copy.deepcopy(ex_actions),
            config=actor_config,
        )
        loaded_actor_params = load_dict['agent']['network']['params']
        actor_agent_state_dict = flax.serialization.to_state_dict(actor_agent)
        actor_agent_params = actor_agent_state_dict['network']['params']

        for key in actor_agent_params:
            if 'actor' in key:
                print(f"for key {key}, copying actor params from actor_restore_path")
                actor_agent_params[key] = loaded_actor_params[key]
        actor_agent = flax.serialization.from_state_dict(actor_agent, actor_agent_state_dict)
        actor_network = actor_agent.network
        return actor_network
    
    def sample_actions(
        self,
        observations,
        rng=None,
        temperature=1.0,
        print_debug=False,
    ):
        """Sample actions: actor_network generates actions, classifier used for rejection sampling."""

        if self.actor_network is None:
            raise ValueError("Actor network not found")

        full_action_dim = self.config["action_dim"] * (self.config["horizon_length"] if self.config["action_chunking"] else 1)
        k = sorted(observations.keys())[0]
        batch_size = observations[k].shape[0]
        
        # Get language embedding before repeating observations
        # lang_embedding = observations.pop('language')  # (batch, 768)
        lang_embedding = observations.get('language')
        
        if print_debug:
            print(f"[DEBUG] batch_size: {batch_size}, num_samples: {self.config['num_samples']}")
            print(f"[DEBUG] lang_embedding shape: {lang_embedding.shape}")
        
        noises = jax.random.normal(
            rng,
            (
                batch_size,
                self.config['num_samples'],
                full_action_dim,
            ),
        )
        observations = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x[:, None, ...], self.config["num_samples"], axis=1),
            observations
        )
        actions = self.compute_flow_actions(observations, noises)  # (batch, num_samples, full_action_dim)
        actions = jnp.clip(actions, -1, 1)

        if print_debug:
            print(f"[DEBUG] actions shape: {actions.shape}")
            print(f"[DEBUG] actions range: [{float(actions.min()):.4f}, {float(actions.max()):.4f}]")

        # Flatten everything for classifier: (batch * num_samples, ...)
        # before feeding into classifier, drop the proprio to be extra safe (classifier was trained without it)
        observations.pop('proprio')
        flat_obs = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]),
            observations
        )
        flat_actions = jnp.reshape(actions, (-1, full_action_dim))
        flat_lang = jnp.repeat(lang_embedding, self.config['num_samples'], axis=0)
        
        if print_debug:
            print(f"[DEBUG] flat_obs shape: {flat_obs[sorted(flat_obs.keys())[0]].shape}")
            print(f"[DEBUG] flat_actions shape: {flat_actions.shape}")
            print(f"[DEBUG] flat_lang shape: {flat_lang.shape}")
        
        # Get classifier logits
        logits = self.classifier_network.select('classifier')(
            flat_obs, flat_actions, flat_lang, train=False
        )  # (batch * num_samples, 1)
        
        # Reshape logits to (batch, num_samples)
        logits = jnp.reshape(logits, (batch_size, self.config['num_samples']))
        
        if print_debug:
            print(f"~~~~~~~~~~~~~~[DEBUG] logits shape: {logits.shape}~~~~~~~~~~~~~~")
            print(f"[DEBUG] logits range: [{float(logits.min()):.4f}, {float(logits.max()):.4f}]")
            print(f"[DEBUG] logits mean: {float(logits.mean()):.4f}, std: {float(logits.std()):.4f}")
            print(f"[DEBUG] logits per batch element - min: {logits.min(axis=-1)}, max: {logits.max(axis=-1)}")
        
        # Pick action with highest logit for each batch element
        indices = jnp.argmax(logits, axis=-1)  # (batch,)
        
        # Get the logits of selected actions
        selected_logits = logits[jnp.arange(batch_size), indices]
        
        if print_debug:
            print(f"[DEBUG] selected indices: {indices}")
            print(f"[DEBUG] selected logits: {selected_logits}")
            print(f"[DEBUG] selected logits min: {selected_logits.min(axis=-1)}, max: {selected_logits.max(axis=-1)}")
            print(f"[DEBUG] selected logits mean: {float(selected_logits.mean()):.4f}")
        
        # Select best actions
        actions = actions[jnp.arange(batch_size), indices, :]  # (batch, full_action_dim)
        
        if print_debug:
            print(f"[DEBUG] final actions shape: {actions.shape}")
            print(f"[DEBUG] final actions range: [{float(actions.min()):.4f}, {float(actions.max()):.4f}]")
        
        return actions

    
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['actor_encoder'] is not None:
            observations = self.actor_network.select('actor_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.actor_network.select('actor_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

NUM_EVAL_EPISODES = 52
NUM_VIDEO_EPISODES = 5
NUM_PARALLEL_ENVS = 2
VIDEO_FRAME_SKIP = 3
def eval_agent(agent, eval_env, example_batch, names_to_return, n, logger):
    print(f"Evaluating agent on {len(eval_env)} environments")
    all_eval_info = []
    for j, eval_env_j in tqdm.tqdm(enumerate(eval_env), total=len(eval_env), desc="Evaluating multi-task", position=0,leave=False):
        eval_info, trajs, renders = evaluate(
            agent=agent, 
            env=eval_env_j, 
            action_dim=example_batch["actions"].shape[-1], 
            num_eval_episodes=NUM_EVAL_EPISODES, 
            num_video_episodes=NUM_VIDEO_EPISODES, 
            num_parallel_envs=NUM_PARALLEL_ENVS, 
            video_frame_skip=VIDEO_FRAME_SKIP)
        all_eval_info.append(eval_info)
        if len(renders) > 0:
            # value_and_reward_visualization(trajs, agent, FLAGS.save_dir, log_step)
            eval_info['video'] = get_wandb_video(renders)
        logger.log(eval_info, f"eval_{names_to_return[j]}", step=n)
        # remove video before taking mean
        if 'video' in eval_info:
            del eval_info['video']

    # aggregate eval info via mean, then log under "eval" prefix
    eval_info = {k: np.mean([eval_info[k] for eval_info in all_eval_info]) for k in all_eval_info[0].keys()}
    logger.log(eval_info, "eval", step=n)
    print(f"Eval info: {eval_info}")
    return eval_info


from envs.env_utils import make_env_and_datasets
import wandb

if __name__ == "__main__":
    task_name = ''
    env_name = 'libero_90-living_room_scene1-pick_up_the_alphabet_soup_and_put_it_in_the_basket|libero_90-living_room_scene1-pick_up_the_ketchup_and_put_it_in_the_basket|libero_goal-open_the_middle_drawer_of_the_cabinet|libero_goal-turn_on_the_stove|libero_spatial-pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate|libero_spatial-pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate'
    
    augmentation_type = 'none'
    augmentation_reward = False
    
    language_embedder = 'bert'
    keys_to_load = ['agentview_rgb', 'eye_in_hand_rgb', 'language', 'proprio']
    
    horizon_length = 5
    discount = 0.99
    actor_encoder = 'combined_encoder_small'

    log_this = True
    if log_this:
        wandb.init(project="multitask_RL", entity="yajatyadav", group="eval_libero_bert_classifier_6task", name=f"25_demo_BERT_actor_2_epoch_p_drop_state_0.5_classifier")
        logger = LoggingHelper(
        wandb_logger=wandb,
        )
    else:
        logger = LoggingHelper(
            wandb_logger=None,
            to_log=False,
        )

    _, eval_env, dataset, _, names_to_return = make_env_and_datasets(env_name, task_name, language_embedder, augmentation_type, augmentation_reward, num_parallel_envs=NUM_PARALLEL_ENVS, keys_to_load=keys_to_load + ['proprio'], demo_nums_to_use_per_task=[0], augmentation_dict=None, is_notebook=False)
    prefixes = ["env", "eval"] + [f"eval_{names_to_return[i]}" for i in range(len(names_to_return))]
    prefixes.append("offline_agent")
    
    classifier_restore_path = '/home/yajatyadav/multitask_reinforcement_learning/checkpoints/libero_bert_classifier_6task_p_drop_state_0.5_2_epoch'
    actor_restore_path = '/home/yajatyadav/multitask_reinforcement_learning/multitask_RL/exp/multitask_RL/bcflowactor_BERT/bcflowactor_6_tasks_BERT_25_demos_IMAGE_sd00020260128_174608/params_20000.pkl'
    example_batch = dataset.sample_sequence(1, sequence_length=horizon_length, discount=discount)

    for N in [256]:
        print(f"Evaluating with N = {N}")
        classifier_agent = ClassifierAgent(classifier_restore_path, actor_restore_path, example_batch, horizon_length, actor_encoder, N)
        eval_info = eval_agent(classifier_agent, eval_env, example_batch, names_to_return, n=N, logger=logger)   