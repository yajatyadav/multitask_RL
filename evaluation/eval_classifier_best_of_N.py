# disable cuda preallocation
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

from utils.flax_utils import restore_agent_with_file, restore_agent_actor_critic_separately
from envs.env_utils import make_env_and_datasets
from agents import agents
from envs.libero_utils import LiberoTopLevelEnvWrapper
from agents.acifql import get_config
import pickle
import copy
import jax
import jax.numpy as jnp
from utils.log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger, get_sample_input_output_log_to_wandb, get_wandb_video
import time
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation_libero import evaluate

import wandb
import tqdm
import numpy as np
import flax
import flax.linen as nn
from utils.networks import MLP
from typing import Sequence
from utils.flax_utils import ModuleDict, TrainState
import optax

from envs.libero_utils import get_dataset as get_libero_dataset
from utils.encoders import encoder_modules


class LoggingHelper:
    def __init__(self, wandb_logger):
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)



seed = 0
horizon_length = 5
discount = 0.99
batch_size = 256
encoder_str = 'image_only_tiny'

keys_to_load = ['agentview_rgb', 'eye_in_hand_rgb', 'language']
NUM_PARALLEL_ENVS = 5
NUM_EVAL_EPISODES = 50
NUM_VIDEO_EPISODES = 5
VIDEO_FRAME_SKIP = 3

from agents.acifql import ACIFQLAgent, get_config as get_acifql_config

class TransClassifier(nn.Module):
    hidden_dims: Sequence[int]
    encoder: nn.Module = None    
    layer_norm: bool = True

    def setup(self):
        self.classifier = MLP((*self.hidden_dims,), activate_final=False, layer_norm=self.layer_norm)
    
    def __call__(self, observations, actions):
        assert actions is not None, "Actions must be provided to the classifier"
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)
        return self.classifier(inputs)

class ClassifierAgent:
    def __init__(self, network, actor_restore_path, example_batch, horizon_length, encoder,num_samples):
        self.network = network
        self.actor_network = self.restore_actor_network(actor_restore_path, copy.deepcopy(example_batch), horizon_length)
        self.config = {"action_dim": example_batch["actions"].shape[-1], "horizon_length": horizon_length, "action_chunking": True, "encoder": encoder, "flow_steps": 10, "num_samples": 10}

    def restore_actor_network(self, actor_restore_path, example_batch, horizon_length):
        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']

        with open(actor_restore_path, 'rb') as f:
            load_dict = pickle.load(f)
        actor_agent_class = ACIFQLAgent
        actor_config = get_acifql_config()
        actor_config['encoder'] = 'combined_encoder_small'
        actor_config['horizon_length'] = horizon_length
        actor_agent = actor_agent_class.create(
            seed=seed,
            ex_observations=copy.deepcopy(ex_observations),
            ex_actions=copy.deepcopy(ex_actions),
            config=actor_config,
        )
        loaded_actor_params = load_dict['agent']['network']['params']
        actor_agent_state_dict = flax.serialization.to_state_dict(actor_agent)
        actor_agent_params = actor_agent_state_dict['network']['params']

        for key in actor_agent_params:
            if 'actor' in key:
                # print(f"for key {key}, copying actor params from {config['actor_restore_path']}")
                actor_agent_params[key] = loaded_actor_params[key]
        actor_agent = flax.serialization.from_state_dict(actor_agent, actor_agent_state_dict)
        actor_network = actor_agent.network
        return actor_network
    
    def sample_actions(
        self,
        observations,
        rng=None,
        temperature=1.0,
    ):
        """Sample actions: actor_network generates actions, classifier used for rejection sampling."""

        if self.actor_network is None:
            raise ValueError("Actor network not found")

        full_action_dim = self.config["action_dim"] * (self.config["horizon_length"] if self.config["action_chunking"] else 1)
        k = sorted(observations.keys())[0]
        noises = jax.random.normal(
            rng,
            (
                observations[k].shape[0],
                self.config['num_samples'],
                full_action_dim,
            ),
        )
        observations = jax.tree_util.tree_map(
                    lambda x: jnp.repeat(x[:, None, ...], self.config["num_samples"], axis=1),
                    observations
                )
        actions = self.compute_flow_actions(observations, noises)
        actions = jnp.clip(actions, -1, 1)

        # Pick the action the classifier is most confident about, as measured by entropy
        pred_lang_logits = self.network.select('classifier')(observations, actions) # has shape (batch, num_samples, num_classes)
        # softmax-normalize and then compute entropy along axis=-1
        # More numerically stable entropy computation
        pred_lang_log_probs = jax.nn.log_softmax(pred_lang_logits, axis=-1)
        pred_lang_probs = jax.nn.softmax(pred_lang_logits, axis=-1)
        entropy = -jnp.sum(pred_lang_probs * pred_lang_log_probs, axis=-1)  # (batch, num_samples)
        
        indices = jnp.argmin(entropy, axis=-1)
        bshape = indices.shape
        indices = indices.reshape(-1)
        bsize = len(indices)
        actions = jnp.reshape(actions, (-1, self.config['num_samples'], full_action_dim))[jnp.arange(bsize), indices, :].reshape(
            bshape + (full_action_dim,))
        return actions

    
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.actor_network.select('actor_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.actor_network.select('actor_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

def build_classifier_agent(classifier_restore_path, env_name, task_name, augmentation_type, augmentation_reward, keys_to_load, num_samples):
    actor_restore_path = 'exp/multitask_RL/bcflowactor_only/libero_90-living_room_scene1/bcflowactor_livingroomscene1__alphabet_soup_ketchup_25_demos_IMAGE_sd00020251227_144347/params_140000.pkl'
    dataset_with_proprio = get_libero_dataset(None, env_name, task_name, augmentation_type, augmentation_reward, keys_to_load + ['proprio'], demo_nums_to_use_per_task=[1], augmentation_dict=None)
    init_batch_for_actor = dataset_with_proprio.sample_sequence(1, sequence_length=horizon_length, discount=discount)

    
    dataset = get_libero_dataset(None, env_name, task_name, augmentation_type, augmentation_reward, keys_to_load, demo_nums_to_use_per_task=[1], augmentation_dict=None)
    example_batch = dataset.sample_sequence(batch_size, sequence_length=horizon_length, discount=discount)
    ex_observations = example_batch['observations']
    ex_actions = example_batch['actions']
    full_actions = jnp.reshape(ex_actions, (ex_actions.shape[0], -1))
    lang_embedding_dim = ex_observations['language'].shape[-1]

    encoder = 'image_only_tiny'
    hidden_dims = (128,)
    layer_norm = True
    lr  = 3e-4

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng, 2)

    classifier_def = TransClassifier(
        encoder=encoder_modules[encoder](),
        hidden_dims=(*hidden_dims, lang_embedding_dim),
        layer_norm=layer_norm,
    )
            
    network_info = dict(
        classifier=(classifier_def, (ex_observations, full_actions)),
    )

    networks = {k: v[0] for k, v in network_info.items()}
    network_args = {k: v[1] for k, v in network_info.items()}

    network_def = ModuleDict(networks)
    network_tx = optax.adam(learning_rate=lr)
    network_params = network_def.init(init_rng, **network_args)['params']
    network = TrainState.create(network_def, network_params, tx=network_tx)

    with open(classifier_restore_path, 'rb') as f:
        load_dict = pickle.load(f)
   
    network_state_dict = flax.serialization.to_state_dict(network)
    network_state_dict['params'] = load_dict
    network = flax.serialization.from_state_dict(network, network_state_dict)

    agent = ClassifierAgent(network, actor_restore_path, init_batch_for_actor, horizon_length, encoder_str, num_samples)
    return agent

def sweep_best_of_N(n_vals, env_name, task_name, classifier_restore_path):  

    env, eval_env, train_dataset, val_dataset, names_to_return = make_env_and_datasets(env_name, task_name, augmentation_type='none', augmentation_reward=False, num_parallel_envs=NUM_PARALLEL_ENVS, keys_to_load=keys_to_load + ['proprio'], use_hardcoded_eval_envs=False)
    example_batch = train_dataset.sample_sequence(batch_size, sequence_length=horizon_length, discount=discount)

     # Setup logging.
    prefixes = ["env", "eval"] + [f"eval_{names_to_return[i]}" for i in range(len(names_to_return))]
    prefixes.append("offline_agent")

    logger = LoggingHelper(
        wandb_logger=wandb,
    )

    infos = {"env_name": env_name, "task_name": task_name, "classifier_restore_path": classifier_restore_path}
    for n in n_vals:
        print(f"Evaluating agent with num_samples: {n}")
        agent = build_classifier_agent(classifier_restore_path, env_name, task_name, augmentation_type='none', augmentation_reward=False, keys_to_load=keys_to_load, num_samples=n)       
        eval_info = eval_agent(agent, eval_env, example_batch, names_to_return, n,logger)
        infos[n] = eval_info
   
   # timestamp the suffix
    suffix = time.strftime("%Y%m%d_%H%M%S")
    with open(f'eval_classifier_best_of_N_infos_{env_name}_{task_name}_{suffix}.pkl', 'wb') as f:
        pickle.dump(infos, f)
    return infos


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



if __name__ == "__main__":   
    # root_dir = '/home/yajatyadav/multitask_reinforcement_learning/multitask_RL/exp/multitask_RL'

    # actor paths
    # one_task_actor_ckpt = 'bcflowactor_only/libero_90-living_room_scene1/bcflowactor_livingroomscene1__alphabet_soup_25_demos_IMAGE_sd00020251227_144306/params_140000.pkl'
    # two_task_actor_ckpt = 'bcflowactor_only/libero_90-living_room_scene1/bcflowactor_livingroomscene1__alphabet_soup_ketchup_25_demos_IMAGE_sd00020251227_144347/params_140000.pkl'

    # critic paths
    # onetask_critic_no_aug_ckpt = 'instruction_following_Q/libero_90-living_room_scene1/libero90_livingroomscene1_singletask_none_augmentation_IMAGE_sd00020251226_214011/params_125000.pkl'
    # two_task_critic_no_aug_ckpt = 'instruction_following_Q/libero_90-living_room_scene1/libero90_livingroomscene1_twotask_none_augmentation_IMAGE_sd00020251226_214200/params_125000.pkl'
    # onetask_critic_aug_one_other_task_ckpt = 'instruction_following_Q/libero_90-living_room_scene1/libero90_livingroomscene1_singletask_one_other_task__augmentation_IMAGE_sd00020251226_215816/params_250000.pkl'
    # two_task_critic_aug_each_other_task_ckpt = 'instruction_following_Q/libero_90-living_room_scene1/libero90_livingroomscene1_twotask_augment_each_other_IMAGE_sd00020251226_215758/params_250000.pkl'

    # TODO(YY): always edit these!!!
    n_vals=[1, 2, 4, 8, 16, 32, 64, 128]
    # actor_ckpt = two_task_actor_ckpt
    # critic_ckpt = two_task_critic_aug_each_other_task_ckpt
    suffix = 'classifier_step_204__actor_25_demos_140k_step__'
    
    # if 'singletask' in critic_ckpt:
        # env_name = 'libero_90-living_room_scene1'
        # task_name = 'pick_up_the_alphabet_soup_and_put_it_in_the_basket'
    # elif 'twotask' in critic_ckpt:
    env_name = 'libero_90-living_room_scene1'
    task_name = 'pick_up_the_alphabet_soup_and_put_it_in_the_basket|pick_up_the_ketchup_and_put_it_in_the_basket'
    # else:
        # raise ValueError(f"Invalid suffix: {suffix}")


    # actor_restore_path = os.path.join(root_dir, actor_ckpt)
    # critic_restore_path = os.path.join(root_dir, critic_ckpt)
    wandb_name = f'eval_libero90_livingroomscene1_{suffix}'
    
    
    wandb.init(
        entity='yajatyadav',
        project='multitask_RL',
        group='eval_libero_classifier_best_of_N',
        name=wandb_name,
    )
    classifier_restore_path = '/home/yajatyadav/multitask_reinforcement_learning/multitask_RL/exp/multitask_RL/living_room_scene1_two_task_classifier/step_204.pkl'
    sweep_best_of_N(n_vals, env_name, task_name, classifier_restore_path)