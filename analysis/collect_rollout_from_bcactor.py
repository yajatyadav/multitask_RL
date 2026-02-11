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
import pickle
import flax
import copy
import jax
import jax.numpy as jnp
from agents.acbcflowactor import ACBCFlowActorAgent, get_config as get_actor_config
from envs.env_utils import make_env_and_datasets
from evaluation_libero import evaluate
import argparse
from pathlib import Path
# save eval_info, trajs, renders
import time
import pickle
import os
import json
import shutil
import random


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor_name', type=str)
    parser.add_argument('--actor_path', type=str)
    parser.add_argument('--env_name', type=str, help='a string that will control which environments to collect rollouts in.')
    parser.add_argument('--num_rollouts', type=int, help='number of rollouts to collect for each environment.')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0], help='list of GPUs, e.g. --gpus 0 1 2')    

    parser.add_argument('--num_parallel_envs', type=int, default=5, help='number of parallel environments to use for evaluation.')    
    parser.add_argument('--task_name', type=str, default='', help='used in certain cases along with env_name to control how many tasks within a suite to collect rollouts in.')
    parser.add_argument('--save_dir', type=str, default='/home/yajatyadav/multitask_reinforcement_learning/multitask_RL/bcactor_collected_rollouts/')   
    parser.add_argument('--horizon_length', type=int, default=5)
    parser.add_argument('--actor_encoder', type=str, default='combined_encoder_small')   
    parser.add_argument('--language_embedder', type=str, default='bert')
    parser.add_argument('--actor_seed', type=int, default=0, help='seed for initializing the actor network.')
    return parser.parse_args()

def restore_actor_network(actor_restore_path, example_batch, horizon_length, actor_encoder, ACTOR_SEED):
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
    return actor_agent


def main(args):
    gpu_id = args.gpu_id
    root_dir = Path(args.save_dir) / args.actor_name
    root_dir.mkdir(parents=True, exist_ok=True)
    # dump args to a json file
    with open(root_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f)
    
    # first, simply instnatiate the actor and collect rollouts
    ACTOR_SEED = args.actor_seed

    horizon_length = args.horizon_length
    actor_encoder = args.actor_encoder
    env_name = args.env_name
    task_name = args.task_name
    language_embedder = args.language_embedder
    NUM_ROLLOUTS = args.num_rollouts
    NUM_PARALLEL_ENVS = args.num_parallel_envs

    
    keys_to_load = ['language', 'proprio', 'agentview_rgb', 'eye_in_hand_rgb']
    discount = 0.99
    augmentation_type = 'none'
    augmentation_reward = False
    batch_level_sampling = True
    _, eval_env, dataset, _ = make_env_and_datasets(env_name, task_name, language_embedder, augmentation_type, augmentation_reward, batch_level_sampling, num_parallel_envs=NUM_PARALLEL_ENVS, keys_to_load=keys_to_load, demo_nums_to_use_per_task=[0], augmentation_dict=None, is_notebook=True)


    example_batch = dataset.sample_sequence(1, sequence_length=horizon_length, discount=discount)
    actor_agent = restore_actor_network(args.actor_path, example_batch, horizon_length, actor_encoder, ACTOR_SEED)

    for (eval_env_j, eval_env_j_name) in eval_env:
        save_dir = Path(root_dir) / eval_env_j_name
        save_dir.mkdir(parents=True, exist_ok=True)
        eval_info, trajs, renders = evaluate(
            agent=actor_agent,
            env=eval_env_j,
            action_dim=example_batch["actions"].shape[-1],
            num_eval_episodes=NUM_ROLLOUTS,
            num_video_episodes=0,
            num_parallel_envs=NUM_PARALLEL_ENVS,
            video_frame_skip=3,
        )
        #
        suffix = 'gpu' + str(gpu_id) + '_' + time.strftime("%Y%m%d_%H%M%S") + '___' + str(
            random.randint(0, 1000000)
        )
        with open(save_dir / f'eval_info_{suffix}.pkl', 'wb') as f:
            pickle.dump(eval_info, f)
        with open(save_dir / f'trajs_{suffix}.pkl', 'wb') as f:
            pickle.dump(trajs, f)
        with open(save_dir / f'renders_{suffix}.pkl', 'wb') as f:
            pickle.dump(renders, f)
        print(f"Saved eval_info, trajs, renders to {save_dir}")

def run_on_gpu(args, gpu_id, num_rollouts_for_this_gpu):
    """Wrapper that sets GPU env vars before importing JAX, then runs main."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['EGL_DEVICE_ID'] = str(gpu_id)
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(gpu_id)
    os.environ['MUJOCO_GL'] = 'egl'

    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
    xla_flags = os.environ.get('XLA_FLAGS', '')
    xla_flags += ' --xla_gpu_triton_gemm_any=True'
    os.environ['XLA_FLAGS'] = xla_flags
    
    # Modify args for this process
    args.num_rollouts = num_rollouts_for_this_gpu
    args.actor_seed = args.actor_seed + gpu_id  # different seed per GPU
    args.gpu_id = gpu_id
    main(args)

import multiprocessing as mp
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Required for CUDA
    
    args = args_parser()
    gpus = args.gpus
    
    total_rollouts = args.num_rollouts
    num_per_gpu = total_rollouts // len(gpus)
    remainder = total_rollouts % len(gpus)
    
    processes = []
    for i, gpu in enumerate(gpus):
        # Distribute remainder across first few GPUs
        rollouts_this_gpu = num_per_gpu + (1 if i < remainder else 0)
        p = mp.Process(target=run_on_gpu, args=(copy.deepcopy(args), gpu, rollouts_this_gpu))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()    
    print("All GPU processes finished.")