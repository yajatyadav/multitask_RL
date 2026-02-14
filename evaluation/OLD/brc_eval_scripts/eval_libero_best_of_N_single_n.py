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
import argparse

import time
import sys
sys.path.insert(0, os.getcwd())
from evaluation_libero import evaluate

import tqdm
import numpy as np

agent_class_str = 'acifql'
agent_class = agents[agent_class_str]

seed = 0
horizon_length = 5
discount = 0.99
batch_size = 256
encoder = 'combined_encoder_small'
KEYS_TO_LOAD = ['agentview_rgb', 'eye_in_hand_rgb', 'proprio', 'language']
NUM_PARALLEL_ENVS = 5
NUM_EVAL_EPISODES = 50
NUM_VIDEO_EPISODES = 5  # Save videos for postprocessing
VIDEO_FRAME_SKIP = 3


def evaluate_single_n(n, env_name, task_name, actor_restore_path, critic_restore_path, 
                     wandb_entity, wandb_project, wandb_run_id, output_dir):
    """
    Evaluate agent for a single n value.
    
    NOTE: This version does NOT log to wandb during evaluation.
    Results are saved to pickle files and logged by the postprocessing script.
    """
    
    print(f"\n{'='*80}")
    print(f"Starting evaluation for n={n}")
    print(f"{'='*80}\n")
    
    # Setup environment and datasets
    env, eval_env, train_dataset, val_dataset, names_to_return = make_env_and_datasets(
        env_name, task_name, augmentation_type='none', augmentation_reward=0.0,
        num_parallel_envs=NUM_PARALLEL_ENVS, keys_to_load=KEYS_TO_LOAD, 
        use_hardcoded_eval_envs=False)
    
    example_batch = train_dataset.sample_sequence(
        batch_size, sequence_length=horizon_length, discount=discount)

    # Create and restore agent
    print(f"Creating agent with num_samples: {n}")
    agent_config = get_config()
    agent_config['encoder'] = encoder
    agent_config['num_samples'] = n
    agent_config['horizon_length'] = horizon_length
    agent = agent_class.create(seed, example_batch['observations'], 
                               example_batch['actions'], agent_config)

    print(f"Restoring agent from:")
    print(f"  Actor:  {actor_restore_path}")
    print(f"  Critic: {critic_restore_path}")
    critic_restore_ckpt_number = int(os.path.basename(critic_restore_path).split('.')[0].split('_')[-1])
    agent = restore_agent_actor_critic_separately(agent, actor_restore_path, critic_restore_path)
    
    # Evaluate
    print(f"\nEvaluating on {len(eval_env)} tasks...")
    all_eval_info = []
    all_renders = []  # Store renders for video generation
    for j, eval_env_j in tqdm.tqdm(enumerate(eval_env), total=len(eval_env), 
                                   desc=f"Evaluating n={n}", position=0, leave=True):
        eval_info, trajs, renders = evaluate(
            agent=agent, 
            env=eval_env_j, 
            action_dim=example_batch["actions"].shape[-1], 
            num_eval_episodes=NUM_EVAL_EPISODES, 
            num_video_episodes=NUM_VIDEO_EPISODES,
            num_parallel_envs=NUM_PARALLEL_ENVS, 
            video_frame_skip=VIDEO_FRAME_SKIP)
        all_eval_info.append(eval_info)
        all_renders.append(renders)  # Save renders for postprocessing
        print(f"  Task {j} ({names_to_return[j]}): success={eval_info.get('success', -1):.3f}, videos={len(renders)}")

    # Aggregate eval info
    eval_info = {k: np.mean([info[k] for info in all_eval_info]) for k in all_eval_info[0].keys()}
    print(f"\nAggregated results for n={n}:")
    for k, v in eval_info.items():
        print(f"  {k}: {v:.4f}")
    
    # Save results to pickle
    infos = {
        "env_name": env_name, 
        "task_name": task_name, 
        "actor_restore_path": actor_restore_path, 
        "critic_restore_path": critic_restore_path,
        "n": n,
        "eval_info": eval_info,
        "per_task_eval_info": {names_to_return[j]: all_eval_info[j] for j in range(len(all_eval_info))},
        "video_renders": {names_to_return[j]: all_renders[j] for j in range(len(all_renders))},  # Save renders
        "task_names": names_to_return,  # Save task names for postprocessing
        "critic_ckpt_number": critic_restore_ckpt_number,
        "wandb_run_id": wandb_run_id,
        "timestamp": time.time(),
    }
    
    os.makedirs(output_dir, exist_ok=True)
    suffix = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'eval_libero_best_of_N_n{n}_{env_name}_{suffix}.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(infos, f)
    
    # Get file size for reporting
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    
    print(f"\n✅ Results saved to: {output_file}")
    print(f"   File size: {file_size_mb:.1f} MB (includes {NUM_VIDEO_EPISODES} videos per task)")
    print(f"📊 Results will be logged to wandb by postprocessing job")
    print(f"   Run ID: {wandb_run_id}")
    print(f"\n{'='*80}\n")
    
    return eval_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate agent for a single n value (parallel-safe version)'
    )
    
    # Required arguments
    parser.add_argument('--n', type=int, required=True, 
                       help='Number of samples for best-of-N evaluation')
    parser.add_argument('--actor_restore_path', type=str, required=True,
                       help='Path to actor checkpoint')
    parser.add_argument('--critic_restore_path', type=str, required=True,
                       help='Path to critic checkpoint')
    
    # Environment arguments
    parser.add_argument('--env_name', type=str, 
                       default='libero_90-living_room_scene1',
                       help='Environment name')
    parser.add_argument('--task_name', type=str,
                       default='pick_up_the_alphabet_soup_and_put_it_in_the_basket|pick_up_the_ketchup_and_put_it_in_the_basket',
                       help='Task name(s), pipe-separated for multi-task')
    
    # Wandb arguments (for postprocessing)
    parser.add_argument('--wandb_entity', type=str, default='yajatyadav',
                       help='Wandb entity')
    parser.add_argument('--wandb_project', type=str, default='multitask_RL',
                       help='Wandb project')
    parser.add_argument('--wandb_run_id', type=str, required=True,
                       help='Wandb run id (for postprocessing)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    evaluate_single_n(
        n=args.n,
        env_name=args.env_name,
        task_name=args.task_name,
        actor_restore_path=args.actor_restore_path,
        critic_restore_path=args.critic_restore_path,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_run_id=args.wandb_run_id,
        output_dir=args.output_dir  
    )