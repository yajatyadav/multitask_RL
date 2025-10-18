import os
import sys

# CRITICAL: Set this BEFORE any mujoco/libero imports for GPU rendering
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

repo_root_dir = os.getenv("MULTITASK_RL_REPO_ROOT_DIR", os.getcwd())
sys.path.insert(0, os.path.join(repo_root_dir, "libero"))

import logging
import math
import pathlib
import numpy as np
import tqdm
from dataclasses import dataclass
from collections import defaultdict
import jax

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs import SubprocVectorEnv  # NEW
from utils.data_utils import get_language_encoder
from utils.data_utils import normalize_libero_eval_obs_for_agent, unnormalize_action_min_max

from utils.data_utils import LIBERO_ENV_RESOLUTION
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

@dataclass
class Args:
    eval_use_images: bool = False # set to False for state-based evals
    seed: int = 0
    num_eval_episodes: int = 20
    num_video_epsiodes: int = 5 # render only the first 5 episodes
    num_steps_wait: int = 3  # Reduced from 10 for speed
    video_frame_skip: int = 3
    eval_temperature: float = 1.0
    task_suite_name: str = "libero_10"
    task_name: str = ""
    dataset_name: str = "" # used to fetch norm stats
    text_encoder: str = "one_hot_libero"
    num_parallel_envs: int = 4  # NEW: number of parallel environments


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "has_renderer": False,  # NEW: disable on-screen rendering
        "has_offscreen_renderer": True,  # NEW: enable GPU offscreen rendering
        "use_camera_obs": True,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def setup_vectorized_eval_env(args: Args):
    """Setup vectorized environments for parallel evaluation"""
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    max_steps_dict = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    if args.task_suite_name not in max_steps_dict:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    max_steps = max_steps_dict[args.task_suite_name]

    # Find the task
    task = None
    task_description = None
    initial_states = None
    for task_id in range(num_tasks_in_suite):
        task = task_suite.get_task(task_id)
        if task.name == args.task_name:
            task_description = task.language
            initial_states = task_suite.get_task_init_states(task_id)
            break
    
    if task is None:
        raise ValueError(f"Task {args.task_name} not found in task suite {args.task_suite_name}")

    # Calculate number of parallel envs (don't create more than we need)
    num_parallel = min(args.num_parallel_envs, args.num_eval_episodes)

    # Create environment factory functions
    def make_env(seed_offset):
        def _init():
            env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed + seed_offset)
            return env
        return _init

    # Create vectorized environment
    env_fns = [make_env(i) for i in range(num_parallel)]
    vec_env = SubprocVectorEnv(env_fns)
    
    return vec_env, max_steps, task_description, initial_states, num_parallel


def format_libero_obs_for_agent_batch(obs_batch, task_embedding, sim_states, dataset_name, eval_use_images: bool = False):
    """Format a batch of observations for the agent"""
    batch_size = len(obs_batch)
    
    # Process proprio for each observation
    proprio_list = []
    for obs in obs_batch:
        proprio = np.concatenate((
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        ), dtype=np.float32)
        proprio_list.append(proprio)
    
    proprio_batch = np.stack(proprio_list)
    sim_state_batch = np.array(sim_states, dtype=np.float32)
    
    # Repeat task embedding for the entire batch
    task_embedding_batch = np.repeat(task_embedding, batch_size, axis=0)
    
    obs_to_normalize = {
        'task_embedding': task_embedding_batch,
        'proprio': proprio_batch,
        'sim_state': sim_state_batch,
    }
    
    if eval_use_images:
        image_primary_list = []
        image_wrist_list = []
        for obs in obs_batch:
            image_primary = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            image_wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            image_primary_list.append(image_primary)
            image_wrist_list.append(image_wrist)
        
        obs_to_normalize.update({
            'image_primary': np.stack(image_primary_list),
            'image_wrist': np.stack(image_wrist_list),
        })
    
    normalized_obs = normalize_libero_eval_obs_for_agent(obs_to_normalize, dataset_name=dataset_name)
    return normalized_obs


def format_agent_actions_for_libero_batch(actions_batch, dataset_name, is_gripper_closed_list, num_consecutive_gripper_change_list, STICKY_GRIPPER_NUM_STEPS=1):
    """Format a batch of actions for libero - processes all at once for efficiency"""
    # Unnormalize all actions at once (expects batch dimension)
    actions_batch = np.array(actions_batch, dtype=np.float32)
    if actions_batch.ndim == 1:
        actions_batch = actions_batch[np.newaxis, :]  # Add batch dim if needed
    
    actions_unnormalized = unnormalize_action_min_max(actions_batch, dataset_name=dataset_name)
    actions_unnormalized = np.clip(actions_unnormalized, -1, 1)
    
    # Process gripper logic for each action in the batch
    formatted_actions = []
    new_gripper_closed = []
    new_consecutive_changes = []
    
    for i, action in enumerate(actions_unnormalized):
        is_gripper_closed = is_gripper_closed_list[i]
        num_consecutive = num_consecutive_gripper_change_list[i]
        
        # Sticky gripper logic
        if (action[-1] > 0) != is_gripper_closed:
            num_consecutive += 1
        else:
            num_consecutive = 0
        
        if num_consecutive >= STICKY_GRIPPER_NUM_STEPS:
            is_gripper_closed = not is_gripper_closed
            num_consecutive = 0
        
        action[-1] = 1.0 if is_gripper_closed else -1.0
        
        formatted_actions.append(action)
        new_gripper_closed.append(is_gripper_closed)
        new_consecutive_changes.append(num_consecutive)
    
    return np.array(formatted_actions), new_gripper_closed, new_consecutive_changes


def evaluate(agent, args: Args):
    """Vectorized evaluation using parallel environments"""
    # Setup vectorized environment
    vec_env, max_steps, task_description, initial_states, num_parallel = setup_vectorized_eval_env(args)
    
    # Setup text encoder
    TEXT_ENCODER = get_language_encoder(args.text_encoder)
    task_embedding = TEXT_ENCODER.encode([task_description])
    
    # Setup RNG state - managed outside JIT
    rng = jax.random.PRNGKey(np.random.randint(0, 2**31 - 1))
    
    # Track results
    all_stats = []
    all_renders = []
    trajs = []  # Keep empty for compatibility
    wrist_renders = []  # Keep empty for compatibility
    
    # Process episodes in batches
    num_batches = (args.num_eval_episodes + num_parallel - 1) // num_parallel
    
    for batch_idx in tqdm.tqdm(range(num_batches), desc=f"Evaluating {task_description}"):
        # Determine how many episodes in this batch
        start_episode = batch_idx * num_parallel
        end_episode = min(start_episode + num_parallel, args.num_eval_episodes)
        batch_size = end_episode - start_episode
        
        # Get the environments we'll use for this batch
        env_ids = list(range(batch_size))
        
        # Initialize environments with different initial states
        init_states_batch = [initial_states[start_episode + i] for i in range(batch_size)]
        obs_batch = vec_env.set_init_state(init_states_batch, id=env_ids)
        
        # Initialize per-env state
        is_gripper_closed = [False] * batch_size
        num_consecutive_gripper_change = [0] * batch_size
        episode_done = [False] * batch_size
        renders_batch = [[] for _ in range(batch_size)]
        episode_stats = [None] * batch_size
        
        # Dummy steps for stabilization - check for early termination
        dummy_actions = np.array([LIBERO_DUMMY_ACTION] * batch_size)
        for step_idx in range(args.num_steps_wait):
            # Only step environments that haven't finished yet
            active_dummy_envs = [i for i in env_ids if not episode_done[i]]
            if not active_dummy_envs:
                break
            
            dummy_actions_active = dummy_actions[active_dummy_envs]
            obs_result, _, dones_dummy, infos_dummy = vec_env.step(dummy_actions_active, id=active_dummy_envs)
            
            # Update obs_batch and check for early completion
            for idx, env_idx in enumerate(active_dummy_envs):
                obs_batch[env_idx] = obs_result[idx]
                if dones_dummy[idx]:
                    episode_done[env_idx] = True
                    episode_stats[env_idx] = flatten(infos_dummy[idx])
                    episode_stats[env_idx]['success'] = 1.0
        
        # Get sim states only for active environments
        sim_states = vec_env.get_sim_state()
        
        # Main evaluation loop
        for t in range(max_steps):
            # Get active environments (not done yet)
            active_envs = [i for i in range(batch_size) if not episode_done[i]]
            if not active_envs:
                break
            
            # Collect renders if needed (before normalization modifies obs in-place)
            for env_idx in active_envs:
                episode_num = start_episode + env_idx
                if episode_num < args.num_video_epsiodes and (t % args.video_frame_skip == 0):
                    renders_batch[env_idx].append(obs_batch[env_idx]["agentview_image"][::-1, ::-1].copy())
            
            # Format observations for active environments
            active_obs = [obs_batch[i] for i in active_envs]
            active_sim_states = [sim_states[i] for i in active_envs]
            
            obs_for_actor = format_libero_obs_for_agent_batch(
                active_obs, task_embedding, active_sim_states, 
                args.dataset_name, args.eval_use_images
            )
            
            # Split RNG and get actions - RNG management outside JIT
            rng, key = jax.random.split(rng)
            actions_normalized = agent.sample_actions(
                observations=obs_for_actor, 
                temperature=args.eval_temperature,
                seed=key
            )
            
            # Format actions in batch (much more efficient)
            active_gripper_closed = [is_gripper_closed[i] for i in active_envs]
            active_consecutive_changes = [num_consecutive_gripper_change[i] for i in active_envs]
            
            actions, new_gripper_closed, new_consecutive_changes = format_agent_actions_for_libero_batch(
                actions_normalized,
                args.dataset_name,
                active_gripper_closed,
                active_consecutive_changes
            )
            
            # Update gripper state for active environments
            for idx, env_idx in enumerate(active_envs):
                is_gripper_closed[env_idx] = new_gripper_closed[idx]
                num_consecutive_gripper_change[env_idx] = new_consecutive_changes[idx]
            
            # Step ONLY active environments
            next_obs_batch, rewards, dones, infos = vec_env.step(actions, id=active_envs)
            sim_states_all = vec_env.get_sim_state()
            
            # Update observations and check for completion
            for idx, env_idx in enumerate(active_envs):
                obs_batch[env_idx] = next_obs_batch[idx]
                sim_states[env_idx] = sim_states_all[env_idx]
                
                if dones[idx]:
                    episode_done[env_idx] = True
                    episode_stats[env_idx] = flatten(infos[idx])
                    episode_stats[env_idx]['success'] = 1.0
        
        # Collect results for all episodes in this batch
        for env_idx in range(batch_size):
            # Handle episodes that didn't finish
            if episode_stats[env_idx] is None:
                episode_stats[env_idx] = {'success': 0.0}
            
            all_stats.append(episode_stats[env_idx])
            all_renders.append(np.array(renders_batch[env_idx]) if renders_batch[env_idx] else np.array([]))
    
    # Close vectorized environment
    vec_env.close()
    
    # Aggregate statistics across all episodes
    aggregated_stats = defaultdict(list)
    for stat in all_stats:
        for k, v in stat.items():
            aggregated_stats[k].append(v)
    
    stats = {k: np.mean(v) for k, v in aggregated_stats.items()}
    
    return stats, trajs, all_renders, wrist_renders