#!/usr/bin/env python3
"""
Fast dataset creation from pickle rollouts using multiprocessing across tasks.

Usage:
    python test_iterate_over_rollout_dataset.py \
        --rollouts_dir /path/to/pickle/rollouts \
        --num_tasks 90 \
        --num_demos 60
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
import argparse
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count


def load_pickle(path):
    """Load a single pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def process_single_task(args):
    """Process a single task - designed for multiprocessing."""
    task_dir, demo_nums_to_use, action_clip_eps = args
    task_name = task_dir.name
    pkl_files = sorted(task_dir.glob('trajs_*.pkl'))
    
    if not pkl_files:
        return None, f"No pickle files for {task_name}"
    
    train_indices = set(demo_nums_to_use)
    
    # Load all pickle files
    t0 = time.time()
    all_datasets = [load_pickle(p) for p in pkl_files]
    load_time = time.time() - t0
    
    # Count total size and gather demo info
    demo_num = -1
    total_timesteps = 0
    demos_info = []
    
    for dataset_idx, dataset in enumerate(all_datasets):
        for ep_idx, ep in enumerate(dataset):
            demo_num += 1
            if demo_num not in train_indices:
                continue
            timesteps = len(ep['actions'])
            total_timesteps += timesteps
            demos_info.append((dataset_idx, ep_idx, timesteps))
    
    if total_timesteps == 0:
        return None, f"No data for {task_name}"
    
    # Get shapes from first episode
    first_ep = all_datasets[demos_info[0][0]][demos_info[0][1]]
    first_obs = first_ep['observations'][0]
    obs_keys = list(first_obs.keys())
    action_dim = len(first_ep['actions'][0])
    lang_emb_dim = len(first_ep['language_embedding'][0])
    
    # Pre-allocate arrays
    t0 = time.time()
    obs_arrays = {}
    next_obs_arrays = {}
    for k in obs_keys:
        shape = first_obs[k].shape
        dtype = first_obs[k].dtype
        obs_arrays[k] = np.empty((total_timesteps, *shape), dtype=dtype)
        next_obs_arrays[k] = np.empty((total_timesteps, *shape), dtype=dtype)
    
    actions = np.empty((total_timesteps, action_dim), dtype=np.float32)
    rewards = np.empty((total_timesteps,), dtype=np.float32)
    dones = np.empty((total_timesteps,), dtype=np.float32)
    masks = np.empty((total_timesteps,), dtype=np.float32)
    successes = np.empty((total_timesteps,), dtype=np.float32)
    language_embeddings = np.empty((total_timesteps, lang_emb_dim), dtype=np.float32)
    
    # Fill arrays
    offset = 0
    for dataset_idx, ep_idx, timesteps in demos_info:
        ep = all_datasets[dataset_idx][ep_idx]
        end = offset + timesteps
        
        obs_list = ep['observations']
        next_obs_list = ep['next_observations']
        
        for k in obs_keys:
            arr = obs_arrays[k]
            next_arr = next_obs_arrays[k]
            for t, (o, no) in enumerate(zip(obs_list, next_obs_list)):
                arr[offset + t] = o[k]
                next_arr[offset + t] = no[k]
        
        act_list = ep['actions']
        rew_list = ep['rewards']
        done_list = ep['dones']
        lang_list = ep['language_embedding']
        
        for t in range(timesteps):
            actions[offset + t] = act_list[t]
            rewards[offset + t] = rew_list[t]
            dones[offset + t] = done_list[t]
            language_embeddings[offset + t] = lang_list[t]
        
        max_rew = max(rew_list)
        successes[offset:end] = max_rew
        offset = end
    
    np.clip(actions, -1 + action_clip_eps, 1 - action_clip_eps, out=actions)
    np.subtract(1.0, dones, out=masks)
    
    fill_time = time.time() - t0
    
    stats = {
        'task_name': task_name,
        'num_demos': len(demos_info),
        'num_timesteps': total_timesteps,
        'load_time': load_time,
        'fill_time': fill_time,
        'total_time': load_time + fill_time,
    }
    
    return stats, None


def main():
    parser = argparse.ArgumentParser(description='Fast dataset creation from pickle rollouts')
    parser.add_argument('--rollouts_dir', type=str, required=True,
                        help='Directory containing pickle rollouts')
    parser.add_argument('--num_tasks', type=int, default=90,
                        help='Number of tasks to load (default: 90)')
    parser.add_argument('--num_demos', type=int, default=60,
                        help='Number of demos to load per task (default: 60)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: cpu_count)')
    
    args = parser.parse_args()
    
    rollouts_dir = Path(args.rollouts_dir)
    
    if not rollouts_dir.exists():
        print(f"Error: Directory not found: {rollouts_dir}")
        sys.exit(1)
    
    task_dirs = sorted([d for d in rollouts_dir.iterdir() if d.is_dir()])
    
    if not task_dirs:
        print(f"Error: No task directories found in {rollouts_dir}")
        sys.exit(1)
    
    num_workers = args.num_workers or min(cpu_count(), args.num_tasks)
    
    print(f"Found {len(task_dirs)} task directories")
    print(f"Will load {min(args.num_tasks, len(task_dirs))} tasks with {args.num_demos} demos each")
    print(f"Using {num_workers} parallel workers")
    print("="*80)
    
    demo_nums_to_use = list(range(args.num_demos))
    action_clip_eps = 1e-5
    
    # Prepare task arguments
    task_args = [
        (task_dir, demo_nums_to_use, action_clip_eps)
        for task_dir in task_dirs[:args.num_tasks]
    ]
    
    all_stats = []
    errors = []
    
    total_start = time.time()
    
    # Process tasks in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_task, arg): i for i, arg in enumerate(task_args)}
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                stats, error = future.result()
                if error:
                    errors.append(error)
                    print(f"[{idx+1}/{args.num_tasks}] ERROR: {error}")
                elif stats:
                    all_stats.append(stats)
                    print(f"[{idx+1}/{args.num_tasks}] {stats['task_name'][:50]}: {stats['num_demos']} demos, "
                          f"{stats['num_timesteps']} ts, {stats['total_time']:.2f}s")
            except Exception as e:
                errors.append(str(e))
                print(f"[{idx+1}/{args.num_tasks}] EXCEPTION: {e}")
    
    total_time = time.time() - total_start
    
    if all_stats:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        total_demos = sum(s['num_demos'] for s in all_stats)
        total_timesteps = sum(s['num_timesteps'] for s in all_stats)
        avg_time = sum(s['total_time'] for s in all_stats) / len(all_stats)
        avg_load = sum(s['load_time'] for s in all_stats) / len(all_stats)
        avg_fill = sum(s['fill_time'] for s in all_stats) / len(all_stats)
        
        print(f"Tasks loaded: {len(all_stats)}")
        print(f"Errors: {len(errors)}")
        print(f"Total demos: {total_demos}")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Total wall time: {total_time:.2f}s")
        print(f"Effective time per task: {total_time/len(all_stats):.2f}s")
        print(f"Avg sequential time per task: {avg_time:.2f}s (load={avg_load:.2f}s, fill={avg_fill:.2f}s)")
        print(f"Throughput: {total_timesteps/total_time:.0f} timesteps/sec")
        print("="*80)


if __name__ == "__main__":
    main()
