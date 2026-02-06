#!/usr/bin/env python3
"""
Profile pickle loading performance to identify bottlenecks.

Usage:
    python profile_pickle_loading.py \
        --rollouts_dir /path/to/pickle/rollouts \
        --num_tasks 5
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from jax import tree_util
import argparse
import time
from collections import defaultdict
import tqdm


def stack_dict_list(dict_list):
    """Stack a list of dictionaries into a dictionary of stacked arrays."""
    if not dict_list:
        return {}
    keys = dict_list[0].keys()
    return {k: np.concatenate([d[k] for d in dict_list], axis=0) for k in keys}


def batch_dicts(dicts):
    """Convert list of dicts to dict of stacked arrays."""
    return tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *dicts)


class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, name, stats_dict):
        self.name = name
        self.stats = stats_dict
        
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start
        self.stats[self.name] += elapsed


def process_task_pickle_profiled(task_paths, task_str, demo_nums_to_use, action_clip_eps=1e-5):
    """
    Process pickle files with detailed profiling.
    Returns dataset and profiling statistics.
    """
    # Profiling stats
    stats = defaultdict(float)
    stats['num_files'] = len(task_paths)
    stats['num_episodes'] = 0
    stats['num_timesteps'] = 0
    
    # Data holders
    observations = []
    actions = []
    next_observations = []
    terminals = []
    rewards = []
    masks = []
    successes = []
    
    demo_indices = set(demo_nums_to_use)
    demo_num = -1
    
    for path in task_paths:
        # Profile pickle loading
        with Timer('1_pickle_load', stats):
            with open(path, 'rb') as f:
                dataset = pickle.load(f)
        
        for ep in dataset:
            demo_num += 1
            
            if demo_num not in demo_indices:
                continue
            
            stats['num_episodes'] += 1
            
            # Profile observation processing
            with Timer('2_obs_extract', stats):
                obs = ep['observations']
                next_obs = ep['next_observations']
            
            # Profile stacking if needed
            with Timer('3_batch_stack', stats):
                if isinstance(obs, list):
                    # This is the expensive operation!
                    obs_batched = tree_util.tree_map(
                        lambda *xs: np.stack(xs, axis=0), 
                        *obs
                    )
                    next_obs_batched = tree_util.tree_map(
                        lambda *xs: np.stack(xs, axis=0), 
                        *next_obs
                    )
                else:
                    obs_batched = obs
                    next_obs_batched = next_obs
            
            # Profile dtype conversion
            with Timer('4_astype_conversion', stats):
                obs_f32 = tree_util.tree_map(lambda x: x.astype(np.float32), obs_batched)
                next_obs_f32 = tree_util.tree_map(lambda x: x.astype(np.float32), next_obs_batched)
            
            # Profile other array operations
            with Timer('5_array_processing', stats):
                a = np.clip(
                    np.array(ep['actions']).astype(np.float32),
                    -1 + action_clip_eps,
                    1 - action_clip_eps
                )
                r = np.array(ep['rewards']).astype(np.float32)
                dones = np.array(ep['dones']).astype(np.float32)
                masks_f32 = 1.0 - dones
                success = np.full(a.shape[0], np.max(r), dtype=np.float32)
                stats['num_timesteps'] += a.shape[0]
            
            # Profile appending
            with Timer('6_list_append', stats):
                observations.append(obs_f32)
                actions.append(a)
                rewards.append(r)
                next_observations.append(next_obs_f32)
                terminals.append(dones)
                masks.append(masks_f32)
                successes.append(success)
    
    # Profile final concatenation
    with Timer('7_final_concat', stats):
        if not observations:
            raise ValueError(f"No data found for task {task_str}")
        
        final_obs = stack_dict_list(observations)
        final_next_obs = stack_dict_list(next_observations)
        final_actions = np.concatenate(actions, axis=0)
        final_rewards = np.concatenate(rewards, axis=0)
        final_terminals = np.concatenate(terminals, axis=0)
        final_masks = np.concatenate(masks, axis=0)
        final_successes = np.concatenate(successes, axis=0)
    
    # Calculate total time
    stats['total_time'] = sum(v for k, v in stats.items() if k.startswith(('1_', '2_', '3_', '4_', '5_', '6_', '7_')))
    
    return stats


def check_obs_structure(pickle_path):
    """Check the structure of observations in pickle file."""
    with open(pickle_path, 'rb') as f:
        dataset = pickle.load(f)
    
    if not dataset:
        return "Empty dataset"
    
    ep = dataset[0]
    obs = ep['observations']
    
    info = {
        'obs_type': type(obs).__name__,
        'is_list': isinstance(obs, list),
    }
    
    if isinstance(obs, list):
        info['list_length'] = len(obs)
        if obs:
            info['element_type'] = type(obs[0]).__name__
            if isinstance(obs[0], dict):
                info['dict_keys'] = list(obs[0].keys())
                # Check shapes
                for k, v in obs[0].items():
                    if isinstance(v, np.ndarray):
                        info[f'shape_{k}'] = v.shape
    elif isinstance(obs, dict):
        info['dict_keys'] = list(obs.keys())
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                info[f'shape_{k}'] = v.shape
    
    return info


def main():
    parser = argparse.ArgumentParser(description='Profile pickle loading performance')
    parser.add_argument('--rollouts_dir', type=str, required=True,
                        help='Directory containing pickle rollouts')
    parser.add_argument('--num_tasks', type=int, default=5,
                        help='Number of tasks to profile (default: 5)')
    parser.add_argument('--num_demos', type=int, default=10,
                        help='Number of demos to load per task (default: 10)')
    parser.add_argument('--check_structure', action='store_true',
                        help='Check and print observation structure')
    
    args = parser.parse_args()
    
    rollouts_dir = Path(args.rollouts_dir)
    
    if not rollouts_dir.exists():
        print(f"Error: Directory not found: {rollouts_dir}")
        sys.exit(1)
    
    # Find all task directories
    task_dirs = [d for d in rollouts_dir.iterdir() if d.is_dir()]
    
    if not task_dirs:
        print(f"Error: No task directories found in {rollouts_dir}")
        sys.exit(1)
    
    print(f"Found {len(task_dirs)} task directories")
    print(f"Will profile {min(args.num_tasks, len(task_dirs))} tasks")
    print("="*80)
    
    # Optionally check structure of first pickle
    if args.check_structure:
        first_task = task_dirs[0]
        pkl_files = list(first_task.glob('trajs_*.pkl'))
        if pkl_files:
            print("\nCHECKING OBSERVATION STRUCTURE:")
            print("-"*80)
            structure = check_obs_structure(pkl_files[0])
            for k, v in structure.items():
                print(f"  {k}: {v}")
            print("="*80)
    
    # Profile each task
    all_stats = []
    demo_nums_to_use = list(range(args.num_demos))
    
    for i, task_dir in enumerate(task_dirs[:args.num_tasks]):
        task_name = task_dir.name
        
        # Find pickle files
        pkl_files = sorted(task_dir.glob('trajs_*.pkl'))
        
        if not pkl_files:
            print(f"Skipping {task_name}: no pickle files found")
            continue
        
        print(f"\n[{i+1}/{args.num_tasks}] Profiling task: {task_name}")
        print(f"  Pickle files: {len(pkl_files)}")
        
        try:
            stats = process_task_pickle_profiled(pkl_files, task_name, demo_nums_to_use)
            all_stats.append(stats)
            
            # Print stats for this task
            print(f"  Episodes loaded: {stats['num_episodes']}")
            print(f"  Total timesteps: {stats['num_timesteps']}")
            print(f"  Total time: {stats['total_time']:.3f}s")
            print(f"\n  Time breakdown:")
            
            # Sort by time
            time_items = [(k, v) for k, v in stats.items() if k.startswith(('1_', '2_', '3_', '4_', '5_', '6_', '7_'))]
            time_items.sort(key=lambda x: x[1], reverse=True)
            
            for key, value in time_items:
                percentage = (value / stats['total_time'] * 100) if stats['total_time'] > 0 else 0
                label = key.split('_', 1)[1].replace('_', ' ').title()
                print(f"    {label:20s}: {value:6.3f}s ({percentage:5.1f}%)")
        
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Print aggregate statistics
    if all_stats:
        print("\n" + "="*80)
        print("AGGREGATE STATISTICS")
        print("="*80)
        
        # Average times
        avg_stats = defaultdict(float)
        for stats in all_stats:
            for k, v in stats.items():
                avg_stats[k] += v
        
        n = len(all_stats)
        for k in avg_stats:
            avg_stats[k] /= n
        
        print(f"\nAverage across {n} tasks:")
        print(f"  Episodes per task: {avg_stats['num_episodes']:.1f}")
        print(f"  Timesteps per task: {avg_stats['num_timesteps']:.1f}")
        print(f"  Total time per task: {avg_stats['total_time']:.3f}s")
        print(f"\n  Average time breakdown:")
        
        time_items = [(k, v) for k, v in avg_stats.items() if k.startswith(('1_', '2_', '3_', '4_', '5_', '6_', '7_'))]
        time_items.sort(key=lambda x: x[1], reverse=True)
        
        for key, value in time_items:
            percentage = (value / avg_stats['total_time'] * 100) if avg_stats['total_time'] > 0 else 0
            label = key.split('_', 1)[1].replace('_', ' ').title()
            print(f"    {label:20s}: {value:6.3f}s ({percentage:5.1f}%)")
        
        # Identify bottleneck
        print("\n" + "-"*80)
        bottleneck = max(time_items, key=lambda x: x[1])
        bottleneck_label = bottleneck[0].split('_', 1)[1].replace('_', ' ').title()
        bottleneck_pct = (bottleneck[1] / avg_stats['total_time'] * 100)
        
        print(f"🔥 BOTTLENECK IDENTIFIED: {bottleneck_label}")
        print(f"   Takes {bottleneck[1]:.3f}s ({bottleneck_pct:.1f}% of total time)")
        
        # Provide recommendations
        print("\n" + "-"*80)
        print("RECOMMENDATIONS:")
        print("-"*80)
        
        if bottleneck[0] == '3_batch_stack':
            print("❌ Problem: Stacking list of dicts is extremely slow")
            print("✅ Solution: Convert pickle files to pre-stacked format using HDF5")
            print("   Expected speedup: 10-100x")
            print("   Run: python convert_pickles_to_hdf5.py ...")
        
        elif bottleneck[0] == '1_pickle_load':
            print("❌ Problem: Pickle file loading is slow")
            print("✅ Solution: Parallelize file loading or switch to HDF5")
            print("   Expected speedup: 2-5x with parallel, 10x with HDF5")
        
        elif bottleneck[0] == '4_astype_conversion':
            print("❌ Problem: Type conversion is slow")
            print("✅ Solution: Store arrays as float32 in pickle files")
            print("   Expected speedup: 2-3x")
        
        elif bottleneck[0] == '7_final_concat':
            print("❌ Problem: Final concatenation is slow")
            print("✅ Solution: Pre-allocate arrays or use HDF5")
            print("   Expected speedup: 2-5x")
        
        print("="*80)


if __name__ == "__main__":
    main()