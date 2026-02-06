#!/usr/bin/env python3
"""
Profile HDF5 loading performance to compare with pickle format.

Usage:
    python profile_hdf5_loading.py \
        --rollouts_dir /path/to/hdf5/rollouts \
        --num_tasks 5
"""

import os
import sys
import h5py
import numpy as np
from pathlib import Path
import argparse
import time
from collections import defaultdict


def stack_dict_list(dict_list):
    """Stack a list of dictionaries into a dictionary of stacked arrays."""
    if not dict_list:
        return {}
    keys = dict_list[0].keys()
    return {k: np.concatenate([d[k] for d in dict_list], axis=0) for k in keys}


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


def process_task_hdf5_profiled(hdf5_path, task_str, demo_nums_to_use, action_clip_eps=1e-5):
    """
    Process HDF5 file with detailed profiling.
    Returns profiling statistics.
    """
    # Profiling stats
    stats = defaultdict(float)
    stats['num_files'] = 1  # Single HDF5 file per task
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
    
    # Profile HDF5 file opening
    with Timer('1_hdf5_open', stats):
        hdf_file = h5py.File(hdf5_path, 'r')
    
    try:
        with Timer('2_metadata_read', stats):
            data_group = hdf_file['data']
            total_demos = hdf_file.attrs['num_demos']
        
        for demo_idx in range(total_demos):
            if demo_idx not in demo_indices:
                continue
            
            stats['num_episodes'] += 1
            demo_key = f'demo_{demo_idx}'
            
            if demo_key not in data_group:
                continue
            
            demo = data_group[demo_key]
            
            # Profile observation loading from HDF5
            with Timer('3_obs_load_hdf5', stats):
                obs_keys = list(demo['obs'].keys())
                next_obs_keys = list(demo['next_obs'].keys())
                
                # Load all observation arrays
                obs_raw = {k: np.array(demo[f'obs/{k}']) for k in obs_keys}
                next_obs_raw = {k: np.array(demo[f'next_obs/{k}']) for k in next_obs_keys}
            
            # Profile dtype conversion
            with Timer('4_astype_conversion', stats):
                obs_f32 = {k: v.astype(np.float32) for k, v in obs_raw.items()}
                next_obs_f32 = {k: v.astype(np.float32) for k, v in next_obs_raw.items()}
            
            # Profile other array operations
            with Timer('5_array_processing', stats):
                a = np.clip(
                    np.array(demo['actions']).astype(np.float32),
                    -1 + action_clip_eps,
                    1 - action_clip_eps
                )
                r = np.array(demo['rewards']).astype(np.float32)
                dones = np.array(demo['dones']).astype(np.float32)
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
    
    finally:
        # Profile file closing
        with Timer('7_hdf5_close', stats):
            hdf_file.close()
    
    # Profile final concatenation
    with Timer('8_final_concat', stats):
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
    stats['total_time'] = sum(v for k, v in stats.items() if k.startswith(('1_', '2_', '3_', '4_', '5_', '6_', '7_', '8_')))
    
    return stats


def check_hdf5_structure(hdf5_path):
    """Check the structure of HDF5 file."""
    with h5py.File(hdf5_path, 'r') as f:
        info = {
            'num_demos': f.attrs.get('num_demos', 'unknown'),
            'num_source_files': f.attrs.get('num_source_files', 'unknown'),
            'format_version': f.attrs.get('format_version', 'unknown'),
        }
        
        # Check first demo structure
        if 'data' in f and 'demo_0' in f['data']:
            demo = f['data/demo_0']
            info['demo_0_keys'] = list(demo.keys())
            
            # Check observations
            if 'obs' in demo:
                info['obs_keys'] = list(demo['obs'].keys())
                # Check shapes
                for k in demo['obs'].keys():
                    info[f'obs_{k}_shape'] = demo[f'obs/{k}'].shape
                    info[f'obs_{k}_dtype'] = demo[f'obs/{k}'].dtype
            
            # Check actions
            if 'actions' in demo:
                info['actions_shape'] = demo['actions'].shape
                info['actions_dtype'] = demo['actions'].dtype
    
    return info


def compare_formats(pickle_stats_file=None):
    """
    Compare results with pickle format if stats file provided.
    """
    if not pickle_stats_file or not Path(pickle_stats_file).exists():
        return
    
    print("\n" + "="*80)
    print("COMPARISON WITH PICKLE FORMAT")
    print("="*80)
    
    # This is a placeholder - in practice you'd save stats from pickle script
    print("Run pickle profiling script first and save results to compare")


def main():
    parser = argparse.ArgumentParser(description='Profile HDF5 loading performance')
    parser.add_argument('--rollouts_dir', type=str, required=True,
                        help='Directory containing HDF5 rollouts')
    parser.add_argument('--num_tasks', type=int, default=5,
                        help='Number of tasks to profile (default: 5)')
    parser.add_argument('--num_demos', type=int, default=10,
                        help='Number of demos to load per task (default: 10)')
    parser.add_argument('--check_structure', action='store_true',
                        help='Check and print HDF5 structure')
    parser.add_argument('--compare_pickle', type=str, default=None,
                        help='Path to pickle profiling results for comparison')
    
    args = parser.parse_args()
    
    rollouts_dir = Path(args.rollouts_dir)
    
    if not rollouts_dir.exists():
        print(f"Error: Directory not found: {rollouts_dir}")
        sys.exit(1)
    
    # Find all task directories with demos.hdf5
    task_dirs = []
    for d in rollouts_dir.iterdir():
        if d.is_dir() and (d / "demos.hdf5").exists():
            task_dirs.append(d)
    
    if not task_dirs:
        print(f"Error: No task directories with demos.hdf5 found in {rollouts_dir}")
        sys.exit(1)
    
    print(f"Found {len(task_dirs)} task directories with HDF5 files")
    print(f"Will profile {min(args.num_tasks, len(task_dirs))} tasks")
    print("="*80)
    
    # Optionally check structure of first HDF5
    if args.check_structure:
        first_task = task_dirs[0]
        hdf5_path = first_task / "demos.hdf5"
        print("\nCHECKING HDF5 STRUCTURE:")
        print("-"*80)
        structure = check_hdf5_structure(hdf5_path)
        for k, v in structure.items():
            print(f"  {k}: {v}")
        print("="*80)
    
    # Profile each task
    all_stats = []
    demo_nums_to_use = list(range(args.num_demos))
    
    for i, task_dir in enumerate(task_dirs[:args.num_tasks]):
        task_name = task_dir.name
        hdf5_path = task_dir / "demos.hdf5"
        
        print(f"\n[{i+1}/{args.num_tasks}] Profiling task: {task_name}")
        print(f"  HDF5 file: {hdf5_path.name}")
        
        try:
            stats = process_task_hdf5_profiled(hdf5_path, task_name, demo_nums_to_use)
            all_stats.append(stats)
            
            # Print stats for this task
            print(f"  Episodes loaded: {stats['num_episodes']}")
            print(f"  Total timesteps: {stats['num_timesteps']}")
            print(f"  Total time: {stats['total_time']:.3f}s")
            print(f"\n  Time breakdown:")
            
            # Sort by time
            time_items = [(k, v) for k, v in stats.items() if k.startswith(('1_', '2_', '3_', '4_', '5_', '6_', '7_', '8_'))]
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
        print("AGGREGATE STATISTICS (HDF5 FORMAT)")
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
        print(f"  Throughput: {avg_stats['num_timesteps']/avg_stats['total_time']:.0f} timesteps/sec")
        print(f"\n  Average time breakdown:")
        
        time_items = [(k, v) for k, v in avg_stats.items() if k.startswith(('1_', '2_', '3_', '4_', '5_', '6_', '7_', '8_'))]
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
        print("OPTIMIZATION OPPORTUNITIES:")
        print("-"*80)
        
        if bottleneck[0] == '3_obs_load_hdf5':
            print("❌ HDF5 loading is the bottleneck")
            print("✅ Possible optimizations:")
            print("   - Use HDF5 chunk caching: hdf5_file.id.set_cache(...)")
            print("   - Load multiple demos in parallel")
            print("   - Consider memory-mapping for very large files")
            print("   Expected speedup: 2-3x")
        
        elif bottleneck[0] == '4_astype_conversion':
            print("❌ Type conversion is the bottleneck")
            print("✅ Possible optimizations:")
            print("   - Store arrays as float32 in HDF5 during conversion")
            print("   - Skip conversion if already float32")
            print("   Expected speedup: 2-3x")
        
        elif bottleneck[0] == '8_final_concat':
            print("❌ Final concatenation is the bottleneck")
            print("✅ Possible optimizations:")
            print("   - Pre-allocate output arrays")
            print("   - Load all demos at once if memory allows")
            print("   Expected speedup: 1.5-2x")
        
        elif bottleneck[0] == '1_hdf5_open':
            print("❌ HDF5 file opening is slow")
            print("✅ Possible optimizations:")
            print("   - Keep files open longer (cache file handles)")
            print("   - Use HDF5 file caching")
            print("   Expected speedup: 1.5-2x")
        
        else:
            print("✅ HDF5 format is well-optimized!")
            print("   No major bottlenecks detected.")
        
        print("="*80)
        
        # Overall assessment
        print("\n" + "="*80)
        print("OVERALL ASSESSMENT")
        print("="*80)
        print(f"⚡ HDF5 loading speed: {avg_stats['num_timesteps']/avg_stats['total_time']:.0f} timesteps/sec")
        print(f"⏱️  Time per episode: {avg_stats['total_time']/avg_stats['num_episodes']:.4f}s")
        print(f"📊 Time per timestep: {avg_stats['total_time']/avg_stats['num_timesteps']*1000:.2f}ms")
        
        if args.compare_pickle:
            compare_formats(args.compare_pickle)


if __name__ == "__main__":
    main()