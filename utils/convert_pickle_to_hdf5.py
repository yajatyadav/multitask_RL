#!/usr/bin/env python3
"""
Convert pickle trajectory files to HDF5 format for faster loading.
Combines all pickle files for each task into a single HDF5 file.

Usage:
    python convert_pickles_to_hdf5.py \
        --input_dir /path/to/pickle/rollouts \
        --output_dir /path/to/hdf5/rollouts
"""

import os
import sys
import pickle
import h5py
import numpy as np
from pathlib import Path
from jax import tree_util
import argparse
from tqdm import tqdm


def stack_list_of_dicts(dict_list):
    """Convert list of dicts to dict of stacked arrays."""
    if not dict_list:
        return {}
    return tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *dict_list)


def write_dict_to_hdf5(hdf_group, data_dict, prefix=''):
    """Recursively write nested dict to HDF5 group."""
    for key, value in data_dict.items():
        full_key = f"{prefix}/{key}" if prefix else key
        
        if isinstance(value, dict):
            # Create subgroup for nested dict
            write_dict_to_hdf5(hdf_group, value, prefix=full_key)
        elif isinstance(value, np.ndarray):
            # Write array directly
            hdf_group.create_dataset(full_key, data=value, compression='gzip', compression_opts=4)
        else:
            # Convert to numpy array first
            hdf_group.create_dataset(full_key, data=np.array(value), compression='gzip', compression_opts=4)


def convert_episode_to_hdf5(ep, hdf_group, demo_idx):
    """Convert a single episode to HDF5 format."""
    demo_name = f"demo_{demo_idx}"
    demo_group = hdf_group.create_group(demo_name)
    
    # Handle observations (might be list of dicts)
    if isinstance(ep['observations'], list):
        obs_stacked = stack_list_of_dicts(ep['observations'])
        next_obs_stacked = stack_list_of_dicts(ep['next_observations'])
    else:
        obs_stacked = ep['observations']
        next_obs_stacked = ep['next_observations']
    
    # Write observations
    write_dict_to_hdf5(demo_group, obs_stacked, prefix='obs')
    write_dict_to_hdf5(demo_group, next_obs_stacked, prefix='next_obs')
    
    # Write simple arrays
    demo_group.create_dataset('actions', data=ep['actions'], compression='gzip', compression_opts=4)
    demo_group.create_dataset('rewards', data=ep['rewards'], compression='gzip', compression_opts=4)
    demo_group.create_dataset('dones', data=ep['dones'], compression='gzip', compression_opts=4)
    
    return demo_group


def convert_task_to_single_hdf5(task_dir, output_path):
    """
    Convert all pickle files in a task directory to a single HDF5 file.
    
    Args:
        task_dir: Path to directory containing trajs_*.pkl files
        output_path: Path to output HDF5 file
    """
    # Find all pickle files
    pkl_files = sorted(task_dir.glob('trajs_*.pkl'))
    
    if not pkl_files:
        print(f"Warning: No pickle files found in {task_dir}")
        return 0
    
    # Load all datasets
    all_episodes = []
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            dataset = pickle.load(f)
            all_episodes.extend(dataset)
    
    # Create single HDF5 file with all demos
    with h5py.File(output_path, 'w') as hdf_file:
        # Create data group
        data_group = hdf_file.create_group('data')
        
        # Convert each episode with sequential numbering
        for demo_idx, ep in enumerate(all_episodes):
            convert_episode_to_hdf5(ep, data_group, demo_idx)
        
        # Store metadata
        hdf_file.attrs['num_demos'] = len(all_episodes)
        hdf_file.attrs['num_source_files'] = len(pkl_files)
        hdf_file.attrs['format_version'] = '1.0'
    
    return len(all_episodes)


def convert_directory(input_dir, output_dir):
    """Convert all pickle files, combining per task into single HDF5 files."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Find all task directories
    task_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(task_dirs)} task directories")
    
    # Count total pickle files for progress tracking
    total_tasks = len(task_dirs)
    
    # Process each task directory
    results = {}
    for task_dir in tqdm(task_dirs, desc="Converting tasks"):
        task_name = task_dir.name
        
        # Create output directory for this task
        output_task_dir = output_dir / task_name
        output_task_dir.mkdir(parents=True, exist_ok=True)
        
        # Combine all pickle files into single HDF5
        output_file = output_task_dir / "demos.hdf5"
        
        try:
            num_demos = convert_task_to_single_hdf5(task_dir, output_file)
            results[task_name] = {
                'num_demos': num_demos,
                'output_file': output_file
            }
        except Exception as e:
            print(f"\n❌ Error converting {task_name}: {e}")
            import traceback
            traceback.print_exc()
            results[task_name] = {'error': str(e)}
    
    print(f"\n✅ Conversion complete!")
    print(f"Output directory: {output_dir}")
    print("\nSummary:")
    total_demos = 0
    for task_name, info in results.items():
        if 'num_demos' in info:
            print(f"  {task_name}: {info['num_demos']} demos")
            total_demos += info['num_demos']
        else:
            print(f"  {task_name}: ERROR - {info.get('error', 'Unknown')}")
    print(f"\nTotal demos converted: {total_demos}")


def verify_conversion(hdf5_path):
    """Quick verification that HDF5 file is valid."""
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    with h5py.File(hdf5_path, 'r') as f:
        num_demos = f.attrs['num_demos']
        num_source_files = f.attrs.get('num_source_files', 'unknown')
        
        print(f"File: {hdf5_path}")
        print(f"  Total demos: {num_demos}")
        print(f"  Source pickle files: {num_source_files}")
        
        # Check first demo
        if num_demos > 0:
            demo_0 = f['data/demo_0']
            print(f"\nFirst demo (demo_0):")
            print(f"  Actions shape: {demo_0['actions'].shape}")
            print(f"  Rewards shape: {demo_0['rewards'].shape}")
            print(f"  Obs keys: {list(demo_0['obs'].keys())}")
            
            # Show one obs key shape
            first_obs_key = list(demo_0['obs'].keys())[0]
            print(f"  Obs['{first_obs_key}'] shape: {demo_0[f'obs/{first_obs_key}'].shape}")
        
        # Check last demo if multiple
        if num_demos > 1:
            demo_last = f[f'data/demo_{num_demos-1}']
            print(f"\nLast demo (demo_{num_demos-1}):")
            print(f"  Actions shape: {demo_last['actions'].shape}")
        
        print("="*60)


def test_loading_speed(hdf5_path):
    """Test loading speed of converted HDF5 file."""
    import time
    
    print("\n" + "="*60)
    print("SPEED TEST")
    print("="*60)
    
    # Test loading all demos
    start = time.time()
    with h5py.File(hdf5_path, 'r') as f:
        data_group = f['data']
        for demo_key in data_group.keys():
            demo = data_group[demo_key]
            # Load all data
            _ = np.array(demo['actions'])
            _ = np.array(demo['rewards'])
            for obs_key in demo['obs'].keys():
                _ = np.array(demo[f'obs/{obs_key}'])
    elapsed = time.time() - start
    
    print(f"Time to load all demos: {elapsed:.2f}s")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert pickle trajectory files to HDF5')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing pickle files organized by task')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for HDF5 files')
    parser.add_argument('--verify', action='store_true',
                        help='Verify one converted file')
    parser.add_argument('--test_speed', action='store_true',
                        help='Test loading speed of converted file')
    parser.add_argument('--test_task', type=str, default=None,
                        help='Only convert a specific task (for testing)')
    
    args = parser.parse_args()
    
    if args.test_task:
        # Test on single task
        input_task_dir = Path(args.input_dir) / args.test_task
        output_task_dir = Path(args.output_dir) / args.test_task
        
        if not input_task_dir.exists():
            print(f"Error: Task directory not found: {input_task_dir}")
            sys.exit(1)
        
        output_task_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_task_dir / "demos.hdf5"
        
        print(f"Converting task: {args.test_task}")
        num_demos = convert_task_to_single_hdf5(input_task_dir, output_file)
        print(f"✅ Converted {num_demos} demos to {output_file}")
        
        if args.verify:
            verify_conversion(output_file)
        
        if args.test_speed:
            test_loading_speed(output_file)
    else:
        # Convert entire directory
        convert_directory(args.input_dir, args.output_dir)
        
        if args.verify or args.test_speed:
            # Find first converted file
            output_dir = Path(args.output_dir)
            first_hdf5 = next(output_dir.rglob('demos.hdf5'), None)
            
            if first_hdf5:
                if args.verify:
                    verify_conversion(first_hdf5)
                if args.test_speed:
                    test_loading_speed(first_hdf5)
            else:
                print("No HDF5 files found for verification")