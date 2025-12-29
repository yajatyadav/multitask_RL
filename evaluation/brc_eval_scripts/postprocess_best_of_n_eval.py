#!/usr/bin/env python3
"""
Postprocessing script to aggregate best-of-N evaluation results and log to wandb.
This script runs after all parallel evaluation jobs complete.

Usage:
    python postprocess_best_of_n_eval.py \
        --output_dir ./eval_results \
        --wandb_entity yajatyadav \
        --wandb_project multitask_RL \
        --wandb_run_id abc123 \
        --env_name libero_90-living_room_scene1
"""

import os
import pickle
import argparse
import glob
import wandb
import sys

# Add project root to path to import utility functions
sys.path.insert(0, os.getcwd())
from utils.log_utils import get_wandb_video


def aggregate_and_log_results(
    output_dir: str,
    wandb_entity: str,
    wandb_project: str,
    wandb_run_id: str,
    env_name: str,
):
    """
    Read all evaluation pickle files from output_dir and log to wandb.
    
    Args:
        output_dir: Directory containing evaluation pickle files
        wandb_entity: Wandb entity
        wandb_project: Wandb project
        wandb_run_id: Wandb run ID to log to
        env_name: Environment name (used to filter pickle files)
    """
    
    print(f"Starting postprocessing for run {wandb_run_id}")
    print(f"Looking for results in: {output_dir}")
    
    # Find all pickle files
    pattern = os.path.join(output_dir, f'eval_libero_best_of_N_n*_{env_name}_*.pkl')
    pickle_files = glob.glob(pattern)
    
    if not pickle_files:
        print(f"ERROR: No pickle files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(pickle_files)} result files")
    
    # Load all results
    results = []
    total_video_count = 0
    for pkl_file in pickle_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                results.append(data)
                # Count videos
                if 'video_renders' in data:
                    for task_renders in data['video_renders'].values():
                        total_video_count += len(task_renders)
                print(f"Loaded: {pkl_file} (n={data['n']})")
        except Exception as e:
            print(f"ERROR loading {pkl_file}: {e}")
    
    if not results:
        print("ERROR: No results loaded successfully")
        return
    
    # Sort by n value
    results.sort(key=lambda x: x['n'])
    
    print(f"\nSuccessfully loaded {len(results)} results")
    print(f"N values: {[r['n'] for r in results]}")
    if total_video_count > 0:
        print(f"Total videos to upload: {total_video_count}")
        print(f"Note: Video upload may take a few minutes...")
    
    # Get checkpoint number (should be same for all results)
    critic_ckpt = int(os.path.basename(results[0]['critic_restore_path']).split('.')[0].split('_')[-1])
    
    # Initialize wandb and resume the run
    print(f"\nInitializing wandb run: {wandb_run_id}")
    run = wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        id=wandb_run_id,
        resume="allow",
    )
    
    # Define custom x-axis for this checkpoint's best-of-n evaluation
    # This creates a separate x-axis (num_samples) for eval_ckpt_X metrics
    wandb.define_metric(f"eval_ckpt_{critic_ckpt}/num_samples")
    wandb.define_metric(f"eval_ckpt_{critic_ckpt}/*", step_metric=f"eval_ckpt_{critic_ckpt}/num_samples")
    
    print("\nLogging structure:")
    print(f"  eval_ckpt_{critic_ckpt}/success (x-axis: num_samples)")
    print(f"  eval_ckpt_{critic_ckpt}_{{task_name}}/success (x-axis: num_samples)")
    print(f"  eval_ckpt_{critic_ckpt}_{{task_name}}/video\n")
    print("Logging results to wandb...")
    
    # Log each result
    for result in results:
        n = result['n']
        eval_info = result['eval_info']
        per_task_eval_info = result.get('per_task_eval_info', {})
        task_names = result.get('task_names', list(per_task_eval_info.keys()))
        video_renders = result.get('video_renders', {})
        
        print(f"  Processing n={n}...")
        
        # Build log data for this n value
        log_data = {
            f'eval_ckpt_{critic_ckpt}/num_samples': n,
        }
        
        # Add aggregated success rate
        if 'success' in eval_info:
            log_data[f'eval_ckpt_{critic_ckpt}/success'] = eval_info['success']
            print(f"    Aggregated success: {eval_info['success']:.3f}")
        
        # Add per-task metrics
        for task_name in task_names:
            if task_name in per_task_eval_info:
                task_eval_info = per_task_eval_info[task_name]
                
                # Log success rate for this task
                if 'success' in task_eval_info:
                    log_data[f'eval_ckpt_{critic_ckpt}_{task_name}/success'] = task_eval_info['success']
                
                # Add video if available
                if task_name in video_renders and len(video_renders[task_name]) > 0:
                    renders = video_renders[task_name]
                    video = get_wandb_video(renders)
                    log_data[f'eval_ckpt_{critic_ckpt}_{task_name}/video'] = video
        
        # Log all data for this n value in one call
        run.log(log_data)
        print(f"  ✅ Logged n={n}")
    
    print("\n✅ Postprocessing complete!")
    print(f"View results at: {run.url}")
    print(f"\nMetrics are organized as:")
    print(f"  - eval_ckpt_{critic_ckpt}/success (aggregated across tasks)")
    print(f"  - eval_ckpt_{critic_ckpt}_<task_name>/success (per-task)")
    print(f"  - eval_ckpt_{critic_ckpt}_<task_name>/video (per-task videos)")
    print(f"  All with x-axis: num_samples (1, 2, 4, 8, 16, 32, ...)")
    
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Postprocess best-of-N evaluation results and log to wandb'
    )
    
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory containing evaluation pickle files')
    parser.add_argument('--wandb_entity', type=str, required=True,
                       help='Wandb entity')
    parser.add_argument('--wandb_project', type=str, required=True,
                       help='Wandb project')
    parser.add_argument('--wandb_run_id', type=str, required=True,
                       help='Wandb run ID to log to')
    parser.add_argument('--env_name', type=str, required=True,
                       help='Environment name')
    
    args = parser.parse_args()
    
    aggregate_and_log_results(
        output_dir=args.output_dir,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_run_id=args.wandb_run_id,
        env_name=args.env_name,
    )