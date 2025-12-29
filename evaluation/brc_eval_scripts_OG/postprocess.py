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
import time


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
    for pkl_file in pickle_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
                results.append(data)
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
    
    # Initialize wandb and resume the run
    print(f"\nInitializing wandb run: {wandb_run_id}")
    run = wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        id=wandb_run_id,
        resume="allow",
    )
    
    print("Logging results to wandb...")
    
    # Log each result
    for result in results:
        n = result['n']
        eval_info = result['eval_info']
        
        # Get checkpoint number from critic path
        critic_path = result['critic_restore_path']
        critic_ckpt = int(os.path.basename(critic_path).split('.')[0].split('_')[-1])
        
        # Log with step=n so all results appear at different steps
        log_data = {}
        for key, value in eval_info.items():
            log_data[f'best_of_N_eval/n{n}/{key}'] = value
            log_data[f'best_of_N_eval/{key}'] = value  # Also log without n prefix
        
        log_data['best_of_N_eval/n'] = n
        log_data['best_of_N_eval/critic_checkpoint'] = critic_ckpt
        
        run.log(log_data, step=n)
        print(f"  Logged n={n}: {eval_info}")
    
    print("\n✅ Postprocessing complete!")
    print(f"View results at: {run.url}")
    
    # Create summary statistics
    summary_stats = {}
    
    # Compute mean across all n values for each metric
    if results:
        all_metrics = results[0]['eval_info'].keys()
        for metric in all_metrics:
            values = [r['eval_info'][metric] for r in results]
            summary_stats[f'best_of_N_eval/summary/mean_{metric}'] = sum(values) / len(values)
            summary_stats[f'best_of_N_eval/summary/max_{metric}'] = max(values)
            summary_stats[f'best_of_N_eval/summary/min_{metric}'] = min(values)
        
        # Log summary
        run.log(summary_stats, step=max([r['n'] for r in results]) + 1)
        print(f"\n📊 Summary statistics logged")
    
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Postprocess best-of-N evaluation results and log to wandb'
    )
    
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory containing evaluation pickle files')
    parser.add_argument('--wandb_entity', type=str, default='yajatyadav',
                       help='Wandb entity')
    parser.add_argument('--wandb_project', type=str, default='multitask_RL',
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