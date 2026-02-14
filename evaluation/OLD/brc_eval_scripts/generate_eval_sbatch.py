import os
from typing import List
import wandb
import time as time_module

"""
Modified generate_sbatch_script function with postprocessing support.
Replace the function in your generate_eval_sbatch.py with this version.
"""

def generate_sbatch_script(
    n_vals: List[int],
    actor_restore_path: str,
    critic_restore_path: str,
    env_name: str,
    task_name: str,
    wandb_name: str,
    wandb_run_id: str = None,
    output_file: str = None,
    output_file_dir: str = 'scripts/shell_scripts',
    wandb_entity: str = 'yajatyadav',
    wandb_project: str = 'multitask_RL',
    wandb_group: str = 'eval_libero_best_of_N',
    output_dir: str = './eval_results',
    # SBATCH parameters
    account: str = 'co_rail',
    partition: str = 'savio4_gpu',
    gpu_type: str = 'A5000',
    num_gpus: int = 1,
    num_nodes: int = 1,
    num_tasks: int = 1,
    cpus_per_task: int = 4,
    qos: str = 'rail_gpu4_high',
    time: str = '24:00:00',
    mem: str = '60G',
    requeue: bool = True,
    script_runner: str = 'scripts/automatic/run.sh',
    postprocess_script: str = 'evaluation/brc_eval_scripts/postprocess_best_of_n_eval.py',
):
    """
    Generate a shell script with sbatch commands for each n value,
    plus a final postprocessing job that aggregates results and logs to wandb.
    
    Args:
        n_vals: List of n values to evaluate
        actor_restore_path: Path to actor checkpoint
        critic_restore_path: Path to critic checkpoint
        env_name: Environment name
        task_name: Task name(s), pipe-separated for multi-task
        wandb_entity: Wandb entity name
        wandb_project: Wandb project name
        wandb_group: Wandb group name
        wandb_name: Base name for wandb runs (will be suffixed with _nX)
        output_dir: Directory to save evaluation results
        account: SLURM account
        partition: SLURM partition
        gpu_type: GPU type to request
        num_gpus: Number of GPUs per job
        num_nodes: Number of nodes
        num_tasks: Number of tasks
        cpus_per_task: CPUs per task
        qos: Quality of service
        time: Time limit
        mem: Memory limit
        requeue: Whether to requeue failed jobs
        script_runner: Path to the script runner
        postprocess_script: Path to the postprocessing script
        output_file: Output shell script filename
    """
    
    # Environment variables for WANDB robustness
    wandb_env_vars = [
        'WANDB_MODE=offline',  # Run in offline mode
        'WANDB_SERVICE_WAIT=86400',
        'WANDB_NETWORK_TIMEOUT=600',
        'WANDB_FILE_TRANSFER_TIMEOUT=1200',
        'WANDB_INIT_TIMEOUT=300',
        'WANDB_HTTP_TIMEOUT=600',
        'WANDB_RETRY_ATTEMPTS=15',
        'WANDB_RETRY_WAIT_MIN=5',
        'WANDB_RETRY_WAIT_MAX=120'
    ]
    
    # Environment variables for system configuration
    system_env_vars = [
        'MUJOCO_GL=egl',
        'XLA_PYTHON_CLIENT_PREALLOCATE=false',
        'OMP_NUM_THREADS=1',
        'OPENBLAS_NUM_THREADS=1',
        'MKL_NUM_THREADS=1',
        'VECLIB_MAXIMUM_THREADS=1',
        'NUMEXPR_NUM_THREADS=1'
    ]
    
    all_env_vars = ' '.join(wandb_env_vars + system_env_vars)
    
    # SBATCH options for evaluation jobs
    sbatch_opts = [
        f'-A {account}',
        f'-p {partition}',
        f'--gres=gpu:{gpu_type}:{num_gpus}',
        f'-N {num_nodes}',
        f'-n {num_tasks}',
        f'-c {cpus_per_task}',
        f'--qos={qos}',
        f'-t {time}',
        f'--mem={mem}',
        '--parsable'
    ]
    
    if requeue:
        sbatch_opts.append('--requeue')
    
    sbatch_opts_str = ' '.join(sbatch_opts)
    
    # Generate the shell script
    lines = ['#!/usr/bin/env bash', '']
    lines.append('# Best-of-N Evaluation Jobs')
    lines.append('# All jobs run in WANDB_MODE=offline')
    lines.append('')

    # initialize wandb run and get run id
    if wandb_run_id is None:
        print(f"😮😮😮 a wandb run id was not provided, so we will first initialize a wandb run before starting evaluation 😮😮😮")
        run = wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            group=wandb_group,
            name=wandb_name,
        )
        wandb_run_id = run.id
        wandb.finish()  # Close it immediately, we just needed the ID

    timestamp = time_module.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f'run_{wandb_run_id}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Track all job IDs for dependency
    job_ids = []
    
    # Generate evaluation jobs
    for i, n in enumerate(n_vals):
        # Build the python command
        python_cmd = (
            f'uv run evaluation/brc_eval_scripts/eval_libero_best_of_N_single_n.py '
            f'--n {n} '
            f'--actor_restore_path "{actor_restore_path}" '
            f'--critic_restore_path "{critic_restore_path}" '
            f'--env_name "{env_name}" '
            f'--task_name "{task_name}" '
            f'--wandb_entity {wandb_entity} '
            f'--wandb_project {wandb_project} '
            f'--wandb_run_id {wandb_run_id} '
            f'--output_dir {output_dir}'
        )
        
        # Build the full sbatch command
        comment = f'eval_best_of_N.n{n}'
        sbatch_cmd = (
            f'jobid{i}=$({all_env_vars} sbatch {sbatch_opts_str} '
            f'--comment="{comment}" {script_runner} \'{python_cmd}\') '
            f'&& echo $jobid{i}'
        )
        
        lines.append(sbatch_cmd)
        job_ids.append(f'$jobid{i}')
    
    lines.append('')
    lines.append('# Postprocessing Job')
    lines.append('# Waits for all evaluation jobs to complete, then aggregates results')
    lines.append('')
    
    # Build dependency string
    dependency_str = ':'.join(job_ids)
    
    # Build postprocessing command
    postprocess_cmd = (
        f'uv run {postprocess_script} '
        f'--output_dir {output_dir} '
        f'--wandb_entity {wandb_entity} '
        f'--wandb_project {wandb_project} '
        f'--wandb_run_id {wandb_run_id} '
        f'--env_name "{env_name}"'
    )
    
    # SBATCH options for postprocessing (CPU-only job)
    postprocess_sbatch_opts = [
        f'-A {account}',
        f'-p {partition}',
        f'--gres=gpu:{gpu_type}:{num_gpus}',
        f'-N {num_nodes}',
        f'-n {num_tasks}',
        f'-c {cpus_per_task}',
        f'--qos={qos}',
        f'-t {time}',
        f'--mem={mem}',
        '--parsable'
    ]
    
    postprocess_sbatch_opts_str = ' '.join(postprocess_sbatch_opts)
    
    # Add postprocessing job with dependency on all eval jobs
    postprocess_job = (
        f'postprocess_jobid=$(sbatch {postprocess_sbatch_opts_str} '
        f'--dependency=afterok:{dependency_str} '
        f'--comment="postprocess_best_of_N" '
        f'{script_runner} \'{postprocess_cmd}\') '
        f'&& echo $postprocess_jobid'
    )
    
    lines.append(postprocess_job)
    lines.append('')
    lines.append('echo "All jobs submitted!"')
    lines.append(f'echo "Evaluation jobs: {len(n_vals)}"')
    lines.append('echo "Postprocessing will run after all evaluations complete"')
    
    # Write the script
    script_content = '\n'.join(lines) + '\n'

    # if output_file is not provided, use wandb_name + timestamp
    if output_file is None:
        output_file = os.path.join(output_file_dir, f'eval_libero_best_of_N_{wandb_name}.sh')
    
    with open(output_file, 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(output_file, 0o755)
    
    print(f"Generated sbatch script: {output_file}")
    print(f"Evaluation jobs: {len(n_vals)}")
    print(f"N values: {n_vals}")
    print(f"Wandb run ID: {wandb_run_id}")
    print(f"\nTo submit all jobs, run:")
    print(f"  bash {output_file}")
    print(f"\nNote: Evaluation jobs run in offline mode.")
    print(f"      Postprocessing job will aggregate and log to wandb online.")
    
    return output_file


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate sbatch script for best-of-N evaluation'
    )
    
    parser.add_argument('--n_vals', type=int, nargs='+', required=True,
                       help='List of n values to evaluate (e.g., 1 2 4 8 16)')
    parser.add_argument('--actor_restore_path', type=str, required=True,
                       help='Path to actor checkpoint')
    parser.add_argument('--critic_restore_path', type=str, required=True,
                       help='Path to critic checkpoint')
    parser.add_argument('--env_name', type=str, required=True,  
                       help='Environment name')
    parser.add_argument('--task_name', type=str, required=True,
                       help='Task name(s)')
    parser.add_argument('--wandb_name', type=str, required=True,
                       help='Base wandb run name')
    parser.add_argument('--wandb_run_id', type=str, default=None,
                       help='Wandb run id (optional)')
    parser.add_argument('--wandb_entity', type=str, default='yajatyadav',
                       help='Wandb entity')
    parser.add_argument('--wandb_project', type=str, default='multitask_RL',
                       help='Wandb project')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                       help='Base output directory')
    
    args = parser.parse_args()
    
    # Call the function with parsed arguments
    output_file = generate_sbatch_script(
        n_vals=args.n_vals,
        actor_restore_path=args.actor_restore_path,
        critic_restore_path=args.critic_restore_path,
        env_name=args.env_name,
        task_name=args.task_name,
        wandb_name=args.wandb_name,
        wandb_run_id=args.wandb_run_id,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        output_dir=args.output_dir,
    )
    
    print(f"\n✅ Script generation complete: {output_file}")