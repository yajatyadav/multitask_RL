"""
Generate sbatch job script for evaluating best-of-N with different n values.
Each n value gets its own sbatch job.
"""
import os
import argparse
from typing import List
import time as time_module
import wandb    
def generate_sbatch_script(
    n_vals: List[int],
    actor_restore_path: str,
    critic_restore_path: str,
    wandb_name: str,
    wandb_run_id: str = None,
    output_file: str = None,
    output_file_dir: str = 'scripts/shell_scripts',
    env_name: str = 'libero_90-living_room_scene1',
    task_name: str = 'pick_up_the_alphabet_soup_and_put_it_in_the_basket|pick_up_the_ketchup_and_put_it_in_the_basket',
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
):
    """
    Generate a shell script with sbatch commands for each n value.
    
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
        output_file: Output shell script filename
    """
    
    # Environment variables for WANDB robustness
    wandb_env_vars = [
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
    
    # SBATCH options
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
    lines = ['#!/bin/bash', '']

    # initialize wandb run and get run id
    if wandb_run_id is None:
        run = wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            group=wandb_group,
            name=wandb_name,
        )
        wandb_run_id = run.id
    
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
            # f'--wandb_group {wandb_group} '
            # f'--wandb_name {wandb_name} '
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
    print(f"Total jobs: {len(n_vals)}")
    print(f"N values: {n_vals}")
    print(f"\nTo submit all jobs, run:")
    print(f"  bash {output_file}")
    
    
    return output_file


def main():
    """Example usage with configurable parameters."""
    
    # Configuration - EDIT THESE VALUES
    root_dir = '/home/yajatyadav/multitask_reinforcement_learning/multitask_RL/exp/multitask_RL'
    
    # Actor paths
    onetask_actor_ckpt = 'bcflowactor_only/libero_90-living_room_scene1/bcflowactor_livingroomscene1__alphabet_soup_25_demos_IMAGE_sd00020251227_144306/params_140000.pkl'
    twotask_actor_ckpt = 'bcflowactor_only/libero_90-living_room_scene1/bcflowactor_livingroomscene1__alphabet_soup_ketchup_25_demos_IMAGE_sd00020251227_144347/params_140000.pkl'
    
    # Critic paths
    onetask_critic_no_aug_ckpt = 'instruction_following_Q/libero_90-living_room_scene1/libero90_livingroomscene1_singletask_none_augmentation_IMAGE_sd00020251226_214011/params_125000.pkl'
    two_task_critic_no_aug_ckpt = 'instruction_following_Q/libero_90-living_room_scene1/libero90_livingroomscene1_twotask_none_augmentation_IMAGE_sd00020251226_214200/params_125000.pkl'
    
    # TODO: EDIT THESE FOR YOUR EXPERIMENT
    n_vals = [1, 2, 4, 8, 16, 32, 64, 128]
    actor_ckpt = twotask_actor_ckpt
    critic_ckpt = two_task_critic_no_aug_ckpt
    
    actor_restore_path = os.path.join(root_dir, actor_ckpt)
    critic_restore_path = os.path.join(root_dir, critic_ckpt)
    
    suffix = 'twotask_no_aug_actor_25_demos_140k_step_critic_125k_step'
    wandb_name = f'eval_libero90_livingroomscene1_{suffix}'
    
    env_name = 'libero_90-living_room_scene1'
    task_name = 'pick_up_the_alphabet_soup_and_put_it_in_the_basket|pick_up_the_ketchup_and_put_it_in_the_basket'
    
    # Generate the script
    generate_sbatch_script(
        n_vals=n_vals,
        actor_restore_path=actor_restore_path,
        critic_restore_path=critic_restore_path,
        env_name=env_name,
        task_name=task_name,
        wandb_name=wandb_name,
        output_file='eval_best_of_n_jobs.sh'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate sbatch script for best-of-N evaluation'
    )
    
    # Option 1: Use command line arguments
    parser.add_argument('--n_vals', type=int, nargs='+', required=True,
                       help='List of n values to evaluate (e.g., 1 2 4 8 16)')
    parser.add_argument('--actor_restore_path', type=str,
    required=True,
                       help='Path to actor checkpoint')
    parser.add_argument('--critic_restore_path', type=str,
    required=True,
                       help='Path to critic checkpoint')
    parser.add_argument('--env_name', type=str,
    required=True,  
                       help='Environment name')
    parser.add_argument('--task_name', type=str,
    required=True,
                       help='Task name(s)')
    parser.add_argument('--wandb_name', type=str,
    required=True,
                       help='Base wandb run name')
    parser.add_argument('--output_file', type=str,
    required=False,
                       help='Output shell script filename')
    parser.add_argument('--wandb_run_id', type=str, default=None,
                       help='Wandb run id, if provided evals will get logged under this run id')
    
    args = parser.parse_args()
    
    # If no command line arguments provided, use the hardcoded configuration
    if args.n_vals is None:
        raise ValueError("n_vals must be provided")
    else:
        generate_sbatch_script(
            n_vals=args.n_vals,
            actor_restore_path=args.actor_restore_path,
            critic_restore_path=args.critic_restore_path,
            env_name=args.env_name,
            task_name=args.task_name,
            wandb_name=args.wandb_name,
            wandb_run_id=args.wandb_run_id,
        )