#!/bin/bash
#SBATCH -A co_rail
#SBATCH -p savio4_gpu
#SBATCH --gres=gpu:A5000:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --qos=rail_gpu4_high
#SBATCH -t 24:00:00
#SBATCH --mem=60G
#SBATCH --requeue
#SBATCH -o ../slurm_logs/slurm-%j.out
#SBATCH -e ../slurm_logs/slurm-%j.err

export WANDB_SERVICE_WAIT=86400
export XLA_PYTHON_PREALLOCATE=false
# use osmesa backend for rendering

uv run main.py \
--exp_name_prefix=test_full_libero_Q__best_of_1 \
--run_group=test_full_libero_Q \
--env_name=all_libero-* \
--task_name='' \
--augment_negative_demos=False \
\
--use_pixels=False \
--use_proprio=True \
--use_language=True \
--use_mj_sim_state=False \
\
--online_steps=0 \
--eval_episodes=50 \
--eval_interval=100000 \
--num_parallel_envs=25 \
--video_episodes=0 \
\
--agent.batch_size=256 \
--agent.actor_type=best-of-n \
--agent.actor_num_samples=1 \
--agent.encoder=language_and_proprio \
