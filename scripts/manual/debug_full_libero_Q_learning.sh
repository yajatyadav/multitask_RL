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
export XLA_PYTHON_CLIENT_PREALLOCATE=false

uv run main.py \
--exp_name_prefix=full_libero_Q_learning_acifql_23_scenes \
--run_group=DEBUG_full_libero_Q_learning_acifql_23_scene_all \
--env_name=all_libero-* \
--task_name='' \
--augment_negative_demos=True \
\
--use_pixels=True \
--use_proprio=True \
--use_language=True \
--use_mj_sim_state=False \
\
--online_steps=0 \
--eval_episodes=50 \
--video_episodes=3 \
--save_interval=25000 \
--eval_interval=25000 \
--num_parallel_envs=5 \
\
--agent.encoder=combined_encoder_small \