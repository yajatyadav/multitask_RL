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
--exp_name_prefix=libero90_livingroomscene1_exhaustive_augmentation_IMAGE_ \
--run_group=instruction_following_Q \
--env_name=libero_90-living_room_scene1 \
--task_name='' \
--augmentation_type=exhaustive \
\
--use_pixels=True \
--use_proprio=True \
--use_language=True \
--use_mj_sim_state=False \
\
--offline_steps=1000000 \
--eval_interval=50000 \
--save_interval=50000 \
\
--horizon_length=5 \
--agent=agents/acifql.py \
--agent.encoder=combined_encoder_small \
--agent.expectile=0.9 \