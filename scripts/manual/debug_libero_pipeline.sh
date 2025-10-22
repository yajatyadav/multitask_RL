#!/bin/bash
#SBATCH -A co_rail
#SBATCH -p savio4_gpu
#SBATCH --gres=gpu:A5000:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --qos=rail_gpu4_high
#SBATCH -t 24:00:00
#SBATCH --mem=60G
#SBATCH --requeue
#SBATCH -o ../slurm_logs/slurm-%j.out
#SBATCH -e ../slurm_logs/slurm-%j.err

export WANDB_SERVICE_WAIT=86400
export XLA_PYTHON_PREALLOCATE=false

uv run main.py \
--exp_name_prefix=best_of_16_image_aug__ \
--run_group=debug_qc_libero \
--env_name=libero_90-study_scene1-pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy \
\
--online_steps=0 \
--eval_interval=100000 \
\
--agent.actor_type=best-of-n \
--agent.actor_num_samples=16 \
