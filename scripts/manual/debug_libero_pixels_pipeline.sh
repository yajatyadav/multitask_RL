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

uv run main.py \
--exp_name_prefix=test_PIXELS_baseandwristcam_kitchen_scene2_open_the_top_drawer_of_the_cabinet_augment_negative_demos_False_best_of_1 \
--run_group=debug_kitchen_scene_2_singletask_ \
--env_name=libero_90-kitchen_scene2 \
--task_name=open_the_top_drawer_of_the_cabinet \
--augment_negative_demos=False \
\
--use_pixels=True \
--use_proprio=False \
--use_mj_sim_state=False \
\
--online_steps=0 \
--eval_interval=50000 \
--num_parallel_envs=1 \
--video_episodes=1 \
\
--agent.actor_type=best-of-n \
--agent.actor_num_samples=1 \
--agent.encoder=impala_debug \
