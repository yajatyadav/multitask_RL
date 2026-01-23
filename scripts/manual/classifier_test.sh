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


# use_proprio=True so that actor agent is initialized correctly...

uv run main.py \
--exp_name_prefix=DEBUG_libero90_livingroomscene1_two_tasks_IMAGE_NO_PROPRIO_ \
--run_group=transition_classifier \
--env_name=libero_90-living_room_scene1 \
--task_name='pick_up_the_alphabet_soup_and_put_it_in_the_basket|pick_up_the_ketchup_and_put_it_in_the_basket' \
--augmentation_type=none \
\
--use_pixels=True \
--use_proprio=True \
--use_language=True \
--use_mj_sim_state=False \
\
--offline_steps=2000 \
--log_interval=50 \
--eval_interval=50 \
--num_parallel_envs=5 \
--save_interval=-1 \
\
--horizon_length=5 \
--agent=agents/trans_classifier.py \
--agent.encoder=image_only_small \
--agent.actor_restore_path=exp/multitask_RL/bcflowactor_only/libero_90-living_room_scene1/bcflowactor_livingroomscene1__alphabet_soup_ketchup_25_demos_IMAGE_sd00020251227_144347/params_140000.pkl