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
--exp_name_prefix=bcflowactor_livingroomscene1__alphabet_soup_and_ketchup_NO_LANG_CONDITIONING_25_demos_IMAGE_ \
--run_group=bcflowactor_unconditional \
--env_name=libero_90-living_room_scene1 \
--task_name='pick_up_the_alphabet_soup_and_put_it_in_the_basket|pick_up_the_ketchup_and_put_it_in_the_basket' \
--augmentation_type=none \
--num_demos_to_use_per_task=25 \
\
--use_pixels=True \
--use_proprio=True \
--use_language=False \
--use_mj_sim_state=False \
\
--offline_steps=250000 \
--eval_interval=20000 \
--save_interval=20000 \
\
--horizon_length=5 \
--agent=agents/acbcflowactor.py \
--agent.encoder=image_proprio_small \





# uv run main.py \
# --exp_name_prefix=bcflowactor_livingroomscene1__alphabet_soup_ketchup_25_demos_IMAGE_ \
# --run_group=bcflowactor_only \
# --env_name=libero_90-living_room_scene1 \
# --task_name='pick_up_the_alphabet_soup_and_put_it_in_the_basket|pick_up_the_ketchup_and_put_it_in_the_basket' \
# --augmentation_type=none \
# --num_demos_to_use_per_task=25 \
# \
# --use_pixels=True \
# --use_proprio=True \
# --use_language=True \
# --use_mj_sim_state=False \
# \
# --offline_steps=1000000 \
# --eval_interval=20000 \
# --save_interval=20000 \
# \
# --horizon_length=5 \
# --agent=agents/acbcflowactor.py \
# --agent.encoder=combined_encoder_small \
