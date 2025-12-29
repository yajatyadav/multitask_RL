#!/bin/bash
# Example workflow for running best-of-N evaluations
# This script:
# 1. Generate the sbatch script
# 2. Submit jobs to SLURM
# 3. Wait for jobs to complete
# 4. Aggregate results

# Step 1: Generate the sbatch script
echo "Step 1: Generating sbatch script..."

# define all variables outside
n_vals=(1 2 4 8 16 32 64 128)
actor_restore_path="exp/multitask_RL/bcflowactor_only/libero_90-living_room_scene1/bcflowactor_livingroomscene1__alphabet_soup_25_demos_IMAGE_sd00020251228_175933/params_140000.pkl"
critic_1_restore_path="exp/multitask_RL/instruction_following_Q/libero_90-living_room_scene1/libero90_livingroomscene1_singletask_one_other_task__augmentation_IMAGE_sd00020251226_215816/params_250000.pkl"
env_name="libero_90-living_room_scene1"
task_name="pick_up_the_alphabet_soup_and_put_it_in_the_basket"
wandb_name_1="eval_libero90_livingroomscene1_singletask_one_other_task_augmentation_actor_25_demos_140k_step_critic_250k_step"


# uv run evaluation/brc_eval_scripts/generate_eval_sbatch.py \
#   --n_vals ${n_vals[@]} \
#   --actor_restore_path ${actor_restore_path} \
#   --critic_restore_path ${critic_1_restore_path} \
#   --env_name ${env_name} \
#   --task_name ${task_name} \
#   --wandb_name ${wandb_name_1} \

  
critic_2_restore_path="exp/multitask_RL/instruction_following_Q/libero_90-living_room_scene1/libero90_livingroomscene1_singletask_none_augmentation_IMAGE_sd00020251226_214011/params_125000.pkl"
wandb_name_2="eval_libero90_livingroomscene1_singletask_no_aug_actor_25_demos_140k_step_critic_125k_step"

uv run evaluation/brc_eval_scripts/generate_eval_sbatch.py \
  --n_vals ${n_vals[@]} \
  --actor_restore_path ${actor_restore_path} \
  --critic_restore_path ${critic_2_restore_path} \
  --env_name ${env_name} \
  --task_name ${task_name} \
  --wandb_name ${wandb_name_2} \