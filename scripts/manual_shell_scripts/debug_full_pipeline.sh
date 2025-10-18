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

# set the GPU ID
gpu_id=6
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=$gpu_id
export MUJOCO_EGL_DEVICE_ID=$gpu_id
export CUDA_VISIBLE_DEVICES=$gpu_id

# Environment variables
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB__SERVICE_WAIT=86400
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

export OMP_NUM_THREADS=1 
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 
export NUMEXPR_NUM_THREADS=1


# Run the training script
uv run scripts/train_offline.py \
--use_wandb=True \
--exp_name_prefix='debug_book_right_compartment_caddy__study_scene1_2_' \
--run_group=debug_book_right_compartment_caddy__study_scene1__ \
--seed=42 \
--save_dir=exp/ \
\
--pixel_observations=False \
--offline_steps=1000000 \
--log_interval=5000 \
--eval_interval=1000 \
--save_interval=1000000 \
--num_input_output_to_log=3 \
\
--eval_episodes=2 \
--num_steps_wait=10 \
--video_frame_skip=3 \
--eval_temperature=1.0 \
--task_suite_name="libero_90" \
--task_name="STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy" \
\
--data_root_dir=../datasets \
--train_dataset_mix='{"libero_90__book_right_compartment_caddy__study_scene1": 1.0}' \
--do_validation=False \
--balance_datasets=True \
--batch_size=512 \
--num_workers=8 \
--do_image_aug=False \
--binarize_gripper=True \
\
--agent=agents/iql.py \
--agent.tanh_squash=True \
--agent.state_dependent_std=False \
--agent.const_std=False \
--agent.alpha=0.5 \
--agent.expectile=0.7 \
--agent.lr=3e-4 \
--agent.encoder="state_space_encoder" \