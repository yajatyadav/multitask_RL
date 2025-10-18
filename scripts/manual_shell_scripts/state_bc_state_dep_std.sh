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


# Environment variables
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB__SERVICE_WAIT=86400
export XLA_PYTHON_CLIENT_PREALLOCATE=false

export OMP_NUM_THREADS=1 
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 
export NUMEXPR_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl


# Run the training script
uv run scripts/train_offline.py \
--exp_name_prefix='BRC_State_space_BC_ON_single_libero_task__value_actor_512_tanh_squashing__state_dep_std_' \
--run_group=Debug \
--seed=42 \
--save_dir=exp/ \
\
--offline_steps=1000000 \
--log_interval=20 \
--eval_interval=10000 \
--save_interval=1000000 \
--num_input_output_to_log=3 \
\
--eval_episodes=20 \
--num_steps_wait=10 \
--video_frame_skip=3 \
--eval_temperature=1.0 \
--task_suite_name="libero_90" \
--task_name="KITCHEN_SCENE1_put_the_black_bowl_on_the_plate" \
\
--data_root_dir=/global/scratch/users/yajatyadav/research/multitask_reinforcement_learning/datasets \
--train_dataset_mix='{"libero_90__black_bowl_on_plate_kitchen_scene1": 1.0}' \
--do_validation=False \
--balance_datasets=True \
--batch_size=256 \
--num_workers=8 \
--do_image_aug=False \
--binarize_gripper=True \
\
--tanh_squash=True \
--state_dependent_std=True \
--const_std=True \
--alpha=0.0 \
--lr=3e-4 \
--encoder="state_space_encoder" \