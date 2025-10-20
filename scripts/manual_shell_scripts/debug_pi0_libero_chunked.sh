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
gpu_id=2
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
--exp_name_prefix='pi0_libero_50_horizon_action_chunk_5__best_of_3__expectile_0.7__LR_3e-4_state_based_' \
--run_group=debug_pi0_libero_pipeline__ \
--seed=42 \
--save_dir=exp/ \
\
--pixel_observations=False \
--offline_steps=1000000 \
--log_interval=5000 \
--eval_interval=100000 \
--save_interval=1000000 \
--num_input_output_to_log=3 \
\
--eval_episodes=30 \
--num_steps_wait=10 \
--video_frame_skip=3 \
--eval_temperature=1.0 \
--task_suite_name="libero_90" \
--task_name="STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy" \
\
--data_root_dir=../datasets \
--train_dataset_mix_name='libero_90__book_right_compartment_caddy__study_scene1' \
--do_validation=False \
--balance_datasets=True \
--batch_size=256 \
--num_workers=8 \
--do_image_aug=False \
--binarize_gripper=True \
\
--agent=agents/iql_pi0actor_chunked.py \
--agent.action_chunk_length=5 \
--agent.expectile=0.7 \
--agent.lr=3e-4 \
--agent.encoder="state_space_encoder" \
--agent.pi0_checkpoint_dir='../checkpoints/pi0_all_libero_but_10_flipped_train_split/pi0_all_libero_but_10_flipped_train_split__batch_64_steps_30k/10000' \
--agent.pi0_config_name='pi0_libero_mine' \
--agent.pi0_best_of_n_samples=3 \
--agent.pi0_action_horizon=50 \