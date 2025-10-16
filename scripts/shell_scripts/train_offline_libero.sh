export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

uv run scripts/train_offline.py \
--run_group=Debug \
--exp_name_prefix='State_space_BC_ON_single_libero_task__value_actor_512_tanh_squashing__' \
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
\
--batch_size=256 \
--num_workers=24 \
--do_image_aug=False \