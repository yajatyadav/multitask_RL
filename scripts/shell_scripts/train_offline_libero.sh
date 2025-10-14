export CUDA_VISIBLE_DEVICES=5,6
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run scripts/train_offline.py \
--run_group=Debug \
--exp_name_prefix='train_test_libero_ON_single_task__' \
--seed=42 \
--save_dir=exp/ \
\
--agent=agents/iql.py \
--offline_steps=100000 \
--log_interval=100 \
--eval_interval=10000 \
--save_interval=50000 \
--num_input_output_to_log=3 \
\
--eval_episodes=20 \
--num_steps_wait=10 \
--video_frame_skip=3 \
--eval_temperature=1.0 \
--task_suite_name="libero_90" \
--task_name="KITCHEN_SCENE5_put_the_black_bowl_on_the_plate" \
\
--data_root_dir=/raid/users/yajatyadav/tensorflow_datasets/ \
--train_dataset_mix='{"libero_90_put_the_black_bowl_on_the_plate": 1.0}' \
--do_validation=False \
--balance_datasets=True \
\
--batch_size=256 \
--num_workers=16 \
--do_image_aug=True \