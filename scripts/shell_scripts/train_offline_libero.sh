export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run scripts/train_offline.py \
--run_group=Debug \
--seed=42 \
--save_dir=exp/ \
--offline_steps=100000 \
--log_interval=10 \
--eval_interval=1000 \
--save_interval=100000 \
--agent=agents/iql.py \
--data_root_dir=/raid/users/yajatyadav/datasets/raw_libero/raw_libero_RLDS \
--train_dataset_mix='{"libero_90": 0.759, "libero_object": 0.08, "libero_spatial": 0.08, "libero_goal": 0.08}' \
--val_dataset_mix='{"libero_90": 0.625, "libero_object": 0.125, "libero_spatial": 0.125, "libero_goal": 0.125}' \
--batch_size=256 \
--num_workers=16 \
--do_image_aug=True \
--binarize_gripper=True \