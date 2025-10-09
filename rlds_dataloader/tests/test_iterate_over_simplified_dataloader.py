import rlds_dataloader.dataloader as rlds_data_loader



# default is mean/std action-proprio normalization. Actions only first 6 dimensions normalized
dataloader_config = {
    "data_root_dir": "/raid/users/yajatyadav/datasets/raw_libero/raw_libero_RLDS",
    "dataset_mix": { 
        "libero_90": 1.0, # weights are in terms of odds
        "libero_object": 0.08,
        "libero_spatial": 0.08,
        "libero_goal": 0.08,
    },
    "batch_size": 32,

    "num_workers": 16, # dataloader workers
    "seed": 42,
    "do_image_aug": True, # check file for sequence: dict(
            #     random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
            #     random_brightness=[0.2],
            #     random_contrast=[0.8, 1.2],
            #     random_saturation=[0.8, 1.2],
            #     random_hue=[0.05],
            #     augment_order=[
            #         "random_resized_crop",
            #         "random_brightness",
            #         "random_contrast",
            #         "random_saturation",
            #         "random_hue",
            #     ],
            # )
    "binarize_gripper": True, # binarizes to 0 and 1 (by scanning for transitions), and then normalizes to -1 and 1
    "train": True, # for picking the right split
}


dataloader = rlds_data_loader.create_data_loader(
    dataloader_config,
)

data_iter = iter(dataloader)
batch = next(data_iter)

import tensorflow as tf
for i, batch in enumerate(data_iter):
    if i > 10:
        break
    
    task = batch['task']
    action = batch['action']
    reward = batch['reward']
    is_terminal = batch['is_terminal']
    curr_and_next_observation = batch['observation']

    observation = tf.nest.map_structure(lambda x: x[:, 0], curr_and_next_observation)
    next_observation = tf.nest.map_structure(lambda x: x[:, 1], curr_and_next_observation)

    batch_of_samples = {
        "task": task,
        "action": action,
        "reward": reward,
        "observation": observation,
        "next_observation": next_observation,
        "is_terminal": is_terminal,
    }

    batch_size = tf.shape(batch_of_samples["action"])[0]
    list_of_samples = [
        tf.nest.map_structure(lambda x: x[i], batch_of_samples)
        for i in range(batch_size)
        ]
    
    for sample in list_of_samples:
        reward = sample['reward']
        if reward > 0:
            print(sample['task'])
            print(sample['observation']['pad_mask'])
            print(sample['next_observation']['pad_mask'])
            print(reward)
            good_sample = sample