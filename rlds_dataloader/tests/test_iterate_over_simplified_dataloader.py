import rlds_dataloader.dataloader as rlds_data_loader
import tqdm
import time
import numpy as np

# default is mean/std action-proprio normalization. Actions only first 6 dimensions normalized
dataloader_config = {
    "data_root_dir": "../datasets/",
    "dataset_mix": { 
        "libero_90__black_bowl_on_plate_kitchen_scene1": 1.0, # weights are in terms of odds
    },
    "batch_size": 256,
    "balance_datasets": True,
    "num_workers": 8, # dataloader workers
    "prefetch_factor": 2,
    "seed": 42,
    "do_image_aug": False, # check file for sequence: dict(
    "binarize_gripper": True, # binarizes to 0 and 1 (by scanning for transitions), and then normalizes to -1 and 1
    "train": True, # for picking the right split
    "text_encoder": "one_hot_libero",
}


dataloader = rlds_data_loader.create_data_loader(
    dataloader_config,
    load_images=False,
    load_proprio=False,
    load_language=False,
    normalize_batches=True,
)

data_iter = iter(dataloader)
print("Starting to iterate over the dataloader")
first_iter_start_time = time.time()
for i, batch in tqdm.tqdm(enumerate(data_iter)):
    if i == 0:
        first_iter_end_time = time.time()
        print(f"Time taken to iterate over the first batch: {first_iter_end_time - first_iter_start_time} seconds")
    if i == 1:
        start_time = time.time()
    if i == 1000:
        break
    
    # actions = batch['actions']
    # proprio = batch['observations']['proprio']
    # sim_state = batch['observations']['sim_state']
    
    # print(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # print max/min of each above
    # print(f"Proprio max: {np.max(proprio, axis=0)}")
    # print(f"Proprio min: {np.min(proprio, axis=0)}")
    # print(f"Sim state max: {np.max(sim_state, axis=0)}")
    # print(f"Sim state min: {np.min(sim_state, axis=0)}")
    # print(f"Actions max: {np.max(actions, axis=0)}")
    # print(f"Actions min: {np.min(actions, axis=0)}")



end_time = time.time()
print(f"Time taken to iterate over 1000 batches: {end_time - start_time} seconds")