from envs.env_utils import make_env_and_datasets
import time
from tqdm import tqdm
import numpy as np
env_name = "libero_90"
keys_to_load = ['agentview_rgb', 'eye_in_hand_rgb', 'proprio', 'language']




augment_negative_demos = False
num_parallel_envs = 5

# make the dataset

_, eval_env, dataset, _ = make_env_and_datasets(env_name, '', keys_to_load=keys_to_load, augmentation_type='none', augmentation_reward=False, batch_level_sampling=True, num_parallel_envs=num_parallel_envs, language_embedder='bert')