from utils.datasets import MultiDatasetWrapper
from envs.libero_utils import get_dataset, get_single_dataset

# env_name = 'libero_90-living_room_scene1-pick_up_the_alphabet_soup_and_put_it_in_the_basket|libero_90-living_room_scene1-pick_up_the_ketchup_and_put_it_in_the_basket'
env_name = 'libero_90'
task_name = ''
language_embedder = 'bert'
keys_to_load = ['language', 'proprio', 'agentview_rgb', 'eye_in_hand_rgb']
horizon_length = 5
discount = 0.99
actor_encoder = 'image_proprio_small'

# datasets = []
# for env in env_name.split('|'):
#     dataset = get_single_dataset(None, env, task_name, language_embedder, 'none', False, keys_to_load, [0], None)
#     datasets.append(dataset)

multi_dataset = get_dataset(None, env_name, task_name, language_embedder, 'none', False, keys_to_load, batch_level_sampling=True, demo_nums_to_use_per_task=None, augmentation_dict=None)
# Create wrapper with uniform sampling
# wrapper = MultiDatasetWrapper(datasets)

# Create wrapper with weighted sampling (60% from dataset1, 40% from dataset2)
# wrapper = MultiDatasetWrapper(
#     [dataset1, dataset2],
#     weights=[0.6, 0.4],
#     batch_level_sampling=True,  # Each batch will have exactly 60/40 split
# )

# Sample a batch
# batch = wrapper.sample(batch_size=256)

# Sample sequences
batch = multi_dataset.sample_sequence(batch_size=256, sequence_length=5, discount=0.99)

# Track which dataset each sample came from
# wrapper = MultiDatasetWrapper(
#     [dataset1, dataset2],
#     weights=[0.5, 0.5],
#     return_dataset_indices=True,
# )
# batch = wrapper.sample(128)
print(batch['dataset_indices'])  # Array showing 0 or 1 for each sample


import pdb; pdb.set_trace()