from envs.env_utils import make_env_and_datasets

env_name = 'libero_90-living_room_scene1-pick_up_the_alphabet_soup_and_put_it_in_the_basket|libero_90-living_room_scene1-pick_up_the_ketchup_and_put_it_in_the_basket'
task_name = ''
language_embedder = 'bert'
keys_to_load = ['language', 'proprio']
horizon_length = 5
discount = 0.99
actor_encoder = 'image_proprio_small'

augmentation_type = 'none'
augmentation_reward = False

_, eval_env, dataset, _, names_to_return = make_env_and_datasets(env_name, task_name, language_embedder, augmentation_type, augmentation_reward, keys_to_load=keys_to_load, demo_nums_to_use_per_task=[0], augmentation_dict=None, is_notebook=True)
