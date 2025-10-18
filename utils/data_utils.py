import numpy as np
import os
import json
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "libero"))

# used by the dataloader during training / in the eval script before feeding into the policy
LIBERO_ENV_RESOLUTION = 128  # set this to resolution used to render training data
# utility func b/c openvla dataloader wants one dataset in a mixture to be the "primary" one and have weight 1.0
def ratios_to_odds_mixture(ratios):
    keys = sorted(ratios.keys())
    first_val = ratios[keys[0]]
    new_dict = {keys[0]: 1.0}
    for key in keys[1:]:
        new_dict[key] = ratios[key] / first_val
    return new_dict

def flatten_dict(d, parent_key='', sep='/'):
    """
    Flatten a nested dictionary, joining keys with separator.
    
    Example:
        {'observations': {'proprio': [1,2,3], 'image': [...]}}
    becomes:
        {'observations/proprio': [1,2,3], 'observations/image': [...]}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d, sep='/'):
    """
    Unflatten a dictionary with separator-joined keys into nested structure.
    
    Example:
        {'observations/proprio': [1,2,3], 'observations/image': [...]}
    becomes:
        {'observations': {'proprio': [1,2,3], 'image': [...]}}
    
    Args:
        d: Flattened dictionary with separator-joined keys
        sep: Separator used to join keys (default: '/')
    
    Returns:
        Nested dictionary
    """
    result = {}
    
    for flat_key, value in d.items():
        # Split the key by separator
        keys = flat_key.split(sep)
        
        # Navigate/create nested structure
        current = result
        for i, key in enumerate(keys[:-1]):
            # Create nested dict if it doesn't exist
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value
        current[keys[-1]] = value
    
    return result



# language encoder that encodes str into vector; used by the dataloader during training / in the eval script before feeding into the policy
import tensorflow_hub as hub
import tensorflow_text
MUSE_MODEL = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
class MuseEmbedding:
    @staticmethod
    def encode(strings):
        return MUSE_MODEL(strings).numpy()

# setup one-hot encoder
NUM_UNIQUE_LIBERO_TASKS = 112
from libero.libero import benchmark
libero_benchmark_dict = benchmark.get_benchmark_dict()
all_libero_languages = set()
benchmarks = sorted(["libero_spatial", "libero_object", "libero_goal", "libero_90", "libero_10"]) # sorted() to retain same order
for benchmark_name in benchmarks:
    benchmark = libero_benchmark_dict[benchmark_name]()
    num_tasks = benchmark.get_num_tasks()
    for i in range(num_tasks):
        task = benchmark.get_task(i)
        task_language = task.language
        if task_language not in all_libero_languages:
            all_libero_languages.add(task_language)
        else:
            pass
            # print(f"Duplicate language: {task_language}, found in task {benchmark.get_task_names()[i]} of benchmark {benchmark_name}")
# once set constructed, sort all keys alphabetically before assinging one-hot label, will ensure consistent ordering
all_libero_languages = sorted(all_libero_languages)
all_libero_languages = {language: i for i, language in enumerate(all_libero_languages)}
assert len(all_libero_languages) == NUM_UNIQUE_LIBERO_TASKS, f"Expected {NUM_UNIQUE_LIBERO_TASKS} unique libero tasks, but found {len(all_libero_languages)}"

class OneHotEmbedding_Libero:
    @staticmethod
    def encode(strings):
        def standardize_string(s):
            if isinstance(s, bytes):
                s = s.decode("utf-8")
            s = s.lower().replace("_", " ")
            return s
        indices = [all_libero_languages[standardize_string(s)] for s in strings]
        return np.eye(NUM_UNIQUE_LIBERO_TASKS)[indices]



def get_language_encoder(encoder_type: str):
    if encoder_type == "muse":
        return MuseEmbedding
    elif encoder_type == "one_hot_libero":
        return OneHotEmbedding_Libero
    else:
        raise ValueError(f"Invalid Text Encoder type: {encoder_type}")




norm_stats_root_dir = os.path.join(os.getcwd(), 'dataset_stats')
ALL_NORM_STATS = {}
for dataset_name in os.listdir(norm_stats_root_dir):
    norm_stats_folder = os.path.join(norm_stats_root_dir, dataset_name)
    norm_stats_file = os.path.join(norm_stats_folder, "stats.json")
    assert os.path.exists(norm_stats_file), f"Norm stats file {norm_stats_file} does not exist"
    with open(norm_stats_file, "r") as f:
        norm_stats = json.load(f)
    ALL_NORM_STATS[dataset_name] = norm_stats

ALL_NORM_STATS["old_libero_norm_stats"] = {
    "observations/proprio": {
        "mean": [-0.07594718039035797, 0.01788640394806862, 0.9031365513801575, 2.9963114261627197, -0.11735942214727402, -0.09229698777198792, 0.02637086622416973, -0.026728888973593712],
        "std": [0.11286084353923798, 0.14507046341896057, 0.2954331636428833, 0.3924177587032318, 0.9082265496253967, 0.31110888719558716, 0.014611119404435158, 0.014481362886726856],
    },
    "actions": {
        "min": [-0.6131696268081666, -0.9530494272947312, -2.0260506366729736, -3.671418887233734, -2.3304990768432616,  -0.6162669191360475, -1.0],
        "max": [ 0.9955515738487244, 1.0174857356309892, 0.4062486431121828, -1.433714524137974, 2.7279676017761236, 1.2093748165130611, 0.9996],
        "mean": [0.0896475613117218, 0.0353005975484848, -0.9628857970237732, -2.993234872817993, 0.12035975605249405, 0.08618046343326569, 0.020620649680495262],
        "std": [0.30294424295425415,  0.3951551020145416, 0.5384603142738342, 0.3950735330581665,  0.9109124541282654,  0.3203604817390442, 0.9997873902320862],
    },
}

logged_norm_stats = [False, False]

# TODO(YY): commenting out image normalization for now- this is really killing the dataloader speed...maybe try to integrate into TFDS pipeline as a frame_transform?
def normalize_libero_batch(batch, dataset_name: str):
    if not logged_norm_stats[0]:
        logged_norm_stats[0] = True
        print(f"😂😂 TRAINING NORMALIZATION: Using norm stats for dataset {dataset_name}")
    norm_stats = ALL_NORM_STATS[dataset_name]

    # first, flatten the batch dict
    batch = flatten_dict(batch, sep="/")

    # normalize proprio and sim state using mean/std, if they have been loaded in (at least one of them must be loaded in)
    if "observations/proprio" in batch:
        proprio = batch["observations/proprio"]
        proprio = normalize_proprio(proprio, norm_stats)
        batch["observations/proprio"] = proprio
    
    if "observations/sim_state" in batch:
        sim_state = batch["observations/sim_state"]
        sim_state = normalize_sim_state(sim_state, norm_stats)
        batch["observations/sim_state"] = sim_state

    
    if "next_observations/proprio" in batch:
        next_proprio = batch["next_observations/proprio"]
        next_proprio = normalize_proprio(next_proprio, norm_stats)
        batch["next_observations/proprio"] = next_proprio

        
    if "next_observations/sim_state" in batch:
        next_sim_state = batch["next_observations/sim_state"]
        next_sim_state = normalize_sim_state(next_sim_state, norm_stats)
        batch["next_observations/sim_state"] = next_sim_state

    # normalize actions using min/max
    action = batch["actions"]
    action = normalize_action_min_max(action, norm_stats)
    batch["actions"] = action

    # unflatten the batch
    batch = unflatten_dict(batch, sep="/")
    return batch

def normalize_libero_eval_obs_for_agent(obs, dataset_name: str):
    if not logged_norm_stats[1]:
        logged_norm_stats[1] = True
        print(f"😂😂 EVAL NORMALIZATION: Using norm stats for dataset {dataset_name}")
    norm_stats = ALL_NORM_STATS[dataset_name]

    if "proprio" in obs:
        proprio = obs["proprio"]
        proprio = normalize_proprio(proprio, norm_stats)
        obs["proprio"] = proprio
    if "sim_state" in obs:
        sim_state = obs["sim_state"]
        sim_state = normalize_sim_state(sim_state, norm_stats)
        obs["sim_state"] = sim_state
    if "image_primary" in obs:
        image_primary = obs["image_primary"]
        image_primary = normalize_image(image_primary, norm_stats)
        obs["image_primary"] = image_primary
    if "image_wrist" in obs:
        image_wrist = obs["image_wrist"]
        image_wrist = normalize_image(image_wrist, norm_stats)
        obs["image_wrist"] = image_wrist

    return obs


# proprio normalization: use mean/std for all dimensions
def normalize_proprio(proprio, norm_stats):
    assert proprio.dtype == np.float32
    proprio = np.array(proprio)
    proprio_norm_stats = norm_stats["observations/proprio"]
    mean, std = proprio_norm_stats["mean"], proprio_norm_stats["std"]
    mean, std = np.array(mean), np.array(std)

    ## apply to all dims
    proprio = (proprio - mean) / (std + 1e-8)
    return proprio

# sim state normalization: use mean/std for all dimensions
def normalize_sim_state(sim_state, norm_stats):
    assert sim_state.dtype == np.float32
    sim_state = np.array(sim_state)
    sim_state_norm_stats = norm_stats["observations/sim_state"]
    mean, std = sim_state_norm_stats["mean"], sim_state_norm_stats["std"]
    mean, std = np.array(mean), np.array(std)
    sim_state = (sim_state - mean) / (std + 1e-8)
    return sim_state


## ALL action normalization must only normalize the first 6 dimensions!

# normalize actions into (-1, 1), based on min/max from norm stats
def normalize_action_min_max(action, norm_stats):
    assert action.dtype == np.float32
    action = np.array(action)
    action_norm_stats = norm_stats["actions"]
    action_min, action_max = action_norm_stats["min"], action_norm_stats["max"]
    action_min, action_max = np.array(action_min), np.array(action_max)
    action[:, :6] = 2 * (action[:, :6] - action_min[:6]) / (action_max[:6] - action_min[:6] + 1e-8) - 1.0

    # scale everthing closer to 0 a bit so we don't get Nan log_probs using tanh
    action = action * (1 - 1e-4)
    return action


# binarization is handled by the dataloader and the eval script
# def normalize_action_mean_std(action, norm_stats):
#     import pdb; pdb.set_trace()
#     assert action.dtype == np.float32
#     # same idea, except only normalize first 6 dims
#     action = np.array(action)
#     action_norm_stats = norm_stats["actions"]
#     mean, std = action_norm_stats["mean"], action_norm_stats["std"]
#     mean, std = np.array(mean), np.array(std)
#     action[:, :6] = (action[:, :6] - mean[:6]) / (std[:6] + 1e-8)
#     # b/c we're using tanh, add/subtract a small epsilon to the last gripper dim based on the sign, so it is not exactly -1 or +1
#     action[:, 6] = action[:, 6] * (1 - 1e-5)
#     return action


def unnormalize_action_min_max(action, dataset_name: str):
    assert action.dtype == np.float32
    norm_stats = ALL_NORM_STATS[dataset_name]
    action = np.array(action)
    action_norm_stats = norm_stats["actions"]
    action_min, action_max = action_norm_stats["min"], action_norm_stats["max"]
    action_min, action_max = np.array(action_min), np.array(action_max)
    action[:, :6] = (action[:, :6] + 1.0) / 2.0 * (action_max[:6] - action_min[:6]) + action_min[:6]
    return action


def unnormalize_action_mean_std(action, dataset_name: str):
    assert action.dtype == np.float32
    action = np.array(action)
    action_norm_stats = ALL_NORM_STATS[dataset_name]["actions"]
    mean, std = action_norm_stats["mean"], action_norm_stats["std"]
    mean, std = np.array(mean), np.array(std)
    
    ## only unnormalize first 6 dims: do nothing w/ last 1 as it will be binarized by eval script
    action[:, :6] = (action[:, :6] * (std[:6] + 1e-8)) + mean[:6]
    return action



# Create lookup table once (at startup)
# NORMALIZE_LUT = (np.arange(256, dtype=np.float32) / 255.0 * 2.0 - 1.0)
# # fast way to normalize images into [-1, 1] by just indexing into a lookup table
# def normalize_image(image):
#     return NORMALIZE_LUT[image]


# puts images in range [-1, 1]
# def normalize_image(image):
#     assert image.dtype == np.uint8
#     image = image.astype(np.float32) / 255.0 * 2.0 - 1.0
#     return image


# def unnormalize_image(image):
#     assert image.dtype == np.float32
#     image = (image + 1.0) / 2.0 * 255.0
#     image = image.astype(np.uint8)
#     return image