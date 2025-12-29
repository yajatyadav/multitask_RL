from os.path import expanduser
import os
import pathlib
import re


import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import imageio
import h5py
import glob
from tqdm import tqdm
import math
import jax
import jax.numpy as jnp

from utils.datasets import Dataset

import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'libero'))
from libero.libero.envs.env_wrapper import ControlEnv, OffScreenRenderEnv
from libero.libero.utils import get_libero_path
from libero.libero.envs import SubprocVectorEnv
from libero.libero import benchmark


LIBERO_WARMUP_STEPS = 15
NUM_UNIQUE_LIBERO_TASKS = 112


# setup one-hot encoder for libero tasks

def build_libero_one_hot_table():
    libero_benchmark_dict = benchmark.get_benchmark_dict()
    all_libero_languages = set()
    benchmarks = sorted(["libero_spatial", "libero_object", "libero_goal", "libero_90", "libero_10"]) # sorted() to retain same order
    for benchmark_name in benchmarks:
        suite = libero_benchmark_dict[benchmark_name]()
        num_tasks = suite.get_num_tasks()
        for i in range(num_tasks):
            task = suite.get_task(i)
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
    return all_libero_languages


LIBERO_ONE_HOT_TABLE = build_libero_one_hot_table()

class OneHotEmbedding_Libero:
    @staticmethod
    def encode(string):
        """Encode a single string into a one-hot vector.
        
        Args:
            string: A single string (not a list)
            
        Returns:
            One-hot vector of shape (NUM_UNIQUE_LIBERO_TASKS,)
        """
        def standardize_string(s):
            if isinstance(s, bytes):
                s = s.decode("utf-8")
            s = s.lower().replace("_", " ")
            return s
        
        # Ensure input is a single string
        if not isinstance(string, (str, bytes)):
            raise TypeError(f"Expected a single string, got {type(string)}")
        
        # Get the index and return the corresponding row from identity matrix
        index = LIBERO_ONE_HOT_TABLE[standardize_string(string)]
        return np.eye(NUM_UNIQUE_LIBERO_TASKS)[index]


def is_libero_env(env_name):
    """determine if an env is libero"""
    return "libero" in env_name


def _get_max_episode_length(env_name):
    if env_name.startswith("libero_spatial"):
        return 220
    elif env_name.startswith("libero_object"):
        return 280
    elif env_name.startswith("libero_goal"):
        return 300
    elif env_name.startswith("libero_90"):
        return 400
    elif env_name.startswith("libero_10"):
        return 520
    else:
        raise ValueError(f"Unsupported environment: {env_name}")

def _get_normalization_path(env_name):
    print(f"TODO(YY): Normalization path not implemented for {env_name}")
    # raise NotImplementedError("TODO(YY): Normalization path not implemented")

def extract_all_libero_env_names(env_name, task_name):
    all_tasks = task_name.split('|')
    print(f"all_tasks: {all_tasks}")
    suites = []
    scenes = []
    if env_name.startswith("all_libero"):
        suites = ["libero_spatial", "libero_object", "libero_goal", "libero_90"]
        scenes = ['*', '*', '*', '*', '*']
    else:
        # in case of single suite
        suite = env_name.split("-")[0]
        scene = ''
        if suite == "libero_90":
            scene = env_name.split("-")[1]
        suites = [suite]
        scenes = [scene]

    print(f"suites: {suites}, scenes: {scenes}")           


    all_names = {}
    for i, suite in enumerate(suites):
        all_names[suite] = []
        task_suite = benchmark.get_benchmark_dict()[suite]()
        num_tasks = task_suite.n_tasks
        for task_id in range(num_tasks):
            task = task_suite.get_task(task_id)
            language = task.language.replace(" ", "_").lower()
            if scenes[i].lower() == '*':
                all_names[suite].append(f"{suite}-{task.name}")
            
            elif task.name.lower().startswith(scenes[i].lower()):
                if task_name != '':
                    if language in all_tasks:
                        all_names[suite].append(f"{suite}-{task.name}")
                else:
                    all_names[suite].append(f"{suite}-{task.name}")
        all_names[suite] = sorted(all_names[suite])
    return all_names


def make_env(env_name, task_name,num_parallel_envs, use_hardcoded_eval_envs=False, render_resolution=128, keys_to_load=[], seed=0):
    """
    NOTE: is now returning a LIST of environments, thus the main script needs to sequentiall loop and call evaluate() on each..
    """
    normalization_path = _get_normalization_path(env_name)
    # max_episode_length = _get_max_episode_length(env_name)

    # eval-time keys look different from those in the training dataset, they are:
    # raw_obs: odict_keys(['robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'akita_black_bowl_1_pos', 'akita_black_bowl_1_quat', 'akita_black_bowl_1_to_robot0_eef_pos', 'akita_black_bowl_1_to_robot0_eef_quat', 'akita_black_bowl_2_pos', 'akita_black_bowl_2_quat', 'akita_black_bowl_2_to_robot0_eef_pos', 'akita_black_bowl_2_to_robot0_eef_quat', 'akita_black_bowl_3_pos', 'akita_black_bowl_3_quat', 'akita_black_bowl_3_to_robot0_eef_pos', 'akita_black_bowl_3_to_robot0_eef_quat', 'plate_1_pos', 'plate_1_quat', 'plate_1_to_robot0_eef_pos', 'plate_1_to_robot0_eef_quat', 'robot0_proprio-state', 'object-state'])😛 keys to load: ['agentview_rgb', 'eye_in_hand_rgb']


    # env keys should remove the 'obs/' prefix
    old_to_new_key = {
        'agentview_rgb': 'agentview_image',
        'eye_in_hand_rgb': 'robot0_eye_in_hand_image',
        'proprio': 'proprio',
        # 'obs/ee_pos': 'robot0_eef_pos',
        # 'obs/ee_ori': 'robot0_eef_quat',
        # 'obs/gripper_states': 'robot0_gripper_qpos',
        'states': 'states',
        'language': 'language',
    }
    keys_to_output_map = {v:k for k, v in old_to_new_key.items()}
    keys_to_load = {old_to_new_key[k] for k in keys_to_load}
    # if any key is a image key, we can no longer use NoRenderEnv, and must use OffScreenRenderEnv
    eval_need_camera_obs = any('image' in k for k in keys_to_load)
    print("evaluation environment will return keys: ", keys_to_load)

    all_env_names = extract_all_libero_env_names(env_name, task_name)
    print(f"All possible envs, sorted alphabetically: there are {len(all_env_names)} total envs")
    
    HARDCODED_EVAL_ENVS = {
        "libero_spatial": [
            "libero_spatial-pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
            
        ],
        "libero_object": [
            'libero_object-pick_up_the_alphabet_soup_and_place_it_in_the_basket',
        ],
        "libero_goal": [
            "libero_goal-open_the_middle_drawer_of_the_cabinet",
        ],
        "libero_90": [
            "libero_90-KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet",
            "libero_90-LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
            "libero_90-STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
        ],
    }
    ## take 10 interspersed envs: 4 from libero_90, 2 from libero_spatial, 2 from libero_object, 2 from libero_goal
    # envs_to_eval = []
    # for suite, suite_eval_envs in all_env_names.items():
        # indices = np.linspace(0, len(suite_eval_envs)-1, 4 if suite == "libero_90" else 2, dtype=int)
        # envs_to_eval.extend([suite_eval_envs[i] for i in indices])

    
    envs_to_eval = []
    if use_hardcoded_eval_envs:
        for suite, suite_eval_envs in HARDCODED_EVAL_ENVS.items():
            envs_to_eval.extend([env for env in suite_eval_envs])
    else:
        for suite, suite_eval_envs in all_env_names.items():
            envs_to_eval.extend([env for env in suite_eval_envs])
    
    print(f" {len(envs_to_eval)=} Environments to evaluate: {envs_to_eval}")

    env_list, names_to_return = [], []
    for i, env_name in enumerate(envs_to_eval):
        env =LiberoTopLevelEnvWrapper(
            env_name=env_name,
            seed=seed + (i * 50_000),
            eval_need_camera_obs=eval_need_camera_obs,
            num_parallel_envs=num_parallel_envs,
            render_resolution=render_resolution,
            obs_keys=keys_to_load,
            keys_to_output_map=keys_to_output_map,
            normalization_path=normalization_path,
        )
        env_list.append(env)
        names_to_return.append(env_name)
    return env_list, names_to_return

def _check_dataset_exists(env_name):
    # enforce that the dataset exists
    if env_name.lower().startswith("libero_90") or env_name.lower().startswith("libero_10"):
        suite, scene_task = env_name.split("-")
    else:
        suite, scene_task = env_name.split("-")
    libero_dataset_dir = os.path.join(os.path.dirname(os.getcwd()), 'datasets/raw_libero')
    dataset_path = os.path.join(
        libero_dataset_dir,
        suite.upper(),
        scene_task + "_demo.hdf5",
    ) 
    # print(f"😈😈😈 dataset path: {dataset_path}")
    assert os.path.exists(dataset_path)
    
    return dataset_path


def stack_dict_list(dict_list):
    """Stack a list of dictionaries into a dictionary of stacked arrays."""
    if not dict_list:
        return {}
    keys = dict_list[0].keys()
    return {k: np.concatenate([d[k] for d in dict_list], axis=0) for k in keys}

def get_dataset(env, env_name, task_name, augmentation_type, augmentation_reward, keys_to_load, demo_nums_to_use_per_task=None, augmentation_dict=None):
    # data holders
    observations = []
    actions = []
    next_observations = []
    terminals = []
    rewards = []
    masks = []
    print(f"🤪🤪🤪 augmentation_reward: {augmentation_reward}")

    def process_task(rm_dataset, zero_out_rewards, target_task_name):
        # print(f"processing dataset for task {this_task_name}")
        demos = list(rm_dataset["data"].keys())
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds] # sort demos!
        if demo_nums_to_use_per_task is not None:
            print(f"😎😎😎 ONLY using {demo_nums_to_use_per_task} demos for target_task: {target_task_name}")
            demos = [demos[i] for i in demo_nums_to_use_per_task]

        task_embedding = OneHotEmbedding_Libero.encode(target_task_name)

        this_task_num_timesteps = 0
        for ep in demos:
            a = np.array(rm_dataset["data/{}/actions".format(ep)])
            this_task_num_timesteps += a.shape[0]
            obs, next_obs = {}, {}
            for k in keys_to_load:
                if k == 'states':
                    obs[k] = np.array(rm_dataset[f"data/{ep}/{k}"])[:, 1:] # drop the first entry, which is the timestep
                elif k == 'language':
                    obs[k] = np.repeat(task_embedding[None, :], a.shape[0], axis=0)
                elif k == 'proprio':
                    obs[k] = np.concatenate(
                        [
                            np.array(rm_dataset[f"data/{ep}/obs/ee_pos"]),
                            np.array(rm_dataset[f"data/{ep}/obs/ee_ori"]),
                            np.array(rm_dataset[f"data/{ep}/obs/gripper_states"]),
                        ],
                        axis=-1,
                    )
                else:
                    obs[k] = np.array(rm_dataset[f"data/{ep}/obs/{k}"])
            
            for k in keys_to_load:
                if k == 'states':
                    obs_array = np.array(rm_dataset[f"data/{ep}/{k}"])[:, 1:]
                elif k == 'language':
                    obs_array = np.repeat(task_embedding[None, :], a.shape[0], axis=0)
                elif k == 'proprio':
                    obs_array = np.concatenate(
                        [
                            np.array(rm_dataset[f"data/{ep}/obs/ee_pos"]),
                            np.array(rm_dataset[f"data/{ep}/obs/ee_ori"]),
                            np.array(rm_dataset[f"data/{ep}/obs/gripper_states"]),
                        ],
                        axis=-1,
                    )
                else:
                    obs_array = np.array(rm_dataset[f"data/{ep}/obs/{k}"])
                
                next_obs[k] = np.concatenate([obs_array[1:], obs_array[-1:]], axis=0) # make next obs by shifting obs array by 1, and then repeating the last element of obs array so obs and next_obs have same size
            
            # obs = np.concatenate(obs, axis=-1)
            # next_obs = np.concatenate(next_obs, axis=-1)
            dones = np.array(rm_dataset["data/{}/dones".format(ep)])

            # read in rewards, and set all to -1 if not positive task
            r = np.array(rm_dataset["data/{}/rewards".format(ep)], dtype=np.float32)
            # the raw data holds 0/1 rewards. if this is a positive task, we will keep the rewards as is, otherwise we will set ALL to 0
            if zero_out_rewards:
                r = np.full_like(r, augmentation_reward)

            # append to data holders
            observations.append(obs)
            actions.append(a.astype(np.float32))
            rewards.append(r.astype(np.float32))
            terminals.append(dones.astype(np.float32))
            masks.append(1.0 - dones.astype(np.float32))
            next_observations.append(next_obs)
        return this_task_num_timesteps

    # TODO(YY): still some issues w/ kitchen_scene1 and kitchen_scene10 in some regex matching locations...
    # such as distinct_scenes containing stuff it shouldn't here...
    # crawl through env_name directory, and add each task's demos
    libero_dataset_dir = os.path.join(os.path.dirname(os.getcwd()), 'datasets/raw_libero')
    all_libero_env_names = extract_all_libero_env_names(env_name, task_name)
    print(f"🤪🤪🤪 all_libero_env_names: {all_libero_env_names}")
    distinct_scenes = {}
    for suite in all_libero_env_names.keys():
        if suite != "libero_90":
            distinct_scenes[suite] = ["*"]
            continue        
        distinct_scenes[suite] = []
        for env_name in all_libero_env_names[suite]:
            prefix = env_name.split("-")[1]
            get_scene_prefix = lambda s: (re.match(r'^[^\d]*\d+', s) or re.match(r'.*', s)).group()
            scene = get_scene_prefix(prefix)
            if scene not in distinct_scenes[suite]:
                distinct_scenes[suite].append(scene)    
    print(f"🤪🤪🤪 distinct_scenes: {distinct_scenes}")

    # using switch case + functions here to handle the different augmentation types
    match augmentation_type:
        case 'none':
            num_timesteps = none_augmentation(distinct_scenes, task_name, process_task, libero_dataset_dir)
        case 'custom_task':
            num_timesteps = custom_task_augmentation(distinct_scenes, task_name, process_task, libero_dataset_dir, augmentation_dict)
        case 'task':
            # assert task_name != '', "Task name must be provided for augmentation type 'task'"
            # num_timesteps = task_augmentation(distinct_scenes, process_task, libero_dataset_dir, task_name)
            raise NotImplementedError("Augmentation type 'task' is not supported yet.")
        case 'first':
            raise NotImplementedError("Augmentation type 'first' is not supported yet.")
        case 'exhaustive':
            num_timesteps = exhaustive_augmentation(distinct_scenes, process_task, libero_dataset_dir)
        case _:
            raise ValueError(f"Invalid augmentation type: {augmentation_type}")
    # j = 0
    # for suite in distinct_scenes.keys():
    #     for scene in distinct_scenes[suite]:
    #         suite = suite.upper()
    #         pattern = os.path.join(libero_dataset_dir, suite, f"{scene.upper()}_*.hdf5")

    #         # print how many files for this scene
    #         print(f"😈😈😈 {suite=} {scene=} , there are {len(sorted(glob.glob(pattern)))} files for this suite + scene")

    #         ##TODO(YY): for now, if task name is blank, we will use the FIRST task for this scene as the positive example
    #         if task_name == '':
    #             first_task_fp = sorted(glob.glob(pattern))[0]
    #             first_task_name = os.path.basename(first_task_fp).split(".")[0][:-5] # remove .hdf5 and _demo to get scene+task name separated by underscores
    #             if "SCENE" in first_task_name:
    #                 first_task_name = extract_libero_task_name_only(first_task_name)
    #             target_task_name = first_task_name
    #         else:
    #             target_task_name = task_name

    #         print(f"😈😈😈 {target_task_name=}")

    #         for filepath in sorted(glob.glob(pattern)):
    #             this_task_name = os.path.basename(filepath).split(".")[0][:-5] # remove .hdf5 and _demo to get scene+task name separated by underscores
    #             _check_dataset_exists(f"{suite}-{this_task_name}")
                
    #             if "SCENE" in this_task_name:
    #                 this_task_name = extract_libero_task_name_only(this_task_name)          
    #             is_positive_task = this_task_name == target_task_name # we will flip reward sign if using demos from same scene but for different task

    #             process_this_dataset = is_positive_task or augment_negative_demos
    #             if not process_this_dataset:
    #                 continue
                
    #             flip_rewards = not is_positive_task

    #             # now actually load the dataset and relabel language + flip rewards if needed
    #             rm_dataset = h5py.File(filepath, "r")
    #             this_task_num_timesteps = process_task(rm_dataset, flip_rewards, target_task_name)
    #             num_timesteps += this_task_num_timesteps
    #             print(f"🥳🥳🥳 {j=} Dataset {this_task_name} has {this_task_num_timesteps}, and {flip_rewards=}, relabeled to {target_task_name=}")
    #             j += 1
    # print(f"the total size of the dataset is {num_timesteps}")

    # once all done, create the dataset object
    return Dataset.create(
        observations=stack_dict_list(observations),
        next_observations=stack_dict_list(next_observations),
        actions=np.concatenate(actions, axis=0),
        rewards=np.concatenate(rewards, axis=0),
        terminals=np.concatenate(terminals, axis=0),
        masks=np.concatenate(masks, axis=0),
    )


# def task_augmentation(distinct_scenes, process_task_fn, libero_dataset_dir, target_task_name):
#     j = 0
#     num_timesteps = 0
#     for suite in distinct_scenes.keys():
#         scenes = distinct_scenes[suite]
#         for scene in scenes:
#             suite = suite.upper()
#             scene = scene.upper()
#             pattern = os.path.join(libero_dataset_dir, suite, f'{scene}_*.hdf5')
#             for filepath in sorted(glob.glob(pattern)):
#                 task_name = os.path.basename(filepath).split(".")[0][:-5]
#                 if "SCENE" in task_name:
#                     task_name = extract_libero_task_name_only(task_name)
#                 if task_name == target_task_name:
#                     continue
                
#                 _check_dataset_exists(f"{suite}-{task_name}")
#                 zero_out_rewards = False
#                 rm_dataset = h5py.File(filepath, "r")
#                 this_task_num_timesteps = process_task_fn(rm_dataset, zero_out_rewards, target_task_name)
#                 num_timesteps += this_task_num_timesteps
#                 print(f"🥳🥳🥳 {j=} Dataset {task_name} has {this_task_num_timesteps}, and {zero_out_rewards=}, relabeled to {target_task_name=}")
#     print(f"the total size of the dataset is {num_timesteps}")
#     return num_timesteps


def none_augmentation(distinct_scenes, task_name, process_task_fn, libero_dataset_dir):
    all_tasks = task_name.split('|')
    # if task_name != '':
    #     assert len(distinct_scenes) == 1, f'distinct_scenes has {len(distinct_scenes)} suites, but expected {len(all_tasks)}'
    #     k = list(distinct_scenes.keys())[0]
    #     assert len(distinct_scenes[k]) == 1, f'key {k} has {len(distinct_scenes[k])} scenes, but expected 1'
    
    j = 0
    num_timesteps = 0
    for suite in distinct_scenes.keys():
        scenes = distinct_scenes[suite]
        for scene in scenes:
            suite = suite.upper()
            scene = scene.upper()
            pattern = os.path.join(libero_dataset_dir, suite, f'{scene}_*.hdf5')
            
            for filepath in sorted(glob.glob(pattern)):
                target_task_name = os.path.basename(filepath).split(".")[0][:-5]
                _check_dataset_exists(f"{suite}-{target_task_name}")
                if "SCENE" in target_task_name:
                    target_task_name = extract_libero_task_name_only(target_task_name)

                if task_name != '' and target_task_name not in all_tasks:
                    continue
                
                print(f"😈😈😈 {target_task_name=}")
                zero_out_rewards = False
                rm_dataset = h5py.File(filepath, "r")
                this_task_num_timesteps = process_task_fn(rm_dataset, zero_out_rewards, target_task_name)
                num_timesteps += this_task_num_timesteps
                print(f"🥳🥳🥳 {j=} Dataset {target_task_name} has {this_task_num_timesteps}, and {zero_out_rewards=}, relabeled to {target_task_name=}")
                j += 1
    print(f"the total size of the dataset is {num_timesteps}")
    return num_timesteps

def custom_task_augmentation(distinct_scenes, task_name, process_task_fn, libero_dataset_dir, augmentation_dict):
    all_tasks = task_name.split('|')
    assert len(augmentation_dict) == len(all_tasks), f'augmentation_dict has {len(augmentation_dict)} tasks, but expected {len(all_tasks)}'
    
    j = 0
    num_timesteps = 0
    for suite in distinct_scenes.keys():
        scenes = distinct_scenes[suite]
        for scene in scenes:
            suite = suite.upper()
            scene = scene.upper()

            for task, augment_tasks in augmentation_dict.items():
                
                # first add task as a positive example
                pattern = os.path.join(libero_dataset_dir, suite,f'{scene}_{task}*.hdf5')
                print(f"😈😈😈 {pattern=}", 'there are', len(sorted(glob.glob(pattern))), 'files for this suite + scene + task')
                positive_filepath = sorted(glob.glob(pattern))[0]
                _check_dataset_exists(f"{suite.upper()}-{scene}_{task}")
                zero_out_rewards = False
                rm_dataset = h5py.File(positive_filepath, "r")
                this_task_num_timesteps = process_task_fn(rm_dataset, zero_out_rewards, task)
                num_timesteps += this_task_num_timesteps
                print(f"🥳🥳🥳 {j=} Dataset {task} has {this_task_num_timesteps}, and {zero_out_rewards=}, relabeled to {task=}")
                j += 1

                # now, add each task in augment_tasks as a negative example
                for augment_task in augment_tasks:
                    pattern = os.path.join(libero_dataset_dir, suite, f'{scene}_{augment_task}*.hdf5')
                    print(f"😈😈😈 {pattern=}", 'there are', len(sorted(glob.glob(pattern))), 'files for this suite + scene + task')
                    negative_filepath = sorted(glob.glob(pattern))[0]
                    _check_dataset_exists(f"{suite.upper()}-{scene}_{augment_task}")
                    zero_out_rewards = True
                    rm_dataset = h5py.File(negative_filepath, "r")
                    this_task_num_timesteps = process_task_fn(rm_dataset, zero_out_rewards, task)
                    num_timesteps += this_task_num_timesteps
                    print(f"🥳🥳🥳 {j=} Dataset {augment_task} has {this_task_num_timesteps}, and {zero_out_rewards=}, relabeled to {task=}")
                    j += 1
    
    print(f"the total size of the dataset is {num_timesteps}")
    return num_timesteps
                
 
def exhaustive_augmentation(distinct_scenes, process_task_fn, libero_dataset_dir):
    j = 0
    num_timesteps = 0
    for suite in distinct_scenes.keys():
        scenes = distinct_scenes[suite]
        for scene in scenes:
            suite = suite.upper()
            scene = scene.upper()
            pattern = os.path.join(libero_dataset_dir, suite, f'{scene}_*.hdf5')
            # print how many files for this scene
            print(f"😈😈😈 {pattern=} {suite=} {scene=} , there are {len(sorted(glob.glob(pattern)))} files for this suite + scene")

            for filepath in sorted(glob.glob(pattern)):
                target_task_name = os.path.basename(filepath).split(".")[0][:-5]
                if "SCENE" in target_task_name:
                    target_task_name = extract_libero_task_name_only(target_task_name)
                print(f"😈😈😈 {target_task_name=}")
                
                for fp in sorted(glob.glob(pattern)):
                    this_task_name = os.path.basename(fp).split(".")[0][:-5]
                    _check_dataset_exists(f"{suite.upper()}-{this_task_name}")
                    if "SCENE" in this_task_name:
                        this_task_name = extract_libero_task_name_only(this_task_name)                    
                                     
                    zero_out_rewards = this_task_name != target_task_name
                    rm_dataset = h5py.File(fp, "r")
                    this_task_num_timesteps = process_task_fn(rm_dataset, zero_out_rewards, target_task_name)
                    num_timesteps += this_task_num_timesteps
                    print(f"🥳🥳🥳 {j=} Dataset {this_task_name} has {this_task_num_timesteps}, and {zero_out_rewards=}, relabeled to {target_task_name=}")
                    j += 1
    print(f"the total size of the dataset is {num_timesteps}")
    return num_timesteps


def extract_libero_task_name_only(s):

    match = re.search(r'\d+', s)  # \d+ matches one or more digits
    if match:
        end_index = match.end()  # Position right after the number
        return s[end_index + 1:]  # Start 1 position after
    return ""  # No number found

class NoRenderEnv(ControlEnv):
    """
    LIBERO environment without any rendering capabilities.
    This is the most compatible option for multiprocessing + testing state-based evals massively in parallel!.
    """
    
    def __init__(self, **kwargs):
        # Completely disable rendering
        kwargs["has_renderer"] = False
        kwargs["has_offscreen_renderer"] = False
        kwargs["use_camera_obs"] = False
        kwargs["camera_depths"] = False
        kwargs["camera_segmentations"] = None
        super().__init__(**kwargs)


# ============================================================================
# LiberoEnvWrapper - Provides consistent gym API for LIBERO environments
# ============================================================================

class LiberoEnvWrapper(gym.Env):
    """
    Environment wrapper for LIBERO OffScreenRenderEnv with a consistent gym API.
    Provides normalization, observation handling, and video recording capabilities.
    """
    def __init__(
        self,
        env,
        normalization_path=None,
        task_embedding=None,
        obs_keys=[],
        keys_to_output_map={},
        clamp_obs=False,
        init_state=None,
        can_render=False,
        render_hw=(256, 256),
        render_camera_name="agentview",
        max_episode_length=500,
    ):
        self.env = env
        self.can_render = can_render
        self.task_embedding = task_embedding
        self.obs_keys = obs_keys
        self.keys_to_output_map = keys_to_output_map
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.video_writer = None
        self.clamp_obs = clamp_obs
        self.max_episode_length = max_episode_length
        self.env_step = 0
        self.n_episodes = 0
        self.t = 0
        self.episode_return = 0
        self.episode_length = 0

        # Set up normalization
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.obs_min = normalization["obs_min"]
            self.obs_max = normalization["obs_max"]
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]

        # Setup spaces - use [-1, 1]
        # Get action dimension from environment
        action_dim = self.env.robots[0].action_dim
        low = np.full(action_dim, fill_value=-1.)
        high = np.full(action_dim, fill_value=1.)
        self.action_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )
        
        # Get observation space
        obs_example = self.get_observation()
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        # self.observation_space = Box(
        #     low=low,
        #     high=high,
        #     shape=low.shape,
        #     dtype=low.dtype,
        # )

    def normalize_obs(self, obs):
        obs = 2 * (
            (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5
        )  # -> [-1, 1]
        if self.clamp_obs:
            obs = np.clip(obs, -1, 1)
        return obs

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    
    def _quat2axisangle(self, quat):
        """
        Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
        """
        # clip quaternion
        if quat[3] > 1.0:
            quat[3] = 1.0
        elif quat[3] < -1.0:
            quat[3] = -1.0

        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            # This is (close to) a zero degree rotation, immediately return
            return np.zeros(3)
        return (quat[:3] * 2.0 * math.acos(quat[3])) / den
    

    def get_observation(self):
        """Extract and concatenate relevant observation keys"""
        raw_obs = self.env.env._get_observations()
        # print(f"raw_obs: {raw_obs.keys()}😛 keys to load: {self.obs_keys}")
        obs_to_return = {}
        for key in self.obs_keys:
            if key == 'states':
                sim_state = self.get_sim_state()
                obs_to_return[key] = sim_state[1:] # drop the first entry, which is the timestep
            elif key == 'language':
                obs_to_return[key] = self.task_embedding
            elif key == 'proprio':
                obs_to_return[key] = np.concatenate(
                    [
                        raw_obs['robot0_eef_pos'],
                        self._quat2axisangle(raw_obs['robot0_eef_quat']),
                        raw_obs['robot0_gripper_qpos'],
                    ],
                    axis=-1,
                )
            else:
                obs_to_return[key] = raw_obs[key]

        if self.normalize:
           obs_to_return = {k: self.normalize_obs(v) for k, v in obs_to_return.items()}

        # remap eval key names to train dataset key names
        obs_to_return = {self.keys_to_output_map[k]: v for k, v in obs_to_return.items()}
        # obs_to_return = jax.tree_util.tree_map(jnp.asarray, obs_to_return)
        return obs_to_return

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
            self.env.seed(seed)
        else:
            np.random.seed()

    def reset(self, options={}, **kwargs):
        """Reset the environment"""
        self.t = 0
        self.episode_return = 0
        self.episode_length = 0
        self.n_episodes += 1
        
        # Close video if exists
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        # Start video if specified
        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        # Handle seed from options
        new_seed = options.get("seed", None)
        if new_seed is not None:
            self.seed(seed=new_seed)

        # Reset environment
        if self.init_state is not None:
            # Reset to specific state
            obs = self.env.set_init_state(self.init_state)
        else:
            # Random reset
            obs = self.env.reset()

        ## take env.step() with dummy action for 15 steps to wait for objects to settle into place
        for _ in range(LIBERO_WARMUP_STEPS):
            obs, reward, done, info = self.step([0.0] * 6 + [-1.0])

        # Get processed observation and return only the observation (not a tuple)
        # This is for compatibility with older gym environments that SubprocVectorEnv expects
        return self.get_observation() ## used to be return obs, {}; but removed the empty dict for SubProcVectorEnv compatibility - add back this empty dict in the caller functions

    def step(self, action):
        """Step the environment"""
        if self.normalize:
            action = self.unnormalize_action(action)
        
        # Step the environment
        raw_obs, reward, done, info = self.env.step(action)        
        # Get processed observation
        obs = self.get_observation()

        # Render if specified
        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            self.video_writer.append_data(video_img)

        # Update counters
        self.t += 1
        self.env_step += 1
        self.episode_return += reward
        self.episode_length += 1

        # Check for success
        if done:
            info["success"] = 1
        else:
            info["success"] = 0

        # Check for truncation
        truncated = False
        if self.t >= self.max_episode_length:
            truncated = True
            done = True  # Mark as done for older gym compatibility

        ##  Return in old gym format (obs, reward, done, info) for SubprocVectorEnv compatibility, instead of (obs, reward, done, truncated, info)
        return obs, reward, done, info
    

    def render(self,):
        """Render the environment (disabled for NoRenderEnv compatibility)"""
        # Since we're using NoRenderEnv, rendering is disabled
        if not self.can_render:
            raise ValueError("Rendering is disabled for this environment")
        else:
            camera_name = self.render_camera_name
            h, w = self.render_hw
            return self.env.sim.render(
                width=w,
                height=h,
                camera_name=camera_name,
            )
    
    def get_sim_state(self):
        """Get current simulation state"""
        return self.env.get_sim_state()
    
    def set_init_state(self, init_state):
        """Set initial state and return observation"""
        obs = self.env.set_init_state(init_state)
        return self.get_observation()
    
    def get_episode_info(self):
        """Get episode statistics"""
        return {"return": self.episode_return, "length": self.episode_length}
    
    def get_info(self):
        """Get environment statistics"""
        return {"env_step": self.env_step, "n_episodes": self.n_episodes}
    
    def close(self):
        """Close the environment and cleanup"""
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        self.env.close()


# ============================================================================
# Factory Function - Create wrapped LIBERO environments
# ============================================================================

def get_libero_task_init_states(
    env_name
):
    """Get the initial states for a given libero task"""
    if env_name.startswith("libero_90") or env_name.startswith("libero_10"):
        suite_str, scene_str, task_str = env_name.split("-", 2)
    else:
        suite_str, task_str = env_name.split("-", 1)
        scene_str = ''
    task_suite = benchmark.get_benchmark_dict()[suite_str]()
    num_tasks_in_suite = task_suite.n_tasks
    desired_task_name = f"{scene_str.upper()}_{task_str}" if scene_str else task_str.upper()
    initial_states = None
    for task_id in range(num_tasks_in_suite):
        candidate_task = task_suite.get_task(task_id)
        if candidate_task.name == desired_task_name:
            initial_states = task_suite.get_task_init_states(task_id)
            break
    return initial_states


def get_libero_task_from_env(env_name):
    suite_str, scene_task_str = env_name.split("-")

    # Get task suite
    task_suite = benchmark.get_benchmark_dict()[suite_str]()
    num_tasks_in_suite = task_suite.n_tasks

    task = None
    for task_id in range(num_tasks_in_suite):
        candidate_task = task_suite.get_task(task_id)
        if candidate_task.name == scene_task_str:
            task = candidate_task
            break
    
    if task is None:
        raise ValueError(f"Task {scene_task_str} not found in suite {suite_str}")

    return task




def make_libero_env(
    env_name,
    initial_state,
    render=False,
    render_resolution=128,
    obs_keys=[],
    keys_to_output_map={},
    normalization_path=None,
    max_episode_length=500,
    seed=0,
):
    """
    Factory function to create a wrapped LIBERO environment without rendering.
    
    Args:
        env_name: Environment name in format "suite-scene-task" 
                 (e.g., "libero_90-study_scene1-pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy")
        render_resolution: Resolution for rendering (kept for API compatibility but not used)
        obs_keys: List of observation keys to extract
        normalization_path: Path to normalization statistics (optional)
        max_episode_length: Maximum episode length
        seed: Random seed
    
    Returns:
        LiberoEnvWrapper instance
    """    
    task = get_libero_task_from_env(env_name)
    # Build environment WITHOUT rendering for subprocess compatibility
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": render_resolution,
        "camera_widths": render_resolution,
    }
    
    # Use NoRenderEnv instead of OffScreenRenderEnv to avoid rendering issues
    if render:
        base_env = OffScreenRenderEnv(**env_args)
    else:
        base_env = NoRenderEnv(**env_args)
    base_env.seed(seed)
    
    # Wrap environment
    task_language_str = task.language.replace(" ", "_").lower()
    task_embedding = OneHotEmbedding_Libero.encode(task_language_str)
    wrapped_env = LiberoEnvWrapper(
        env=base_env,
        normalization_path=normalization_path,
        task_embedding=task_embedding,
        obs_keys=obs_keys,
        keys_to_output_map=keys_to_output_map,
        max_episode_length=max_episode_length,
        can_render=render,
        render_hw=(render_resolution, render_resolution),
        init_state=initial_state,
    )
    
    return wrapped_env, task_embedding


class LiberoTopLevelEnvWrapper(gym.Env):
    """ To maintain API consistency with make_env_and_datasets, this is just a thin wrapper that holds a sbuprocvectorenv(liberoenvwrapper(norenderenv)) + liberoenvwrapper(offscreenrender). First used for fast evals and second for video logging """
    def __init__(
        self,
        env_name, # a string!!
        seed,
        eval_need_camera_obs,
        num_parallel_envs,
        normalization_path=None,
        obs_keys=[],
        keys_to_output_map={},
        render_resolution=128,
        max_episode_length=-1,
    ):        
        # TODO(YY): uncommenting this line causes that weird red ball + green line coming from robot arm issue (might just be an artifact...)
        # all_initial_states = get_libero_task_init_states(env_name) # pull all starting init states for this task, and distribute to all workers
        max_episode_length = _get_max_episode_length(env_name)

        all_initial_states = [None] * num_parallel_envs
        def make_env_fn(env_name, initial_state, render, render_resolution, max_episode_length, obs_keys, keys_to_output_map, normalization_path, seed_val):
            def _init():
                wrapped_env, task_embedding = make_libero_env(
                    env_name=env_name,
                    initial_state=initial_state,
                    render=render,
                    render_resolution=render_resolution,
                    max_episode_length=max_episode_length,
                    obs_keys=obs_keys,
                    keys_to_output_map=keys_to_output_map,
                    normalization_path=normalization_path,
                    seed=seed_val
                )
                return wrapped_env
            return _init
        
        # for consistency, the offscreen-env used for video rendering will still be a subprocenv just with 1 subprocess
        offscreen_env_fn = [
            make_env_fn(env_name, all_initial_states[0], True, render_resolution, max_episode_length, obs_keys, keys_to_output_map, normalization_path, seed)
        ]

        # if eval_need_camera_obs:
            # assert num_parallel_envs == 1, "cannot parallelize eval environment as you have requested camera observations. Please pass num_parallel_envs as 1."
        
        # list of functions that when called, will create identical eval envs w/ just different seeds
        vec_env_fns = [
            make_env_fn(env_name, all_initial_states[(i + 1) % len(all_initial_states)], eval_need_camera_obs, render_resolution, max_episode_length, obs_keys, keys_to_output_map, normalization_path, seed + i + 1)
            for i in range(num_parallel_envs)
            ]
        
        self.vec_env = SubprocVectorEnv(vec_env_fns)        
        self.offscreen_env = SubprocVectorEnv(offscreen_env_fn)

        # create a single offscreen env to get the task embedding...
        self.task_embedding = OneHotEmbedding_Libero.encode(get_libero_task_from_env(env_name).language.replace(" ", "_").lower())
        self.env_str = env_name
    
    def get_eval_env(self):
        return self.vec_env
    
    def get_video_env(self):
        return self.offscreen_env

    def get_task_embedding(self):
        return self.task_embedding

    def get_env_str(self):
        return self.env_str





if __name__ == "__main__":
    # for testing 
    num_parallel_envs = 50
    env = make_env("libero_90-study_scene1-pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy", keys_to_load=['states'], num_parallel_envs=num_parallel_envs) # other states something like 'robot0_eef_pos', 'robot0_eef_quat' (which needs to get converted!!!), 'robot0_gripper_qpos', etc.
    dataset = get_dataset(env, "libero_90-study_scene1-pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy", keys_to_load=['states'])
    print(dataset)
    eval_env, video_env = env.get_eval_env(), env.get_video_env()


    # test parallelized iterartion over eval env
    num_video_episodes = 4
    num_eval_episodes = 50
    assert num_eval_episodes % num_parallel_envs == 0, "num_eval_episodes must be divisible by num_parallel_envs"
    num_iterations = num_eval_episodes // num_parallel_envs
    timesteps_per_episode = 500

    
    # test regular iteratio over video env
    print(f"testing regular iteration over video env for {num_video_episodes} episodes")
    for i in tqdm(range(num_video_episodes), position=0, leave=True):
        obs = video_env.reset()
        for i in tqdm(range(timesteps_per_episode), position=1, leave=False):
            actions = np.random.uniform(-1, 1, size=(7,))
            obs, reward, done, info = video_env.step(actions)

    
    print(f"testing parallelized iteration over eval env for {num_eval_episodes} episodes")
    for i in tqdm(range(num_iterations), position=0, leave=True):
        obs = eval_env.reset()
        for i in tqdm(range(timesteps_per_episode), position=1, leave=False):
            actions = np.random.uniform(-1, 1, size=(num_parallel_envs, 7))
            obs, reward, done, info = eval_env.step(actions)
