import argparse
import pickle as pkl
from tqdm import tqdm

from typing import Iterator, Tuple, Any
import glob
import numpy as np
import os
import cv2
import h5py
import json
import io
from collections import defaultdict
import random
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from transformers import AutoModel, AutoTokenizer
from hydra.utils import to_absolute_path
import torch
from data4robotics import transforms
import hydra
import yaml
from pathlib import Path



scene_names = ["KITCHEN_SCENE10", "KITCHEN_SCENE1", "KITCHEN_SCENE2", "KITCHEN_SCENE3", "KITCHEN_SCENE4", "KITCHEN_SCENE5", "KITCHEN_SCENE6", \
               "KITCHEN_SCENE7", "KITCHEN_SCENE8", "KITCHEN_SCENE9", "LIVING_ROOM_SCENE1", "LIVING_ROOM_SCENE2", "LIVING_ROOM_SCENE3", \
                "LIVING_ROOM_SCENE4", "LIVING_ROOM_SCENE4", "LIVING_ROOM_SCENE5", "LIVING_ROOM_SCENE6", "STUDY_SCENE1", "STUDY_SCENE2", \
                "STUDY_SCENE3", "STUDY_SCENE4", "LIBERO_OBJECT", "LIBERO_SPATIAL"]

task_names_all =  [
    "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it",
    "KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it",
    "KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it",
    "KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it",
    "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet",
    "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it",
    "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate",
    "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet",
    "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate",
    "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate",
    "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate",
    "KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet",
    "KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle",
    "KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl",
    "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove",
    "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove",
    "KITCHEN_SCENE3_turn_on_the_stove",
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it",
    "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet",
    "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet",
    "KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet",
    "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet",
    "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack",
    "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE5_put_the_black_bowl_on_the_plate",
    "KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet",
    "KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE6_close_the_microwave",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug",
    "KITCHEN_SCENE7_open_the_microwave",
    "KITCHEN_SCENE7_put_the_white_bowl_on_the_plate",
    "KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate",
    "KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove",
    "KITCHEN_SCENE8_turn_off_the_stove",
    "KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf",
    "KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet",
    "KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf",
    "KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet",
    "KITCHEN_SCENE9_turn_on_the_stove",
    "KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it",
    "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
    "LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray",
    "LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray",
    "LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray",
    "LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray",
    "LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray",
    "LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray",
    "LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray",
    "LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_put_it_in_the_tray",
    "LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray",
    "LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray",
    "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate",
    "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate",
    "LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate",
    "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate",
    "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
    "STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy",
    "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
    "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
    "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
    "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
    "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
    "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
    "STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy",
    "STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy",
    "STUDY_SCENE4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf",
    "STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf",
    "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf",
    "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf",

    # LIBERO OBJECT
    "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
    "pick_up_the_milk_and_place_it_in_the_basket",
    "pick_up_the_butter_and_place_it_in_the_basket",
    "pick_up_the_ketchup_and_place_it_in_the_basket",
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
    "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
    "pick_up_the_cream_cheese_and_place_it_in_the_basket",
    "pick_up_the_orange_juice_and_place_it_in_the_basket",
    "pick_up_the_salad_dressing_and_place_it_in_the_basket",

    # LIBERO SPATIAL
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo",
]


task_names =  [
    "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet",
    "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it",
    "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate",
    "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet",
    "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate",
    "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate",
    "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate",
    "KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet",
    "KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle",
    "KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl",
    "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove",
    "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove",
    "KITCHEN_SCENE3_turn_on_the_stove",
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it",
]


IMAGE_SIZE = (128, 128)
CAM_NAMES = ['agentview_rgb', 'eye_in_hand_rgb']


def crawler(dirname):
    # return glob.glob(os.path.join(dirname, '**/*.hdf5'), recursive=True)
    # return glob.glob(os.path.join(dirname, '**.hdf5'), recursive=True)
    files = []
    tasks = []
    for f in os.listdir(dirname):
        if 'hdf5' in f:
            files.append(os.path.join(dirname, f))
            tasks.append(f)
    return files, tasks


def _resize_and_encode( bgr_image, size=(128, 128)):
    bgr_image = cv2.resize(bgr_image, size, interpolation=cv2.INTER_AREA)
    _, encoded = cv2.imencode(".jpg", bgr_image)
    return encoded


def convert_dataset_object(max_demos=10, split=['90', 'object', 'spatial', 'goal'], name=None):
    paths = [f'../datasets/raw_libero/LIBERO_{split_i.upper()}' for split_i in split]

    out_trajs, all_acs = [], []
    all_states = []
    all_traj = {} 
    used_transitions = 0
    unused_transitions = 0

    for base_path in paths:
        print(base_path)
        episode_paths, tasks = crawler(base_path)

        for episode_path, task in tqdm(zip(episode_paths, tasks)):
            print(task)
            print(used_transitions, unused_transitions)
            task_idx = -1
            for i in range(len(task_names)):
                if task[0:-10] == task_names[i]:
                    task_idx = i
                    break
            print(task_idx)
            if task_idx == -1:
                continue
            task_embed = np.zeros(100)
            task_embed[task_idx] = 1.0
            assert (task_idx >= 0)
            count = 0
            task_demos = {}
            with h5py.File(episode_path, 'r') as f:
                for demo_key in f['data'].keys():
                    proc_traj = []
                    actions = f['data'][demo_key]['actions'][:]     
                    traj_states = np.concatenate((f['data'][demo_key]['obs']['gripper_states'], f['data'][demo_key]['obs']['joint_states']), axis=1)
                    task_demos[demo_key] = traj_states
            
                    for t, a in enumerate(actions):
                        used_transitions += 1
                        all_acs.append(a) # for normalization later
                        reward = 0 # dummy reward
                        robot_state = np.concatenate((f['data'][demo_key]['obs']['gripper_states'][t], f['data'][demo_key]['obs']['joint_states'][t]))
                        all_states.append(np.concatenate((f['data'][demo_key]['obs']['gripper_states'][t], f['data'][demo_key]['obs']['joint_states'][t])).flatten())
                        obs = dict(state=robot_state, task_idx=task_idx, task_name=task, demo_key=demo_key, traj_t=t)
                        for idx, key in enumerate(CAM_NAMES):
                            bgr_img = f['data'][demo_key]['obs'][key][t]
                            obs[f'enc_cam_{idx}'] = _resize_and_encode(bgr_img)
                        proc_traj.append((obs, a, reward))
                    out_trajs.append(proc_traj)
                    count += 1
                    if count >= max_demos:
                        break
                    # print(np.max(np.array(all_acs)), np.min(np.array(all_acs)))
            all_traj[task] = task_demos

    ac_dict = dict(loc=np.zeros(7).tolist(), scale=np.ones(7).tolist())
    data_dir = f'/home/ajwagen/research/dpt/dit-policy-bridge/data_buffers/libero_{split}'
    if name is not None:
        data_dir += f'_{name}'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    with open(data_dir + '/ac_norm_future.json', 'w') as f:
        json.dump(ac_dict, f)

    with open(data_dir + f'/buf_libero_object_{max_demos}.pkl', 'wb') as f:
        pkl.dump(out_trajs, f)



if __name__ == '__main__':
    convert_dataset_object(max_demos=50, split='90', name='kitchen1-3')