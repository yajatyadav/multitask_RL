# first create agent, then restore_agent_with_file
from utils.flax_utils import restore_agent_with_file
from envs.env_utils import make_env_and_datasets
from agents import agents
from envs.libero_utils import LiberoTopLevelEnvWrapper
from agents.acifql import get_config
import pickle
# eval on all 23 envs agent was trained on
from utils.log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger, get_sample_input_output_log_to_wandb, get_wandb_video
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation_libero import evaluate

import wandb
import tqdm
import numpy as np

class LoggingHelper:
    def __init__(self, wandb_logger):
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)


agent_class_str = 'acifql'
agent_class = agents[agent_class_str]

seed = 0
horizon_length = 5
discount = 0.99
batch_size = 256
encoder = 'combined_encoder_small'
KEYS_TO_LOAD = ['agentview_rgb', 'eye_in_hand_rgb', 'proprio', 'language']

def get_example_batch(batch_size):
    env_name = 'libero_90-kitchen_scene2'
    task_name = 'open_the_top_drawer_of_the_cabinet'
    augment_negative_demos = False
    num_parallel_envs = 1
    env, eval_env, train_dataset, val_dataset, names_to_return = make_env_and_datasets(env_name, task_name, augment_negative_demos, num_parallel_envs=num_parallel_envs, keys_to_load=KEYS_TO_LOAD)
    return train_dataset.sample_sequence(batch_size, sequence_length=horizon_length, discount=discount), names_to_return

def sweep_best_of_N(n_vals, restore_path=None):  
    
    example_batch, names_to_return = get_example_batch(batch_size)

     # Setup logging.
    prefixes = ["env", "eval"] + [f"eval_{names_to_return[i]}" for i in range(len(names_to_return))]
    prefixes.append("offline_agent")

    logger = LoggingHelper(
        wandb_logger=wandb,
    )

    infos = {}
    for n in n_vals:
        print(f"Evaluating agent with num_samples: {n}")
        agent_config = get_config()
        agent_config['encoder'] = encoder
        agent_config['num_samples'] = n
        agent_config['horizon_length'] = horizon_length
        agent = agent_class.create(seed, example_batch['observations'], example_batch['actions'], agent_config)
        
        # now restore
        if restore_path is not None:
            print(f"Restoring agent from {restore_path}")
            agent = restore_agent_with_file(agent, restore_path)
        
        eval_info = eval_agent(agent, example_batch, names_to_return, logger)
        infos[n] = eval_info

    with open('eval_libero_best_of_N_infos.pkl', 'wb') as f:
        pickle.dump(infos, f)
    return infos


# TODO(YY): make sure to set these to the 23 envs we trained on...
envs_to_eval = ['libero_spatial-pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate',
                'libero_object-pick_up_the_alphabet_soup_and_place_it_in_the_basket',
                'libero_goal-open_the_middle_drawer_of_the_cabinet',
                'libero_90-KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet',
                'libero_90-LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket',
                'libero_90-STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy']

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
EVAL_KEYS_TO_LOAD = {old_to_new_key[k] for k in KEYS_TO_LOAD}

def get_all_eval_envs():
    envs_list = []

    for i, env_name in enumerate(envs_to_eval):
        print(f"Getting eval env for {env_name}")

        env = LiberoTopLevelEnvWrapper(
            env_name=env_name,
            seed=seed + i,
            eval_need_camera_obs=True,
            num_parallel_envs=5,
            obs_keys=EVAL_KEYS_TO_LOAD,
            keys_to_output_map=keys_to_output_map,
            normalization_path=None,
        )
        envs_list.append(env)

    return envs_list



def eval_agent(agent, example_batch, names_to_return, logger):
    envs_list = get_all_eval_envs()
    print(f"Evaluating agent on {len(envs_list)} environments")
    all_eval_info = []
    for j, eval_env_j in tqdm.tqdm(enumerate(envs_list), total=len(envs_list), desc="Evaluating multi-task", position=0,leave=False):
        eval_info, trajs, renders = evaluate(
            agent=agent, 
            env=eval_env_j, 
            action_dim=example_batch["actions"].shape[-1], 
            num_eval_episodes=50, 
            num_video_episodes=3, 
            num_parallel_envs=5, 
            video_frame_skip=1)
        all_eval_info.append(eval_info)
        if len(renders) > 0:
            # value_and_reward_visualization(trajs, agent, FLAGS.save_dir, log_step)
            eval_info['video'] = get_wandb_video(renders)
        logger.log(eval_info, f"eval_{names_to_return[j]}", step=0)
        # remove video before taking mean
        if 'video' in eval_info:
            del eval_info['video']

    # aggregate eval info via mean, then log under "eval" prefix
    eval_info = {k: np.mean([eval_info[k] for eval_info in all_eval_info]) for k in all_eval_info[0].keys()}
    logger.log(eval_info, "eval", step=0)
    return eval_info



if __name__ == "__main__":
    wandb.init(
        entity='yajatyadav',
        project='multitask_RL',
        group='eval_libero_best_of_N',
        name='eval_libero_best_of_N',
    )
    

    n_vals=[1, 2, 4, 8, 16, 32, 64]
    restore_path = '/home/yajatyadav/multitask_reinforcement_learning/multitask_RL/exp/multitask_RL/DEBUG_full_libero_Q_learning_acifql_23_scene_all/all_libero-*/DEBUG_full_libero_Q_learning_acifql_23_scenessd00020251224_145540/params_25000.pkl'
    sweep_best_of_N(n_vals, restore_path)