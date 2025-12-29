# disable cuda preallocation
import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

from utils.flax_utils import restore_agent_with_file, restore_agent_actor_critic_separately
from envs.env_utils import make_env_and_datasets
from agents import agents
from envs.libero_utils import LiberoTopLevelEnvWrapper
from agents.acifql import get_config
import pickle

from utils.log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger, get_sample_input_output_log_to_wandb, get_wandb_video
import time
import sys
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
NUM_PARALLEL_ENVS = 5
NUM_EVAL_EPISODES = 50
NUM_VIDEO_EPISODES = 5
VIDEO_FRAME_SKIP = 3


def sweep_best_of_N(n_vals, env_name, task_name, actor_restore_path, critic_restore_path):  

    env, eval_env, train_dataset, val_dataset, names_to_return = make_env_and_datasets(env_name, task_name, augmentation_type='none', num_parallel_envs=NUM_PARALLEL_ENVS, keys_to_load=KEYS_TO_LOAD, use_hardcoded_eval_envs=False)
    example_batch = train_dataset.sample_sequence(batch_size, sequence_length=horizon_length, discount=discount)

     # Setup logging.
    prefixes = ["env", "eval"] + [f"eval_{names_to_return[i]}" for i in range(len(names_to_return))]
    prefixes.append("offline_agent")

    logger = LoggingHelper(
        wandb_logger=wandb,
    )

    infos = {"env_name": env_name, "task_name": task_name, "actor_restore_path": actor_restore_path, "critic_restore_path": critic_restore_path}
    for n in n_vals:
        print(f"Evaluating agent with num_samples: {n}")
        agent_config = get_config()
        agent_config['encoder'] = encoder
        agent_config['num_samples'] = n
        agent_config['horizon_length'] = horizon_length
        agent = agent_class.create(seed, example_batch['observations'], example_batch['actions'], agent_config)

    
        print(f"Restoring agent from actor {actor_restore_path} and critic {critic_restore_path}")
        agent = restore_agent_actor_critic_separately(agent, actor_restore_path, critic_restore_path)
        
        eval_info = eval_agent(agent, eval_env, example_batch, names_to_return, n,logger)
        infos[n] = eval_info
   
   # timestamp the suffix
    suffix = time.strftime("%Y%m%d_%H%M%S")
    with open(f'eval_libero_best_of_N_infos_{env_name}_{task_name}_{suffix}.pkl', 'wb') as f:
        pickle.dump(infos, f)
    return infos


def eval_agent(agent, eval_env, example_batch, names_to_return, n, logger):
    print(f"Evaluating agent on {len(eval_env)} environments")
    all_eval_info = []
    for j, eval_env_j in tqdm.tqdm(enumerate(eval_env), total=len(eval_env), desc="Evaluating multi-task", position=0,leave=False):
        eval_info, trajs, renders = evaluate(
            agent=agent, 
            env=eval_env_j, 
            action_dim=example_batch["actions"].shape[-1], 
            num_eval_episodes=NUM_EVAL_EPISODES, 
            num_video_episodes=NUM_VIDEO_EPISODES, 
            num_parallel_envs=NUM_PARALLEL_ENVS, 
            video_frame_skip=VIDEO_FRAME_SKIP)
        all_eval_info.append(eval_info)
        if len(renders) > 0:
            # value_and_reward_visualization(trajs, agent, FLAGS.save_dir, log_step)
            eval_info['video'] = get_wandb_video(renders)
        logger.log(eval_info, f"eval_{names_to_return[j]}", step=n)
        # remove video before taking mean
        if 'video' in eval_info:
            del eval_info['video']

    # aggregate eval info via mean, then log under "eval" prefix
    eval_info = {k: np.mean([eval_info[k] for eval_info in all_eval_info]) for k in all_eval_info[0].keys()}
    logger.log(eval_info, "eval", step=n)
    print(f"Eval info: {eval_info}")
    return eval_info



if __name__ == "__main__":   
    root_dir = '/home/yajatyadav/multitask_reinforcement_learning/multitask_RL/exp/multitask_RL'

    # actor paths
    one_task_actor_ckpt = 'bcflowactor_only/libero_90-living_room_scene1/bcflowactor_livingroomscene1__alphabet_soup_25_demos_IMAGE_sd00020251227_144306/params_140000.pkl'
    two_task_actor_ckpt = 'bcflowactor_only/libero_90-living_room_scene1/bcflowactor_livingroomscene1__alphabet_soup_ketchup_25_demos_IMAGE_sd00020251227_144347/params_140000.pkl'

    # critic paths
    onetask_critic_no_aug_ckpt = 'instruction_following_Q/libero_90-living_room_scene1/libero90_livingroomscene1_singletask_none_augmentation_IMAGE_sd00020251226_214011/params_125000.pkl'
    two_task_critic_no_aug_ckpt = 'instruction_following_Q/libero_90-living_room_scene1/libero90_livingroomscene1_twotask_none_augmentation_IMAGE_sd00020251226_214200/params_125000.pkl'
    onetask_critic_aug_one_other_task_ckpt = 'instruction_following_Q/libero_90-living_room_scene1/libero90_livingroomscene1_singletask_one_other_task__augmentation_IMAGE_sd00020251226_215816/params_250000.pkl'
    two_task_critic_aug_each_other_task_ckpt = 'instruction_following_Q/libero_90-living_room_scene1/libero90_livingroomscene1_twotask_augment_each_other_IMAGE_sd00020251226_215758/params_250000.pkl'

    # TODO(YY): always edit these!!!
    n_vals=[1, 2, 4, 8, 16, 32, 64, 128]
    actor_ckpt = two_task_actor_ckpt
    critic_ckpt = two_task_critic_aug_each_other_task_ckpt
    suffix = 'twotask_aug_each_other_task_actor_25_demos_140k_step_critic_250k_step'
    
    if 'singletask' in critic_ckpt:
        env_name = 'libero_90-living_room_scene1'
        task_name = 'pick_up_the_alphabet_soup_and_put_it_in_the_basket'
    elif 'twotask' in critic_ckpt:
        env_name = 'libero_90-living_room_scene1'
        task_name = 'pick_up_the_alphabet_soup_and_put_it_in_the_basket|pick_up_the_ketchup_and_put_it_in_the_basket'
    else:
        raise ValueError(f"Invalid suffix: {suffix}")


    actor_restore_path = os.path.join(root_dir, actor_ckpt)
    critic_restore_path = os.path.join(root_dir, critic_ckpt)
    wandb_name = f'eval_libero90_livingroomscene1_{suffix}'
    
    
    wandb.init(
        entity='yajatyadav',
        project='multitask_RL',
        group='eval_libero_best_of_N',
        name=wandb_name,
    )
    sweep_best_of_N(n_vals, env_name, task_name, actor_restore_path, critic_restore_path)