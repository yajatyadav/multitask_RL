import os
import sys
repo_root_dir = os.getenv("MULTITASK_RL_REPO_ROOT_DIR", os.getcwd())
sys.path.insert(0, os.path.join(repo_root_dir, "libero"))

import logging
import copy
import math
import pathlib
import numpy as np
import tqdm
from dataclasses import dataclass
from collections import defaultdict, deque
import jax

from openpi_client import image_tools

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from utils.data_utils import get_language_encoder
from utils.data_utils import normalize_libero_eval_obs_for_agent, unnormalize_action_mean_std, unnormalize_action_min_max

from utils.data_utils import LIBERO_ENV_RESOLUTION
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

@dataclass
class Args:
    eval_use_images: bool # set to False for state-based evals
    eval_with_pi0: bool # set to True to use pi0 with best-of-N sampling actions
    seed: int = 0
    num_eval_episodes: int = 20
    num_video_episodes: int = 5 # render only the first 5 episodes
    num_steps_wait: int = 10
    num_replan_steps: int = 1 # number of open-loop steps to execute in eval before requeing the actor. This should be less than actor action horizon.
    video_frame_skip: int = 3
    eval_temperature: float = 1.0
    task_suite_name: str = "libero_10"
    task_name: str = ""
    dataset_name: str = "" # used to fetch norm stats
    text_encoder: str = "one_hot_libero"

def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)

def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description

def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def setup_eval_env(args: Args):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    max_steps_dict = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    if args.task_suite_name not in max_steps_dict:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    max_steps = max_steps_dict[args.task_suite_name]


    env, task_description, initial_states = None, None, None
    for task_id in range(num_tasks_in_suite):
        task = task_suite.get_task(task_id)
        if task.name == args.task_name:
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            initial_states = task_suite.get_task_init_states(task_id)
            break    
    
    if env is None:
        raise ValueError(f"Task {args.task_name} not found in task suite {args.task_suite_name}")
    return env, max_steps, task_description, initial_states



def format_libero_obs_for_agent(obs, prompt, task_embedding, sim_state, dataset_name, eval_use_images: bool = False):
    
    # construct and normalize proprio
    proprio = np.concatenate((obs["robot0_eef_pos"],
                              _quat2axisangle(obs["robot0_eef_quat"]),
                              obs["robot0_gripper_qpos"],
                            ), dtype=np.float32)

    sim_state = np.array(sim_state, dtype=np.float32)
    
    # add a batch dimension to all elements
    obs_to_normalize = {
            'task_embedding': task_embedding[np.newaxis, :],
            'proprio': proprio[np.newaxis, :],
            'sim_state': sim_state[np.newaxis, :],
            # 'prompt': prompt[np.newaxis, :],
    }
    if eval_use_images:
        # flip image horizontally and normalize into [-1, 1]
        image_primary = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]) # going back to flipping both, to be consistent with the training pipeline
        image_wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        obs_to_normalize.update({
            'image_primary': image_primary[np.newaxis, :],
            'image_wrist': image_wrist[np.newaxis, :],
        })
        
    obs = normalize_libero_eval_obs_for_agent(obs_to_normalize, dataset_name=dataset_name)
    return obs

def format_libero_obs_for_pi0_obs(obs, task_description):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, 224, 224)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, 224, 224)
    )    
    obs_pi_zero = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    ),
                    "prompt": str(task_description),
                }
    return obs_pi_zero

def format_agent_action_for_libero(action, dataset_name, is_gripper_closed, num_consecutive_gripper_change_actions, STICKY_GRIPPER_NUM_STEPS=1):
    # first, unnormalize + clip the non-gripper actions
    action = unnormalize_action_min_max(np.array(action, dtype=np.float32), dataset_name=dataset_name).flatten() # flatten to remove batch dimension
    action = np.clip(action, -1, 1) # MUST have this - without it, env.step() can eventually go out of bounds and crash mujoco

    # sticky gripper logic: set gripper to -1/1 based on consecutive gripper change requests
    if (action[-1] > 0) != is_gripper_closed:
        num_consecutive_gripper_change_actions += 1
    else:
        num_consecutive_gripper_change_actions = 0
    if (
        num_consecutive_gripper_change_actions
        >= STICKY_GRIPPER_NUM_STEPS
    ):
        is_gripper_closed = not is_gripper_closed
        num_consecutive_gripper_change_actions = 0
    
    action[-1] = 1.0 if is_gripper_closed else -1.0
    return action, is_gripper_closed, num_consecutive_gripper_change_actions


def evaluate(agent, args: Args):
    # setup actor and env
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    env, max_steps, task_description, initial_states = setup_eval_env(args)
    TEXT_ENCODER = get_language_encoder(args.text_encoder)
    task_embedding = TEXT_ENCODER.encode([task_description]) # putting in list to add a batch dimension
    trajs = []
    stats = defaultdict(list)
    renders = []   
    wrist_renders = []

    # eval loop
    num_success_episodes = 0
    for episode_idx in tqdm.tqdm(range(args.num_eval_episodes), desc=f"🤪🤪🤪  Evaulating episodes on task: {task_description} ", position=0, leave=True):
        # traj = defaultdict(list)

        env.reset()    
        obs = env.set_init_state(initial_states[episode_idx])
        action_plan = deque()

        # setup sticky gripper, inspired from https://github.com/rail-berkeley/bridge_data_v2/blob/main/experiments/eval_lc.py
        is_gripper_closed = False
        num_consecutive_gripper_change_actions = 0

        # store frames for video rendering      
        render = []
        # wrist_render = []

        # take some dummy steps to initialize the environment
        for _ in range(args.num_steps_wait):
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
        sim_state = env.get_sim_state()
        
        # start the main eval loop per episode
        for t in tqdm.trange(max_steps, desc=f"Episode {episode_idx} Progress ", position=1, leave=False):  
            if (episode_idx < args.num_video_episodes) and (t % args.video_frame_skip == 0 or done):
                render.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1], dtype=np.uint8)) # need to copy since obs images get changed in-place later on during normalization
                # wrist_render.append(obs["robot0_eye_in_hand_image"][::-1, ::-1])

            if not action_plan: # only bother with below steps if action plan is empty
                
                # format input/output for pi0
                if args.eval_with_pi0:
                    pi0_obs = copy.deepcopy(obs) # take deepcopy of raw observation to feed into pi0 (pi0 expects completely raw obs)
                    pi0_obs = format_libero_obs_for_pi0_obs(pi0_obs, task_description)
                
                # format input/output for value networks. This format_fn might modify the obs IN PLACE!
                obs = format_libero_obs_for_agent(obs, task_description, task_embedding, sim_state, args.dataset_name, args.eval_use_images)

                if args.eval_with_pi0:
                    action_chunk, sampling_info = actor_fn(value_obs=obs, pi0_obs=pi0_obs, temperature=args.eval_temperature)
                else:
                    action_chunk, sampling_info = actor_fn(observations=obs, temperature=args.eval_temperature), {} # TODO(YY): log some sampling metrics in other methods like AWR?
                    action_chunk, is_gripper_closed, num_consecutive_gripper_change_actions = format_agent_action_for_libero(action, args.dataset_name, is_gripper_closed, num_consecutive_gripper_change_actions) # extra layer of formatting needed if using awr/ddpg+c; if using openpi, there is already a transform stack for unnormalizing, etc.

                assert len(action_chunk) >= args.num_replan_steps, f"We want to replan every {args.num_replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                action_plan.extend(action_chunk[: args.num_replan_steps])
                
            # pop next action from plan, and take it in environment
            action = action_plan.popleft()            
            next_obs, reward, done, info = env.step(action)
            sim_state = env.get_sim_state()
            info['success'] = float(1 if done else 0)
            info.update(sampling_info)
            obs = next_obs

            # transition = dict(
            #     observation=obs,
            #     next_observation=next_obs,
            #     action=action,
            #     reward=reward,
            #     done=done,
            #     info=info,
            # )
            # add_to(traj, transition)
            

            if done:
                num_success_episodes += 1
                break

        add_to(stats, flatten(info))
        # trajs.append(traj)
        if episode_idx < args.num_video_episodes:
            renders.append(np.array(render, dtype=np.uint8))
            # wrist_renders.append(np.array(wrist_render))
    
    env.close()
    for k, v in stats.items():
        stats[k] = np.mean(v)    
    
    return stats, trajs, renders, wrist_renders
