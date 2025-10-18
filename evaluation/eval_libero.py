import os
import sys
repo_root_dir = os.getenv("MULTITASK_RL_REPO_ROOT_DIR", os.getcwd())
sys.path.insert(0, os.path.join(repo_root_dir, "libero"))

import logging
import math
import pathlib
import numpy as np
import tqdm
from dataclasses import dataclass
from collections import defaultdict
import jax

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from utils.data_utils import get_language_encoder
from utils.data_utils import normalize_libero_eval_obs_for_agent, unnormalize_action_mean_std, unnormalize_action_min_max

from utils.data_utils import LIBERO_ENV_RESOLUTION
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

@dataclass
class Args:
    seed: int = 0
    num_eval_episodes: int = 20
    num_steps_wait: int = 10
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


def format_libero_obs_for_agent(obs, task_embedding, sim_state, dataset_name):
    
    # construct and normalize proprio
    proprio = np.concatenate((obs["robot0_eef_pos"],
                              _quat2axisangle(obs["robot0_eef_quat"]),
                              obs["robot0_gripper_qpos"],
                            ), dtype=np.float32)

    # flip image horizontally and normalize into [-1, 1]
    image_primary = np.array(obs["agentview_image"][::-1, ::])
    image_wrist = np.array(obs["robot0_eye_in_hand_image"][::-1, ::])

    sim_state = np.array(sim_state, dtype=np.float32)
    
    # add a batch dimension to all elements
    obs_to_normalize = {
            'task_embedding': task_embedding[np.newaxis, :],
            'image_primary': image_primary[np.newaxis, :],
            'image_wrist': image_wrist[np.newaxis, :],
            'proprio': proprio[np.newaxis, :],
            'sim_state': sim_state[np.newaxis, :],
    }
    obs = normalize_libero_eval_obs_for_agent(obs_to_normalize, dataset_name=dataset_name)
    return obs

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
    task_embedding = TEXT_ENCODER.encode(task_description)
    trajs = []
    stats = defaultdict(list)
    renders = []   
    wrist_renders = []

    # eval loop
    num_success_episodes = 0
    for episode_idx in tqdm.tqdm(range(args.num_eval_episodes), desc=f"🤪🤪🤪  Evaulating episodes on task: {task_description}, num_success_episodes: {num_success_episodes}", position=0, leave=True):
        traj = defaultdict(list)

        env.reset()    
        obs = env.set_init_state(initial_states[episode_idx])

        # setup sticky gripper, inspired from https://github.com/rail-berkeley/bridge_data_v2/blob/main/experiments/eval_lc.py
        is_gripper_closed = False
        num_consecutive_gripper_change_actions = 0

        # store frames for video rendering      
        render = []
        wrist_render = []

        # take some dummy steps to initialize the environment
        for _ in range(args.num_steps_wait):
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            sim_state = env.get_sim_state()
        
        # start the main eval loop per episode
        for t in tqdm.trange(max_steps, desc=f"Episode {episode_idx} Progress ", position=1, leave=False):            
            # format input/output from actor
            obs_for_actor = format_libero_obs_for_agent(obs, task_embedding, sim_state, args.dataset_name)
            action = actor_fn(observations=obs_for_actor, temperature=args.eval_temperature)
            action, is_gripper_closed, num_consecutive_gripper_change_actions = format_agent_action_for_libero(action, args.dataset_name, is_gripper_closed, num_consecutive_gripper_change_actions)
            # take action in environment
            next_obs, reward, done, info = env.step(action)
            sim_state = env.get_sim_state()
            info['success'] = float(1 if done else 0)

            if t % args.video_frame_skip == 0 or done:
                # TODO(YY): when normalizing images is added, make sure to unnormalize them here!
                render.append(obs["agentview_image"][::-1, ::])
                wrist_render.append(obs["robot0_eye_in_hand_image"][::-1, ::])

            transition = dict(
                observation=obs,
                next_observation=next_obs,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            obs = next_obs

            if done:
                num_success_episodes += 1
                break

        add_to(stats, flatten(info))
        trajs.append(traj)
        renders.append(np.array(render))
        wrist_renders.append(np.array(wrist_render))
    
    env.close()
    for k, v in stats.items():
        stats[k] = np.mean(v)    
    
    return stats, trajs, renders, wrist_renders