import os
import sys
repo_root_dir = os.getenv("MULTITASK_RL_REPO_ROOT_DIR", "/home/yajatyadav/multitask_reinforcement_learning/multitask_RL")
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
from utils.data_utils import MuseEmbedding
from utils.data_utils import unnormalize_action, normalize_proprio, normalize_image, unnormalize_image


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 128  # resolution used to render training data
TEXT_ENCODER = MuseEmbedding


@dataclass
class Args:
    seed: int = 0
    num_eval_episodes: int = 20
    num_steps_wait: int = 10
    video_frame_skip: int = 3
    eval_temperature: float = 1.0
    task_suite_name: str = "libero_10"
    task_name: str = ""




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
    logging.info(f"Task suite: {args.task_suite_name}")

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    env, task_description, initial_states = None, None, None
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        if task.name != args.task_name:
            continue
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        initial_states = task_suite.get_task_init_states(task_id)
        break
    
    if env is None:
        raise ValueError(f"Task {args.task_name} not found in task suite {args.task_suite_name}")
    return env, max_steps, task_description, initial_states


def format_libero_obs_for_agent(obs, task_embedding):
    proprio = np.concatenate((obs["robot0_eef_pos"],
                              _quat2axisangle(obs["robot0_eef_quat"]),
                              obs["robot0_gripper_qpos"],
                            ), dtype=np.float32)
    proprio = normalize_proprio(proprio)

    image_primary = np.array(obs["agentview_image"][::-1, ::])
    image_wrist = np.array(obs["robot0_eye_in_hand_image"][::-1, ::])
    
    image_primary = normalize_image(image_primary)
    image_wrist = normalize_image(image_wrist)

    

    proprio = proprio[np.newaxis, :] # go from (8,) to (1, 8)
    obs = {
        'task_embedding': task_embedding,
        'proprio': proprio,
        'image_primary': image_primary,
        'image_wrist': image_wrist,
    }
    return obs

def format_agent_action_for_libero(action, is_gripper_closed, num_consecutive_gripper_change_actions, STICKY_GRIPPER_NUM_STEPS=1):
    action = unnormalize_action(np.array(action))
    action = np.clip(action, -1, 1) # TODO(YY): might not be needed...

    # sticky gripper logic
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
    print(f"🤪🤪🤪 Task description: {task_description}")
    task_embedding = TEXT_ENCODER.encode(task_description)
    trajs = []
    stats = defaultdict(list)
    renders= []   
    wrist_renders = []

    # eval loop
    for episode_idx in tqdm.tqdm(range(args.num_eval_episodes)):
        traj = defaultdict(list)

        env.reset()    
        obs = env.set_init_state(initial_states[episode_idx])

        # setup sticky gripper, inspired from https://github.com/rail-berkeley/bridge_data_v2/blob/main/experiments/eval_lc.py
        is_gripper_closed = False
        num_consecutive_gripper_change_actions = 0

       
        render = []
        wrist_render = []
        t = 0
        
        while t < max_steps + args.num_steps_wait:
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue
            

            ## TODO(YY): format obs output by libero into obs expected by agent; especially NORMALIZATION !! 
            obs = format_libero_obs_for_agent(obs, task_embedding)
            action = actor_fn(observations=obs, temperature=args.eval_temperature)
            action = np.array(action)
            action, is_gripper_closed, num_consecutive_gripper_change_actions = format_agent_action_for_libero(action, is_gripper_closed, num_consecutive_gripper_change_actions)
            ## TODO(YY): format action output by agent into action expected by libero; especially UN-NORMALIZATION !! 

            next_obs, reward, done, info = env.step(action)
            t += 1

            if t % args.video_frame_skip == 0 or done:
                render.append(unnormalize_image(obs["image_primary"]))
                # wrist_render.append(obs["image_wrist"])

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

        add_to(stats, flatten(info))
        trajs.append(traj)
        renders.append(np.array(render))
        wrist_renders.append(np.array(wrist_render))
    
    for k, v in stats.items():
        stats[k] = np.mean(v)
    
    return stats, trajs, renders, wrist_renders