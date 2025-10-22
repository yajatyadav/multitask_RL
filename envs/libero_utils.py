from os.path import expanduser
import os
import pathlib


import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import imageio
import h5py
from tqdm import tqdm

# hack again for now
from utils.datasets import Dataset

import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'libero'))
from libero.libero.envs.env_wrapper import ControlEnv, OffScreenRenderEnv
from libero.libero.utils import get_libero_path
from libero.libero.envs import SubprocVectorEnv
from libero.libero import benchmark


LIBERO_WARMUP_STEPS = 15


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
    ## TODO(YY): need to implememt this, redo norm_stats computation to instead np.save() with keys as expected by normalize_obs and unnormalize_action below!!
    return None

def make_env(env_name, num_parallel_envs, render_resolution=128, keys_to_load=[], seed=0):
    """
    NOTE: should get_dataset() first, so that the metadata is downloaded before creating the environment
    """
    normalization_path = _get_normalization_path(env_name)
    max_episode_length = _get_max_episode_length(env_name)
    env = LiberoTopLevelEnvWrapper(env_name, seed, num_parallel_envs=num_parallel_envs, render_resolution=render_resolution, obs_keys=keys_to_load, max_episode_length=max_episode_length, normalization_path=normalization_path)
    return env

def _check_dataset_exists(env_name):
    # enforce that the dataset exists
    if env_name.startswith("libero_90") or env_name.startswith("libero_10"):
        suite, scene, task = env_name.split("-")
    else:
        suite, task = env_name.split("-")
        scene = ''
    file_name = f'{scene.upper()}_{task}_demo.hdf5'
    dataset_path = os.path.join(
        '/raid/users/yajatyadav/datasets/raw_libero', # fix!!
        suite.upper(),
        file_name
    )
    print(f"dataset path: {dataset_path}")
    assert os.path.exists(dataset_path)
    
    return dataset_path

def get_dataset(env, env_name, keys_to_load):
    dataset_path = _check_dataset_exists(env_name)

    rm_dataset = h5py.File(dataset_path, "r")
    demos = list(rm_dataset["data"].keys())
    num_demos = len(demos)
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    num_timesteps = 0
    for ep in demos:
        num_timesteps += int(rm_dataset[f"data/{ep}/actions"].shape[0])

    print(f"the size of the dataset is {num_timesteps}")
    # example_action = env.action_space.sample() ## can't do this as 'env' is a collection of 2 envs, that will be unpacked during eval time...

    # data holder
    observations = []
    actions = []
    next_observations = []
    terminals = []
    rewards = []
    masks = []
    
    # go through and add to the data holder; TODO(YY): only works for state-based observations right now...need to adapt later so keys_to_load can also specify 'obs/agentview_rgb' and 'obs/eye_in_hand_rgb'!
    for ep in demos:
        a = np.array(rm_dataset["data/{}/actions".format(ep)])
        obs, next_obs = [], []
        for k in keys_to_load:
            if k == 'states':
                obs.append(np.array(rm_dataset[f"data/{ep}/{k}"])[:, 1:]) # drop the first entry, which is the timestep
            else:
                obs.append(np.array(rm_dataset[f"data/{ep}/{k}"]))
        for k in keys_to_load:
            if k == 'states':
                obs_array = np.array(rm_dataset[f"data/{ep}/{k}"])[:, 1:]
            else:
                obs_array = np.array(rm_dataset[f"data/{ep}/{k}"])
            next_obs.append(np.concatenate([obs_array[1:], obs_array[-1:]], axis=0)) # make next obs by shifting obs array by 1, and then repeating the last element of obs array so obs and next_obs have same size
        obs = np.concatenate(obs, axis=-1)
        next_obs = np.concatenate(next_obs, axis=-1)
        dones = np.array(rm_dataset["data/{}/dones".format(ep)])
        r = np.array(rm_dataset["data/{}/rewards".format(ep)])
        
        observations.append(obs.astype(np.float32))
        actions.append(a.astype(np.float32))
        rewards.append(r.astype(np.float32))
        terminals.append(dones.astype(np.float32))
        masks.append(1.0 - dones.astype(np.float32))
        next_observations.append(next_obs.astype(np.float32))
    return Dataset.create(
        observations=np.concatenate(observations, axis=0),
        actions=np.concatenate(actions, axis=0),
        rewards=np.concatenate(rewards, axis=0),
        terminals=np.concatenate(terminals, axis=0),
        masks=np.concatenate(masks, axis=0),
        next_observations=np.concatenate(next_observations, axis=0),
    )

# ============================================================================
# NoRenderEnv - Environment without any rendering for subprocess compatibility
# ============================================================================

class NoRenderEnv(ControlEnv):
    """
    LIBERO environment without any rendering capabilities.
    This is the most compatible option for multiprocessing.
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
        obs_keys=[
            "robot0_eef_pos",
            "robot0_eef_quat", 
            "robot0_gripper_qpos",
        ],
        clamp_obs=False,
        init_state=None,
        can_render=False,
        render_hw=(256, 256),
        render_camera_name="agentview",
        max_episode_length=500,
    ):
        self.env = env
        self.can_render = can_render
        self.obs_keys = obs_keys
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
        self.observation_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )

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

    def get_observation(self):
        """Extract and concatenate relevant observation keys"""
        raw_obs = self.env.env._get_observations()
        obs_to_return = []
        
        for key in self.obs_keys:
            if key == 'states':
                sim_state = self.get_sim_state()
                obs_to_return.append(sim_state[1:]) # drop the first entry, which is the timestep
            else:
                obs_to_return.append(raw_obs[key])
        raw_obs = np.concatenate(obs_to_return, axis=0)


        if self.normalize:
            return self.normalize_obs(raw_obs)
        return raw_obs

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
    
    # def get_segmentation_of_interest(self, segmentation_image):
    #     """Get segmentation of objects of interest"""
    #     if hasattr(self.env, 'get_segmentation_of_interest'):
    #         return self.env.get_segmentation_of_interest(segmentation_image)
    #     else:
    #         # If not a segmentation env, return as-is
    #         return segmentation_image
    
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


def make_libero_env(
    env_name,
    initial_state,
    render=False,
    render_resolution=128,
    obs_keys=[
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    ],
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
    # Parse environment name
    if env_name.startswith("libero_90") or env_name.startswith("libero_10"):
        suite_str, scene_str, task_str = env_name.split("-", 2)
    else:
        suite_str, task_str = env_name.split("-", 1)
        scene_str = ''

    # Get task suite
    task_suite = benchmark.get_benchmark_dict()[suite_str]()
    num_tasks_in_suite = task_suite.n_tasks

    # Find the task
    desired_task_name = f"{scene_str.upper()}_{task_str}" if scene_str else task_str.upper()
    task = None
    for task_id in range(num_tasks_in_suite):
        candidate_task = task_suite.get_task(task_id)
        if candidate_task.name == desired_task_name:
            task = candidate_task
            break
    
    if task is None:
        raise ValueError(f"Task {desired_task_name} not found in suite {suite_str}")
    
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
    wrapped_env = LiberoEnvWrapper(
        env=base_env,
        normalization_path=normalization_path,
        obs_keys=obs_keys,
        max_episode_length=max_episode_length,
        can_render=render,
        render_hw=(render_resolution, render_resolution),
        init_state=initial_state,
    )
    
    return wrapped_env


class LiberoTopLevelEnvWrapper(gym.Env):
    """ To maintain API consistency with make_env_and_datasets, this is just a thin wrapper that holds a sbuprocvectorenv(liberoenvwrapper(norenderenv)) + liberoenvwrapper(offscreenrender). First used for fast evals and second for video logging """
    def __init__(
        self,
        env_name, # a string!!
        seed,
        num_parallel_envs,
        normalization_path=None,
        obs_keys=[
            "robot0_eef_pos",
            "robot0_eef_quat", 
            "robot0_gripper_qpos",
        ],
        render_resolution=128,
        max_episode_length=500,
    ):        
        all_initial_states = get_libero_task_init_states(env_name) # pull all starting init states for this task, and distribute to all workers
        # all_initial_states = [None] * num_parallel_envs
        def make_env_fn(env_name, initial_state, render, render_resolution, max_episode_length, obs_keys, normalization_path, seed_val):
            def _init():
                return make_libero_env(
                    env_name=env_name,
                    initial_state=initial_state,
                    render=render,
                    render_resolution=render_resolution,
                    max_episode_length=max_episode_length,
                    obs_keys=obs_keys,
                    normalization_path=normalization_path,
                    seed=seed_val
                )
            return _init
        
        # for consistency, the offscreen-env used for video rendering will still be a subprocenv just with 1 subprocess
        offscreen_env_fn = [
            make_env_fn(env_name, all_initial_states[0], True, render_resolution, max_episode_length, obs_keys, normalization_path, seed)
        ]
        
        # list of functions that when called, will create identical eval envs w/ just different seeds
        vec_env_fns = [
            make_env_fn(env_name, all_initial_states[(i + 1) % len(all_initial_states)], False, render_resolution, max_episode_length, obs_keys, normalization_path, seed + i + 1)
            for i in range(num_parallel_envs)
            ]
        self.vec_env = SubprocVectorEnv(vec_env_fns)

        
        self.offscreen_env = SubprocVectorEnv(offscreen_env_fn)
    def get_eval_env(self):
        return self.vec_env
    
    def get_video_env(self):
        return self.offscreen_env

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
