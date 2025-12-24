import sys
import os
repo_root_dir = os.getenv("MULTITASK_RL_REPO_ROOT_DIR", os.getcwd())
sys.path.insert(0, os.path.join(repo_root_dir, "libero"))

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import imageio
import pathlib
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils import get_libero_path


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
        render_hw=(256, 256),
        render_camera_name="agentview",
        max_episode_length=500,
    ):
        self.env = env
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
        
        # Extract specified keys
        obs_list = []
        for key in self.obs_keys:
            if key in raw_obs:
                obs_list.append(raw_obs[key])
            else:
                # Handle missing keys gracefully
                print(f"Warning: observation key '{key}' not found in environment")
        
        if len(obs_list) == 0:
            raise ValueError(f"None of the specified obs_keys {self.obs_keys} found in environment observations")
        
        raw_obs_array = np.concatenate(obs_list, axis=0)
        
        if self.normalize:
            return self.normalize_obs(raw_obs_array)
        return raw_obs_array

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

        return self.get_observation(), {}

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
        success = self.env.check_success()
        if success:
            done = True
            info["success"] = 1
        else:
            info["success"] = 0

        # Handle termination vs truncation (gym >= 0.26 style)
        if done:
            return obs, reward, True, False, info
        if self.t >= self.max_episode_length:
            return obs, reward, False, True, info
        return obs, reward, False, False, info

    def render(self, mode="rgb_array"):
        """Render the environment"""
        h, w = self.render_hw
        # LIBERO uses different camera names, map common names
        camera_name = self.render_camera_name
        if camera_name not in self.env.env.sim.model.camera_names:
            # Default to first available camera
            camera_name = self.env.env.sim.model.camera_names[0]
        
        return self.env.env.sim.render(
            width=w,
            height=h,
            camera_name=camera_name,
        )
    
    def check_success(self):
        """Check if task is successful"""
        return self.env.check_success()
    
    def get_segmentation_of_interest(self, segmentation_image):
        """Get segmentation of objects of interest"""
        if hasattr(self.env, 'get_segmentation_of_interest'):
            return self.env.get_segmentation_of_interest(segmentation_image)
        else:
            # If not a segmentation env, return as-is
            return segmentation_image
    
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


def make_libero_env(
    env_name,
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
    Factory function to create a wrapped LIBERO environment.
    
    Args:
        env_name: Environment name in format "suite-scene-task" (e.g., "libero_90-study_scene1-pick_up_book")
        render_resolution: Resolution for rendering
        obs_keys: List of observation keys to extract
        normalization_path: Path to normalization statistics (optional)
        max_episode_length: Maximum episode length
        seed: Random seed
    
    Returns:
        LiberoEnvWrapper instance
    """
    # Parse environment name
    if env_name.startswith("libero_90") or env_name.startswith("libero_10"):
        suite_str, scene_str, task_str = env_name.split("-")
    else:
        suite_str, task_str = env_name.split("-")
        scene_str = ''

    # Get task suite
    task_suite = benchmark.get_benchmark_dict()[suite_str]()
    num_tasks_in_suite = task_suite.n_tasks

    # Find the task
    desired_task_name = f"{scene_str.upper()}_{task_str}"
    task = None
    for task_id in range(num_tasks_in_suite):
        candidate_task = task_suite.get_task(task_id)
        if candidate_task.name == desired_task_name:
            task = candidate_task
            break
    
    if task is None:
        raise ValueError(f"Task {desired_task_name} not found in suite {suite_str}")
    
    # Build environment
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": render_resolution,
        "camera_widths": render_resolution
    }
    base_env = OffScreenRenderEnv(**env_args)
    base_env.seed(seed)
    
    # Wrap environment
    wrapped_env = LiberoEnvWrapper(
        env=base_env,
        normalization_path=normalization_path,
        obs_keys=obs_keys,
        max_episode_length=max_episode_length,
        render_hw=(render_resolution, render_resolution),
    )
    
    return wrapped_env


if __name__ == "__main__":
    # Create a wrapped LIBERO environment
    env = make_libero_env(
        env_name="libero_90-study_scene1-pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
        render_resolution=128,
        max_episode_length=500,
        seed=42
    )
    dummy_action = [0.0] * 6 + [-1.0]


    # Use like a standard gym environment
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(dummy_action)
    env.close()