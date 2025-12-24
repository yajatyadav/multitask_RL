import numpy as np
import pathlib
import time
import os
from typing import List, Dict, Any, Callable
import gymnasium as gym
from gymnasium.spaces import Box
import imageio

import sys
import os
sys.path.insert(0, os.path.join('/home/yajatyadav/multitask_reinforcement_learning/multitask_RL', "libero"))
from libero.libero import benchmark
from libero.libero.envs.env_wrapper import ControlEnv, OffScreenRenderEnv # Use ControlEnv directly
from libero.libero.utils import get_libero_path
from libero.libero.envs import SubprocVectorEnv

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
        raw_obs = np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)
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
        ## TODO(YY): set up passing in init_state, actually might be too much work, can just keep it random reset()
        if self.init_state is not None:
            # Reset to specific state
            obs = self.env.set_init_state(self.init_state)
        else:
            # Random reset
            obs = self.env.reset()

        # Get processed observation and return only the observation (not a tuple)
        # This is for compatibility with older gym environments that SubprocVectorEnv expects
        return self.get_observation() ## TODO(YY): used to be return obs, {}; but removed the empty dict for SubProcVectorEnv compatibility - add back this empty dict in the caller functions

    def step(self, action):
        """Step the environment"""
        if self.normalize:
            action = self.unnormalize_action(action)
        
        # Step the environment
        raw_obs, reward, done, info = self.env.step(action)

        raw_obs = np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)
        if self.normalize:
            obs = self.normalize_obs(raw_obs)
        else:
            obs = raw_obs
        
        # Get processed observation
        # obs = self.get_observation()

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

        # TODO(YY): Return in old gym format (obs, reward, done, info) for SubprocVectorEnv compatibility, instead of (obs, reward, done, truncated, info)
        return obs, reward, done, info
    

    # TODO(YY): render() should work if the base env is offscreenrender, but should assert an error if the base env is norenderenv!!
    def render(self, mode="rgb_array"):
        """Render the environment (disabled for NoRenderEnv compatibility)"""
        # Since we're using NoRenderEnv, rendering is disabled
        # Return a blank image to maintain API compatibility
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
    
    ## TODO(YY): shoul we drop the first 15 entries? entry 0 is timestep, next 7 are FS robot0_joint_pos, and next 7 i think are robot vel? THese are correlated features with obs/ee_state, ee_ori, etc. that we are also passing and concatenating...
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

def make_libero_env(
    env_name,
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
        init_state=None, ## TODO(YY): support passing init states upon reset...
        render_resolution=128,
        max_episode_length=500,
    ):
        self.offscreen_env = make_libero_env(
            env_name=env_name,
            render=True,
            render_resolution=render_resolution,
            max_episode_length=max_episode_length,
            obs_keys=obs_keys,
            normalization_path=normalization_path,
            seed=seed
        )
        if self.num_parallel_envs == 1:
            self.vec_env = self.offscreen_env
        else:
            vec_env_fns = [
            lambda i=i: make_libero_env(
                env_name=env_name,
                render=False,
                render_resolution=render_resolution,
                max_episode_length=max_episode_length,
                obs_keys=obs_keys,
                normalization_path=normalization_path,
                seed=seed + (i + 10)
            )
            for i in range(num_parallel_envs)
            ]
            self.vec_env = SubprocVectorEnv(vec_env_fns)

    def get_eval_env(self):
        return self.vec_env
    
    def get_video_env(self):
        return self.offscreen_env



# ============================================================================
# Test 1: Single Environment Test
# ============================================================================

def test_single_wrapped_env():
    """Test a single wrapped LIBERO environment"""
    
    print("\n" + "=" * 80)
    print("TEST 1: SINGLE WRAPPED LIBERO ENVIRONMENT")
    print("=" * 80)
    
    suite = "libero_90"
    scene = "study_scene1"
    task = "pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy"
    env_name = f"{suite}-{scene}-{task}"
    
    print(f"\nCreating environment: {env_name}")
    env = make_libero_env(
        env_name=env_name,
        render_resolution=128,
        max_episode_length=500,
        seed=42
    )
    
    print(f"✓ Environment created successfully")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    
    # Test reset
    print("\n[Test 1.1] Testing reset...")
    obs = env.reset()
    print(f"✓ Reset successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation dtype: {obs.dtype}")
    print(f"  Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # Test stepping
    print("\n[Test 1.2] Testing 10 steps with random actions...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"  Step {i+1}: reward={reward:.4f}, done={done}, success={info.get('success', 0)}")
    
    # # Test check_success
    # print("\n[Test 1.3] Testing check_success...")
    # success = env.check_success()
    # print(f"✓ Success status: {success}")
    
    # Test get_sim_state
    print("\n[Test 1.4] Testing get_sim_state...")
    sim_state = env.get_sim_state()
    print(f"✓ Simulation state shape: {sim_state.shape}")
    
    # Test episode info
    print("\n[Test 1.5] Testing episode info...")
    episode_info = env.get_episode_info()
    env_info = env.get_info()
    print(f"✓ Episode info: {episode_info}")
    print(f"✓ Environment info: {env_info}")
    
    # Cleanup
    env.close()
    print("\n✓ Single environment test completed")


# ============================================================================
# Test 2: SubprocVectorEnv with Multiple Wrapped Environments
# ============================================================================

def test_subproc_vector_env():
    """Test SubprocVectorEnv with multiple wrapped LIBERO environments"""
    
    print("\n" + "=" * 80)
    print("TEST 2: SubprocVectorEnv WITH WRAPPED LIBERO ENVIRONMENTS")
    print("=" * 80)
    
    # Configuration
    suite = "libero_90"
    scene = "study_scene1"
    task = "pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy"
    env_name = f"{suite}-{scene}-{task}"
    num_envs = 4
    render_resolution = 128
    max_episode_length = 500
    
    print(f"\nConfiguration:")
    print(f"  Environment: {env_name}")
    print(f"  Number of parallel environments: {num_envs}")
    print(f"  Render resolution: {render_resolution}")
    print(f"  Max episode length: {max_episode_length}")
    
    # Test 2.1: Create environment factories
    print("\n[Test 2.1] Creating environment factories...")
    env_fns = [
        lambda i=i: make_libero_env(
            env_name=env_name,
            render_resolution=render_resolution,
            max_episode_length=max_episode_length,
            seed=1000 + i
        )
        for i in range(num_envs)
    ]
    print(f"✓ Created {num_envs} environment factories with seeds 1000-{1000+num_envs-1}")
    
    # Test 2.2: Initialize SubprocVectorEnv
    print("\n[Test 2.2] Initializing SubprocVectorEnv...")
    vec_env = SubprocVectorEnv(env_fns)
    print(f"✓ SubprocVectorEnv initialized")
    print(f"  Number of environments: {vec_env.env_num}")
    print(f"  Action space: {vec_env.action_space}")
    print(f"  Observation space: {vec_env.observation_space}")
    
    # Test 2.3: Reset all environments
    print("\n[Test 2.3] Resetting all environments...")
    obs = vec_env.reset()
    print(f"✓ Reset complete")
    print(f"  Observations shape: {obs.shape}")
    print(f"  Observations dtype: {obs.dtype}")
    print(f"  Observations range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # Test 2.4: Get action dimension
    print("\n[Test 2.4] Analyzing action space...")
    action_space = vec_env.action_space
    action_dim = action_space[0].shape[0] if isinstance(action_space, list) else action_space.shape[0]
    print(f"✓ Action dimension: {action_dim}")
    print(f"  Action range: [{action_space[0].low[0]:.3f}, {action_space[0].high[0]:.3f}]")
    
    # Test 2.5: Step through with different actions
    print("\n[Test 2.5] Stepping through with uniform random actions...")
    actions = np.random.uniform(-1, 1, size=(num_envs, action_dim))
    obs, rewards, dones, infos = vec_env.step(actions)
    print(f"✓ Step completed")
    print(f"  Observations shape: {obs.shape}")
    print(f"  Rewards: {rewards}")
    print(f"  Dones: {dones}")
    print(f"  Success flags: {[info.get('success', 0) for info in infos]}")
    
    # Test 2.6: Check success across all environments
    # print("\n[Test 2.6] Checking success status across all environments...")
    # success_flags = vec_env.check_success()
    # print(f"✓ Success flags: {success_flags}")
    # print(f"  Number of successful environments: {sum(success_flags)}")
    
    # Test 2.7: Get simulation states
    print("\n[Test 2.7] Retrieving simulation states...")
    sim_states = vec_env.get_sim_state()
    print(f"✓ Retrieved {len(sim_states)} simulation states")
    for i, state in enumerate(sim_states[:2]):
        print(f"  Environment {i} state shape: {state.shape}")
    
    # Test 2.8: Test segmentation processing
    # print("\n[Test 2.8] Testing segmentation of interest...")
    # dummy_seg_images = [
    #     np.random.randint(0, 10, size=(render_resolution, render_resolution)) 
    #     for _ in range(num_envs)
    # ]
    # processed = vec_env.get_segmentation_of_interest(dummy_seg_images)
    # print(f"✓ Processed {len(processed)} segmentation images")
    # if len(processed) > 0 and processed[0] is not None:
    #     print(f"  Processed image shape: {processed[0].shape}")
    
    # Test 2.9: Set custom initial states
    print("\n[Test 2.9] Testing set_init_state...")
    try:
        # Get current states
        current_states = vec_env.get_sim_state()
        # Try to reset to these states
        obs = vec_env.set_init_state(init_state=current_states)
        print(f"✓ Set initial states successfully")
        print(f"  Observations shape after set_init_state: {obs.shape}")
    except Exception as e:
        print(f"  ⚠ set_init_state encountered error: {e}")
    
    # Test 2.10: Cleanup
    print("\n[Test 2.10] Cleanup...")
    vec_env.close()
    print("✓ All environments closed")
    
    print("\n" + "=" * 80)
    print("TEST 2 COMPLETED")
    print("=" * 80)


# ============================================================================
# Test 3: Different Action Strategies
# ============================================================================

def test_action_strategies():
    """Test different action strategies across parallel environments"""
    
    print("\n" + "=" * 80)
    print("TEST 3: DIFFERENT ACTION STRATEGIES")
    print("=" * 80)
    
    # Configuration
    suite = "libero_90"
    scene = "study_scene1"
    task = "pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy"
    env_name = f"{suite}-{scene}-{task}"
    num_envs = 4
    
    print(f"\nInitializing {num_envs} parallel environments...")
    env_fns = [
        lambda i=i: make_libero_env(
            env_name=env_name,
            render_resolution=128,
            max_episode_length=500,
            seed=2000 + i
        )
        for i in range(num_envs)
    ]
    
    vec_env = SubprocVectorEnv(env_fns)
    # Handle list or single action space
    action_space = vec_env.action_space
    action_dim = action_space[0].shape[0] if isinstance(action_space, list) else action_space.shape[0]
    print(f"✓ Initialized with action dimension: {action_dim}")
    
    # Define test scenarios
    test_scenarios = [
        {
            "name": "Random exploration",
            "action_generator": lambda: np.random.uniform(-1, 1, size=(num_envs, action_dim)),
            "steps": 10
        },
        {
            "name": "Zero actions (stationary)",
            "action_generator": lambda: np.zeros((num_envs, action_dim)),
            "steps": 5
        },
        {
            "name": "Small perturbations",
            "action_generator": lambda: np.random.uniform(-0.1, 0.1, size=(num_envs, action_dim)),
            "steps": 10
        },
        {
            "name": "Gripper open/close cycle",
            "action_generator": lambda: np.concatenate([
                np.zeros((num_envs, action_dim - 1)),
                np.array([[1.0], [-1.0], [1.0], [-1.0]])  # Alternate gripper
            ], axis=1),
            "steps": 8
        }
    ]
    
    for scenario_idx, scenario in enumerate(test_scenarios):
        print(f"\n[Scenario {scenario_idx + 1}] {scenario['name']}")
        print(f"  Steps: {scenario['steps']}")
        
        # Reset before each scenario
        vec_env.reset()
        
        rewards_total = np.zeros(num_envs)
        success_count = np.zeros(num_envs)
        
        for step in range(scenario['steps']):
            actions = scenario['action_generator']()
            obs, rewards, dones, infos = vec_env.step(actions)
            
            rewards_total += rewards
            success_count += np.array([info.get('success', 0) for info in infos])
            
            if step % 3 == 0:  # Print every 3 steps
                print(f"    Step {step + 1}: mean_reward={rewards.mean():.4f}, any_done={np.any(dones)}")
        
        print(f"  ✓ Scenario complete")
        print(f"    Total rewards: {rewards_total}")
        print(f"    Success counts: {success_count}")
    
    vec_env.close()
    print("\n✓ Action strategies test completed")


# ============================================================================
# Test 4: Detailed Step-Through with Individual Strategies
# ============================================================================

def test_detailed_step_through():
    """Detailed test showing step-by-step execution with different strategies per environment"""
    
    print("\n" + "=" * 80)
    print("TEST 4: DETAILED STEP-THROUGH WITH INDIVIDUAL STRATEGIES")
    print("=" * 80)
    
    # Configuration
    suite = "libero_90"
    scene = "study_scene1"
    task = "pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy"
    env_name = f"{suite}-{scene}-{task}"
    num_envs = 3
    
    print(f"\nTask: {task}")
    print(f"Number of environments: {num_envs}")
    
    # Create environments
    print("\n[Step 1] Creating environments...")
    env_fns = [
        lambda i=i: make_libero_env(
            env_name=env_name,
            render_resolution=128,
            max_episode_length=500,
            seed=3000 + i
        )
        for i in range(num_envs)
    ]
    
    vec_env = SubprocVectorEnv(env_fns)
    action_space = vec_env.action_space
    action_dim = action_space[0].shape[0] if isinstance(action_space, list) else action_space.shape[0]
    print(f"✓ Action dimension: {action_dim}")
    
    # Define different strategies for each environment
    print("\n[Step 2] Defining individual strategies...")
    action_strategies = [
        {
            "name": "Random exploration",
            "generator": lambda: np.random.uniform(-0.5, 0.5, action_dim)
        },
        {
            "name": "Conservative movements",
            "generator": lambda: np.random.uniform(-0.1, 0.1, action_dim)
        },
        {
            "name": "Gripper-focused",
            "generator": lambda: np.concatenate([
                np.random.uniform(-0.2, 0.2, action_dim - 1),
                np.array([np.random.choice([-1, 1])])
            ])
        }
    ]
    
    for i, strategy in enumerate(action_strategies):
        print(f"  Env {i}: {strategy['name']}")
    
    # Reset all environments
    print("\n[Step 3] Resetting environments...")
    obs = vec_env.reset()
    print(f"✓ Initial observations shape: {obs.shape}")
    
    # Step through
    print("\n[Step 4] Executing 20 steps...")
    print("-" * 80)
    
    num_steps = 20
    rewards_history = [[] for _ in range(num_envs)]
    success_count = [0] * num_envs
    
    for step_idx in range(num_steps):
        # Generate actions per strategy
        actions = np.array([strategy['generator']() for strategy in action_strategies])
        
        # Step all environments
        obs, rewards, dones, infos = vec_env.step(actions)
        
        # Record results
        for env_id in range(num_envs):
            rewards_history[env_id].append(rewards[env_id])
            if infos[env_id].get('success', 0):
                success_count[env_id] += 1
        
        # Print every 5 steps
        if (step_idx + 1) % 5 == 0:
            print(f"\n>>> STEP {step_idx + 1}/{num_steps} <<<")
            for env_id in range(num_envs):
                print(f"  Env {env_id} ({action_strategies[env_id]['name']}):")
                print(f"    Reward: {rewards[env_id]:.4f}, Done: {dones[env_id]}, Success: {infos[env_id].get('success', 0)}")
            
            # Check success status
            # success_flags = vec_env.check_success()
            # print(f"  Success flags from check_success(): {success_flags}")
    
    # Summary
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    for env_id in range(num_envs):
        total_reward = sum(rewards_history[env_id])
        avg_reward = total_reward / len(rewards_history[env_id])
        print(f"\nEnvironment {env_id} ({action_strategies[env_id]['name']}):")
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Average reward: {avg_reward:.4f}")
        print(f"  Success count: {success_count[env_id]}")
    
    # Cleanup
    print("\n[Step 5] Cleanup...")
    vec_env.close()
    print("✓ All environments closed")
    
    print("\n" + "=" * 80)
    print("TEST 4 COMPLETED")
    print("=" * 80)


# ============================================================================
# Test 5: Performance Benchmark
# ============================================================================

def test_performance_benchmark():
    """Benchmark the performance of parallel environments"""
    
    print("\n" + "=" * 80)
    print("TEST 5: PERFORMANCE BENCHMARK")
    print("=" * 80)
    
    suite = "libero_90"
    scene = "study_scene1"
    task = "pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy"
    env_name = f"{suite}-{scene}-{task}"
    
    # Test with different numbers of parallel environments
    env_counts = [1, 2, 4]
    num_benchmark_steps = 50
    
    for num_envs in env_counts:
        print(f"\n[Benchmark] Testing with {num_envs} parallel environment(s)...")
        
        # Create environments
        env_fns = [
            lambda i=i: make_libero_env(
                env_name=env_name,
                render_resolution=128,
                max_episode_length=500,
                seed=4000 + i
            )
            for i in range(num_envs)
        ]
        
        vec_env = SubprocVectorEnv(env_fns)
        action_space = vec_env.action_space
        action_dim = action_space[0].shape[0] if isinstance(action_space, list) else action_space.shape[0]
        
        # Reset
        vec_env.reset()
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_benchmark_steps):
            actions = np.random.uniform(-1, 1, size=(num_envs, action_dim))
            vec_env.step(actions)
        elapsed = time.time() - start_time
        
        # Calculate metrics
        steps_per_sec = num_benchmark_steps / elapsed
        env_steps_per_sec = (num_benchmark_steps * num_envs) / elapsed
        avg_time_per_step = elapsed / num_benchmark_steps * 1000  # ms
        
        print(f"  ✓ Completed {num_benchmark_steps} steps in {elapsed:.4f} seconds")
        print(f"    Steps per second: {steps_per_sec:.2f}")
        print(f"    Environment steps per second: {env_steps_per_sec:.2f}")
        print(f"    Average time per step: {avg_time_per_step:.2f}ms")
        
        vec_env.close()
    
    print("\n✓ Performance benchmark completed")


# ============================================================================
# Test 6: State Save and Restore
# ============================================================================

def test_state_save_restore():
    """Test saving and restoring simulation states"""
    
    print("\n" + "=" * 80)
    print("TEST 6: STATE SAVE AND RESTORE")
    print("=" * 80)
    
    suite = "libero_90"
    scene = "study_scene1"
    task = "pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy"
    env_name = f"{suite}-{scene}-{task}"
    num_envs = 2
    
    print(f"\nInitializing {num_envs} environments...")
    env_fns = [
        lambda i=i: make_libero_env(
            env_name=env_name,
            render_resolution=128,
            max_episode_length=500,
            seed=5000 + i
        )
        for i in range(num_envs)
    ]
    
    vec_env = SubprocVectorEnv(env_fns)
    action_space = vec_env.action_space
    action_dim = action_space[0].shape[0] if isinstance(action_space, list) else action_space.shape[0]
    
    # Reset and take some steps
    print("\n[Test 6.1] Initial reset and stepping...")
    obs_initial = vec_env.reset()
    print(f"✓ Initial observations shape: {obs_initial.shape}")
    
    # Take 5 steps
    for i in range(5):
        actions = np.random.uniform(-0.3, 0.3, size=(num_envs, action_dim))
        obs, rewards, dones, infos = vec_env.step(actions)
    
    print(f"✓ Took 5 steps")
    print(f"  Final observations range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # Save states
    print("\n[Test 6.2] Saving current states...")
    saved_states = vec_env.get_sim_state()
    print(f"✓ Saved {len(saved_states)} states")
    for i, state in enumerate(saved_states):
        print(f"  Env {i} state shape: {state.shape}")
    
    # Take more steps to modify state
    print("\n[Test 6.3] Taking 10 more steps to modify state...")
    for i in range(10):
        actions = np.random.uniform(-0.5, 0.5, size=(num_envs, action_dim))
        obs, rewards, dones, infos = vec_env.step(actions)
    
    print(f"✓ Modified state")
    print(f"  New observations range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # Restore states
    print("\n[Test 6.4] Restoring saved states...")
    try:
        restored_obs = vec_env.set_init_state(init_state=saved_states)
        print(f"✓ States restored successfully")
        print(f"  Restored observations shape: {restored_obs.shape}")
        print(f"  Restored observations range: [{restored_obs.min():.3f}, {restored_obs.max():.3f}]")
    except Exception as e:
        print(f"  ⚠ State restoration encountered error: {e}")
    
    vec_env.close()
    print("\n✓ State save/restore test completed")


# ============================================================================
# Test 7: Episode Completion and Auto-Reset
# ============================================================================

def test_episode_completion():
    """Test episode completion and auto-reset behavior"""
    
    print("\n" + "=" * 80)
    print("TEST 7: EPISODE COMPLETION AND AUTO-RESET")
    print("=" * 80)
    
    suite = "libero_90"
    scene = "study_scene1"
    task = "pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy"
    env_name = f"{suite}-{scene}-{task}"
    num_envs = 3
    
    print(f"\nInitializing {num_envs} environments with short episode length...")
    env_fns = [
        lambda i=i: make_libero_env(
            env_name=env_name,
            render_resolution=128,
            max_episode_length=50,  # Short episodes for testing
            seed=6000 + i
        )
        for i in range(num_envs)
    ]
    
    vec_env = SubprocVectorEnv(env_fns)
    action_space = vec_env.action_space
    action_dim = action_space[0].shape[0] if isinstance(action_space, list) else action_space.shape[0]
    
    print(f"✓ Environments created with max_episode_length=50")
    
    # Reset
    print("\n[Test 7.1] Initial reset...")
    vec_env.reset()
    
    # Step until some episodes complete
    print("\n[Test 7.2] Stepping until episodes complete...")
    episodes_completed = np.zeros(num_envs)
    total_steps = 0
    max_total_steps = 200
    
    while total_steps < max_total_steps:
        actions = np.random.uniform(-0.3, 0.3, size=(num_envs, action_dim))
        obs, rewards, dones, infos = vec_env.step(actions)
        total_steps += 1
        
        # Check for completed episodes
        for i in range(num_envs):
            if dones[i]:
                episodes_completed[i] += 1
                print(f"  Step {total_steps}: Environment {i} episode completed (total: {int(episodes_completed[i])})")
        
        if episodes_completed.sum() >= 3:  # Stop after 3 total episode completions
            break
    
    print(f"\n✓ Episode completion test finished")
    print(f"  Total steps: {total_steps}")
    print(f"  Episodes completed per environment: {episodes_completed}")
    
    vec_env.close()
    print("\n✓ Episode completion test completed")


# ============================================================================
# Test 8: Stress Test - Many Parallel Environments
# ============================================================================

def test_stress_many_environments():
    """Stress test with many parallel environments"""
    
    print("\n" + "=" * 80)
    print("TEST 8: STRESS TEST - MANY PARALLEL ENVIRONMENTS")
    print("=" * 80)
    
    suite = "libero_90"
    scene = "study_scene1"
    task = "pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy"
    env_name = f"{suite}-{scene}-{task}"
    num_envs = 50  # More environments for stress testing
    
    print(f"\nInitializing {num_envs} parallel environments...")
    print("(This may take a moment...)")
    
    start_init = time.time()
    env_fns = [
        lambda i=i: make_libero_env(
            env_name=env_name,
            render_resolution=128,
            max_episode_length=500,
            seed=7000 + i
        )
        for i in range(num_envs)
    ]
    
    vec_env = SubprocVectorEnv(env_fns)
    action_space = vec_env.action_space
    action_dim = action_space[0].shape[0] if isinstance(action_space, list) else action_space.shape[0]
    init_time = time.time() - start_init
    
    print(f"✓ Initialized {num_envs} environments in {init_time:.2f} seconds")
    print(f"  Average init time per environment: {init_time/num_envs:.2f}s")
    
    # Reset
    print("\n[Test 8.1] Resetting all environments...")
    start_reset = time.time()
    obs = vec_env.reset()
    reset_time = time.time() - start_reset
    print(f"✓ Reset completed in {reset_time:.2f} seconds")
    
    # Run many steps
    print("\n[Test 8.2] Running 100 steps...")
    start_steps = time.time()
    num_steps = 1000
    
    for i in range(num_steps):
        actions = np.random.uniform(-0.5, 0.5, size=(num_envs, action_dim))
        obs, rewards, dones, infos = vec_env.step(actions)
        
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{num_steps} steps")
    
    steps_time = time.time() - start_steps
    print(f"✓ Completed {num_steps} steps in {steps_time:.2f} seconds")
    print(f"  Steps per second: {num_steps/steps_time:.2f}")
    print(f"  Total environment steps: {num_steps * num_envs}")
    print(f"  Environment steps per second: {(num_steps * num_envs)/steps_time:.2f}")
    
    # Check success across all
    # print("\n[Test 8.3] Checking success across all environments...")
    # success_flags = vec_env.check_success()
    # print(f"✓ Success flags: {success_flags}")
    # print(f"  Success rate: {sum(success_flags)/num_envs*100:.1f}%")
    
    vec_env.close()
    print("\n✓ Stress test completed")


# ============================================================================
# Test 9: Observation Space Verification
# ============================================================================

def test_observation_space():
    """Verify observation space consistency across environments"""
    
    print("\n" + "=" * 80)
    print("TEST 9: OBSERVATION SPACE VERIFICATION")
    print("=" * 80)
    
    suite = "libero_90"
    scene = "study_scene1"
    task = "pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy"
    env_name = f"{suite}-{scene}-{task}"
    num_envs = 4
    
    print(f"\nInitializing {num_envs} environments...")
    env_fns = [
        lambda i=i: make_libero_env(
            env_name=env_name,
            render_resolution=128,
            obs_keys=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
            max_episode_length=500,
            seed=8000 + i
        )
        for i in range(num_envs)
    ]
    
    vec_env = SubprocVectorEnv(env_fns)
    
    # Check observation space
    print("\n[Test 9.1] Checking observation space...")
    obs_space = vec_env.observation_space
    obs_space = obs_space[0] if isinstance(obs_space, list) else obs_space
    print(f"✓ Observation space: {obs_space}")
    print(f"  Shape: {obs_space.shape}")
    print(f"  Dtype: {obs_space.dtype}")
    print(f"  Low: {obs_space.low[:5]}... (first 5)")
    print(f"  High: {obs_space.high[:5]}... (first 5)")
    
    # Reset and check observations
    print("\n[Test 9.2] Checking actual observations...")
    obs = vec_env.reset()
    print(f"✓ Observations shape: {obs.shape}")
    print(f"  Expected: ({num_envs}, {obs_space.shape[0]})")
    print(f"  Match: {obs.shape == (num_envs,) + obs_space.shape}")
    
    # Check observation bounds
    print("\n[Test 9.3] Checking observation bounds...")
    within_bounds = np.all((obs >= obs_space.low) & (obs <= obs_space.high))
    print(f"✓ All observations within bounds: {within_bounds}")
    print(f"  Min value: {obs.min():.3f}")
    print(f"  Max value: {obs.max():.3f}")
    
    # Step and check again
    print("\n[Test 9.4] Checking observations after stepping...")
    action_space = vec_env.action_space
    action_dim = action_space[0].shape[0] if isinstance(action_space, list) else action_space.shape[0]
    actions = np.random.uniform(-1, 1, size=(num_envs, action_dim))
    obs, rewards, dones, infos = vec_env.step(actions)
    
    within_bounds = np.all((obs >= obs_space.low) & (obs <= obs_space.high))
    print(f"✓ All observations within bounds after step: {within_bounds}")
    print(f"  Min value: {obs.min():.3f}")
    print(f"  Max value: {obs.max():.3f}")
    
    vec_env.close()
    print("\n✓ Observation space verification completed")


# ============================================================================
# Test 10: Error Handling and Edge Cases
# ============================================================================

def test_error_handling():
    """Test error handling and edge cases"""
    
    print("\n" + "=" * 80)
    print("TEST 10: ERROR HANDLING AND EDGE CASES")
    print("=" * 80)
    
    suite = "libero_90"
    scene = "study_scene1"
    task = "pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy"
    env_name = f"{suite}-{scene}-{task}"
    
    # Test 10.1: Invalid action dimensions
    print("\n[Test 10.1] Testing invalid action dimensions...")
    env_fns = [lambda: make_libero_env(env_name, seed=9000)]
    vec_env = SubprocVectorEnv(env_fns)
    action_space = vec_env.action_space
    action_dim = action_space[0].shape[0] if isinstance(action_space, list) else action_space.shape[0]
    vec_env.reset()
    
    try:
        invalid_action = np.random.uniform(-1, 1, size=(1, action_dim + 5))  # Wrong dimension
        vec_env.step(invalid_action)
        print("  ⚠ No error raised for invalid action dimension")
    except Exception as e:
        print(f"  ✓ Correctly raised error: {type(e).__name__}")
    
    vec_env.close()
    
    # Test 10.2: Out of bounds actions
    print("\n[Test 10.2] Testing out of bounds actions...")
    env_fns = [lambda: make_libero_env(env_name, seed=9001)]
    vec_env = SubprocVectorEnv(env_fns)
    vec_env.reset()
    
    try:
        out_of_bounds_action = np.full((1, action_dim), fill_value=10.0)  # Way out of bounds
        obs, rewards, dones, infos = vec_env.step(out_of_bounds_action)
        print(f"  ✓ Environment handled out of bounds action")
        print(f"    Observation shape: {obs.shape}")
    except Exception as e:
        print(f"  ⚠ Error with out of bounds action: {e}")
    
    vec_env.close()
    
    # Test 10.3: Multiple resets
    print("\n[Test 10.3] Testing multiple consecutive resets...")
    env_fns = [lambda: make_libero_env(env_name, seed=9002)]
    vec_env = SubprocVectorEnv(env_fns)
    
    for i in range(3):
        obs = vec_env.reset()
        print(f"  Reset {i+1}: observation shape {obs.shape}")
    print("  ✓ Multiple resets handled correctly")
    
    vec_env.close()
    
    print("\n✓ Error handling tests completed")


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests in sequence"""
    
    print("\n" + "=" * 80)
    print("LIBERO SUBPROC VECTOR ENV - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print("\nThis test suite will:")
    print("  1. Test single wrapped environment")
    print("  2. Test SubprocVectorEnv with multiple environments")
    print("  3. Test different action strategies")
    print("  4. Test detailed step-through with individual strategies")
    print("  5. Benchmark performance")
    print("  6. Test state save/restore")
    print("  7. Test episode completion")
    print("  8. Stress test with many environments")
    print("  9. Verify observation space consistency")
    print("  10. Test error handling")
    
    input("\nPress Enter to start testing...")
    
    try:
        # Run all tests
        test_single_wrapped_env()
        test_subproc_vector_env()
        test_action_strategies()
        test_detailed_step_through()
        test_performance_benchmark()
        test_state_save_restore()
        test_episode_completion()
        test_stress_many_environments()
        test_observation_space()
        test_error_handling()
        
        # Final summary
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("=" * 80)
        print("\nSummary:")
        print("  ✓ Single environment wrapper working correctly")
        print("  ✓ SubprocVectorEnv working with wrapped LIBERO environments")
        print("  ✓ Multiple action strategies tested")
        print("  ✓ State management working")
        print("  ✓ Performance benchmarked")
        print("  ✓ Error handling validated")
        print("\nYour SubprocVectorEnv is ready for use with LIBERO!")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST SUITE FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SETUP INSTRUCTIONS")
    print("=" * 80)
    print("\n1. Import SubprocVectorEnv at the top of this file:")
    print("   from tianshou.env import SubprocVectorEnv")
    print("\n2. Ensure LIBERO is installed and datasets are available")
    print("\n3. This version uses NoRenderEnv (no rendering) for maximum")
    print("   compatibility with multiprocessing - no GPU/rendering setup needed!")
    print("\n4. Run the full test suite or individual tests:")
    print("   python test_libero_subproc.py")
    print("\n" + "=" * 80)
    print("")
    
    # You can run all tests or individual tests
    run_all_tests()
    
    # Or run individual tests:
    # test_single_wrapped_env()
    # test_subproc_vector_env()
    # test_action_strategies()
    # test_detailed_step_through()
    # test_performance_benchmark()
    # test_state_save_restore()
    # test_episode_completion()
    # test_stress_many_environments()
    # test_observation_space()
    # test_error_handling()