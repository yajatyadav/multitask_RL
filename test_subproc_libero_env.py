import numpy as np
import os
import pathlib
import time
from typing import List, Dict, Any
import sys
sys.path.insert(0, '/home/yajatyadav/multitask_reinforcement_learning/multitask_RL/libero') # TODO(YY): hack for now, set up submoduling later..
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from libero.libero.utils import get_libero_path



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


def make_env(env_name, render_resolution=128, keys_to_load=[], seed=0):
    """
    Create a LIBERO environment
    NOTE: should get_dataset() first, so that the metadata is downloaded before creating the environment
    """
    dataset_path = _check_dataset_exists(env_name)
    
    if env_name.startswith("libero_90") or env_name.startswith("libero_10"):
        suite_str, scene_str, task_str = env_name.split("-")
    else:
        suite_str, task_str = env_name.split("-")
        scene_str = ''

    max_episode_length = _get_max_episode_length(env_name)

    task_suite = benchmark.get_benchmark_dict()[suite_str]()
    num_tasks_in_suite = task_suite.n_tasks

    # obtain task object by just iterating over suite
    desired_task_name = f"{scene_str.upper()}_{task_str}"
    for task_id in range(num_tasks_in_suite):
        task = task_suite.get_task(task_id)
        if task.name == desired_task_name:
            break
    
    # use task to build OffScreenRenderEnv object
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file, 
        "camera_heights": render_resolution, 
        "camera_widths": render_resolution
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)    
    return env


def test_subproc_vector_env_with_libero():
    """Comprehensive test suite for SubprocVectorEnv with LIBERO environments"""
    
    print("=" * 80)
    print("TESTING SubprocVectorEnv with LIBERO Environments")
    print("=" * 80)
    
    # Configuration
    suite = "libero_90"
    scene = "study_scene1"
    task = "pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy"
    env_name = f"{suite}-{scene}-{task}"
    keys_to_load = ['obs/ee_pos', 'obs/ee_ori', 'obs/gripper_states', 'states']
    render_resolution = 128
    num_envs = 4
    
    print(f"\nEnvironment Configuration:")
    print(f"  Suite: {suite}")
    print(f"  Scene: {scene}")
    print(f"  Task: {task}")
    print(f"  Number of parallel environments: {num_envs}")
    
    # Test 1: Initialize multiple environments
    print("\n[Test 1] Creating environment factories...")
    env_fns = [
        lambda i=i: make_env(
            env_name, 
            render_resolution=render_resolution,
            keys_to_load=keys_to_load,
            seed=1000 + i  # Different seed for each environment
        ) 
        for i in range(num_envs)
    ]
    print(f"✓ Created {num_envs} environment factories with different seeds")
    
    # Initialize SubprocVectorEnv
    print("\n[Test 2] Initializing SubprocVectorEnv...")
    vec_env = SubprocVectorEnv(env_fns)
    print(f"✓ SubprocVectorEnv initialized")
    print(f"  Action space: {vec_env.action_space}")
    print(f"  Observation space: {vec_env.observation_space}")
    
    # Test 3: Reset all environments
    print("\n[Test 3] Resetting all environments...")
    obs = vec_env.reset()
    print(f"✓ Reset complete")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation dtype: {obs.dtype}")
    print(f"  Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # Test 4: Get action space dimensions
    print("\n[Test 4] Analyzing action space...")
    action_space = vec_env.action_space
    print(f"✓ Action space type: {type(action_space)}")
    print(f"  Action shape: {action_space.shape}")
    print(f"  Action range: [{action_space.low[0]:.3f}, {action_space.high[0]:.3f}]")
    action_dim = action_space.shape[0]
    
    # Test 5: Step through with different action strategies
    print("\n[Test 5] Stepping through environments with different actions...")
    
    test_scenarios = [
        {
            "name": "Random actions (exploration)",
            "action_generator": lambda: np.random.uniform(-1, 1, size=(num_envs, action_dim)),
            "steps": 10
        },
        {
            "name": "Zero actions (stationary)",
            "action_generator": lambda: np.zeros((num_envs, action_dim)),
            "steps": 5
        },
        {
            "name": "Gripper open/close only",
            "action_generator": lambda: np.concatenate([
                np.zeros((num_envs, action_dim - 1)),
                np.random.choice([-1, 1], size=(num_envs, 1))
            ], axis=1),
            "steps": 8
        },
        {
            "name": "Small perturbations",
            "action_generator": lambda: np.random.uniform(-0.1, 0.1, size=(num_envs, action_dim)),
            "steps": 15
        }
    ]
    
    for scenario_idx, scenario in enumerate(test_scenarios):
        print(f"\n  Scenario {scenario_idx + 1}: {scenario['name']}")
        print(f"  Steps: {scenario['steps']}")
        
        # Reset before each scenario
        obs = vec_env.reset()
        
        for step in range(scenario['steps']):
            actions = scenario['action_generator']()
            obs, rewards, dones, infos = vec_env.step(actions)
            
            print(f"    Step {step + 1}:")
            print(f"      Action range: [{actions.min():.3f}, {actions.max():.3f}]")
            print(f"      Rewards: {rewards}")
            print(f"      Dones: {dones}")
            print(f"      Any episode ended: {np.any(dones)}")
            
            # Auto-reset done environments if needed
            if np.any(dones):
                print(f"      Episodes {np.where(dones)[0].tolist()} ended, auto-resetting...")
    
    # Test 6: Check success across environments
    print("\n[Test 6] Checking success status...")
    success_flags = vec_env.check_success()
    print(f"✓ Success flags: {success_flags}")
    print(f"  Number of successful environments: {sum(success_flags)}")
    
    # Test 7: Get simulation states
    print("\n[Test 7] Retrieving simulation states...")
    sim_states = vec_env.get_sim_state()
    print(f"✓ Retrieved {len(sim_states)} simulation states")
    for i, state in enumerate(sim_states[:2]):  # Show first 2
        print(f"  Environment {i} state type: {type(state)}")
        if isinstance(state, dict):
            print(f"    Keys: {list(state.keys())}")
    
    # Test 8: Test segmentation processing
    print("\n[Test 8] Testing segmentation of interest...")
    dummy_seg_images = [np.random.randint(0, 10, size=(render_resolution, render_resolution)) 
                        for _ in range(num_envs)]
    processed = vec_env.get_segmentation_of_interest(dummy_seg_images)
    print(f"✓ Processed {len(processed)} segmentation images")
    if len(processed) > 0:
        print(f"  Processed image shape: {processed[0].shape}")
    
    # Test 9: Set custom initial states (if supported)
    print("\n[Test 9] Testing custom initial state setting...")
    custom_init_states = [i * 0.1 for i in range(num_envs)]
    try:
        obs = vec_env.set_init_state(init_state=custom_init_states)
        print(f"✓ Set custom initial states successfully")
        print(f"  Observation shape after init: {obs.shape}")
    except (AttributeError, NotImplementedError) as e:
        print(f"  ⚠ set_init_state may not be supported: {e}")
    
    # Test 10: Performance benchmark
    print("\n[Test 10] Performance benchmark - 50 steps...")
    num_benchmark_steps = 50
    vec_env.reset()
    start_time = time.time()
    for _ in range(num_benchmark_steps):
        actions = np.random.uniform(-1, 1, size=(num_envs, action_dim))
        vec_env.step(actions)
    elapsed = time.time() - start_time
    print(f"✓ Completed {num_benchmark_steps} steps in {elapsed:.4f} seconds")
    print(f"  Average time per step: {elapsed/num_benchmark_steps*1000:.2f}ms")
    print(f"  Steps per second: {num_benchmark_steps/elapsed:.2f}")
    print(f"  Environment steps per second: {num_benchmark_steps * num_envs / elapsed:.2f}")
    
    # Test 11: Cleanup
    print("\n[Test 11] Cleanup...")
    vec_env.close()
    print("✓ All environments closed")
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)


def detailed_libero_step_through_test():
    """Detailed test showing step-by-step execution with different action sequences"""
    
    print("\n" + "=" * 80)
    print("DETAILED STEP-THROUGH TEST WITH LIBERO")
    print("=" * 80)
    
    # Configuration
    suite = "libero_90"
    scene = "study_scene1"
    task = "pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy"
    env_name = f"{suite}-{scene}-{task}"
    keys_to_load = ['obs/ee_pos', 'obs/ee_ori', 'obs/gripper_states', 'states']
    render_resolution = 128
    num_envs = 3
    
    print(f"\nTask: {task}")
    print(f"Number of parallel environments: {num_envs}")
    
    # Create environments
    print("\n[Step 1] Creating environment factories...")
    env_fns = [
        lambda i=i: make_env(
            env_name,
            render_resolution=render_resolution,
            keys_to_load=keys_to_load,
            seed=2000 + i
        )
        for i in range(num_envs)
    ]
    
    # Get action dimension
    temp_env = make_env(env_name, render_resolution=render_resolution, keys_to_load=keys_to_load, seed=0)
    action_dim = temp_env.action_space.shape[0]
    temp_env.close()
    print(f"✓ Action dimension: {action_dim}")
    
    # Initialize vectorized environment
    print("\n[Step 2] Initializing SubprocVectorEnv...")
    vec_env = SubprocVectorEnv(env_fns)
    print("✓ Initialized")
    
    # Define different action strategies for each environment
    print("\n[Step 3] Defining action strategies...")
    action_strategies = [
        {
            "name": "Random exploration",
            "generator": lambda: np.random.uniform(-0.5, 0.5, action_dim)
        },
        {
            "name": "Gripper-focused",
            "generator": lambda: np.concatenate([
                np.random.uniform(-0.2, 0.2, action_dim - 1),
                np.array([np.random.choice([-1, 1])])
            ])
        },
        {
            "name": "Conservative movements",
            "generator": lambda: np.random.uniform(-0.1, 0.1, action_dim)
        }
    ]
    
    for i, strategy in enumerate(action_strategies):
        print(f"  Env {i}: {strategy['name']}")
    
    # Reset all environments
    print("\n[Step 4] Resetting all environments...")
    obs = vec_env.reset()
    print(f"✓ Initial observations shape: {obs.shape}")
    
    # Step through environments
    print("\n[Step 5] Stepping through environments...")
    print("-" * 80)
    
    num_steps = 20
    rewards_history = [[] for _ in range(num_envs)]
    success_count = [0] * num_envs
    
    for step_idx in range(num_steps):
        print(f"\n>>> STEP {step_idx + 1}/{num_steps} <<<")
        
        # Generate actions for each environment based on their strategy
        actions = np.array([strategy['generator']() for strategy in action_strategies])
        print(f"Actions generated: shape={actions.shape}")
        
        # Step all environments
        obs, rewards, dones, infos = vec_env.step(actions)
        
        # Display results
        for env_id in range(num_envs):
            print(f"  Env {env_id} ({action_strategies[env_id]['name']}):")
            print(f"    Action: {actions[env_id][:3]}... (showing first 3 dims)")
            print(f"    Reward: {rewards[env_id]:.4f}")
            print(f"    Done: {dones[env_id]}")
            rewards_history[env_id].append(rewards[env_id])
        
        # Check success periodically
        if (step_idx + 1) % 5 == 0:
            print(f"\n  Checking success at step {step_idx + 1}...")
            success_flags = vec_env.check_success()
            for env_id, success in enumerate(success_flags):
                if success:
                    success_count[env_id] += 1
                    print(f"    Env {env_id}: SUCCESS! ✓")
        
        # Handle episode termination
        if np.any(dones):
            print(f"\n  Episodes ended: {np.where(dones)[0].tolist()}")
    
    # Summary
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    for env_id in range(num_envs):
        total_reward = sum(rewards_history[env_id])
        avg_reward = total_reward / len(rewards_history[env_id]) if rewards_history[env_id] else 0
        print(f"\nEnvironment {env_id} ({action_strategies[env_id]['name']}):")
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Average reward per step: {avg_reward:.4f}")
        print(f"  Success count: {success_count[env_id]}")
    
    # Cleanup
    print("\n[Step 6] Cleaning up...")
    vec_env.close()
    print("✓ All environments closed")
    
    print("\n" + "=" * 80)
    print("DETAILED TEST COMPLETED")
    print("=" * 80)


def quick_single_env_test():
    """Quick test with a single LIBERO environment to verify setup"""
    
    print("\n" + "=" * 80)
    print("QUICK SINGLE ENVIRONMENT TEST")
    print("=" * 80)
    
    suite = "libero_90"
    scene = "study_scene1"
    task = "pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy"
    env_name = f"{suite}-{scene}-{task}"
    keys_to_load = ['obs/ee_pos', 'obs/ee_ori', 'obs/gripper_states', 'states']
    
    print(f"\nCreating single environment: {env_name}")
    env = make_env(env_name, render_resolution=128, keys_to_load=keys_to_load, seed=42)
    
    print(f"✓ Environment created successfully")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    
    # Test reset
    obs = env.reset()
    print(f"\n✓ Reset successful")
    print(f"  Observation type: {type(obs)}")
    if isinstance(obs, dict):
        print(f"  Observation keys: {obs.keys()}")
    else:
        print(f"  Observation shape: {obs.shape}")
    
    # Test a few steps
    print("\n✓ Testing 5 steps...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"  Step {i+1}: reward={reward:.4f}, done={done}")
    
    env.close()
    print("\n✓ Single environment test completed")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("LIBERO ENVIRONMENT TESTING SUITE")
    print("=" * 80)
    
    # Run quick single environment test first
    quick_single_env_test()
    
    # Run comprehensive test suite
    test_subproc_vector_env_with_libero()
    
    # Run detailed step-through test
    detailed_libero_step_through_test()
    
    print("\n✅ All testing completed successfully!")
    print("\n" + "=" * 80)
    print("SETUP INSTRUCTIONS:")
    print("=" * 80)
    print("1. Import SubprocVectorEnv at the top of this file")
    print("2. Implement _check_dataset_exists() and _get_max_episode_length()")
    print("3. Ensure LIBERO datasets are downloaded")
    print("4. Run: python test_subproc_vector_env.py")
    print("=" * 80)