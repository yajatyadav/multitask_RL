import os
import sys

# Add the libero package to the Python path
repo_root_dir = os.getenv("MULTITASK_RL_REPO_ROOT_DIR", "/home/yajatyadav/multitask_reinforcement_learning/multitask_RL")
sys.path.insert(0, os.path.join(repo_root_dir, "libero"))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

import tqdm

benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_10" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()

# retrieve a specific task
task_id = 0
task = task_suite.get_task(task_id)
task_name = task.name
task_description = task.language
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
      f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

# step over the environment
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128
}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
env.reset()
init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
init_state_id = 0
env.set_init_state(init_states[init_state_id])

dummy_action = [0.] * 7
for step in tqdm.tqdm(range(900)):
    obs, reward, done, info = env.step(dummy_action)
env.close()