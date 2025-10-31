from envs.env_utils import make_env_and_datasets
import time
from utils.datasets_ALT import get_dataset_fast, Prefetcher, Profiler
from tqdm import tqdm
import numpy as np
suite = "libero_90"
scene = "kitchen_scene2"
task = ""
env_name = f"{suite}-{scene}"
keys_to_load = ['agentview_rgb', 'eye_in_hand_rgb', 'proprio', 'states']
# keys_to_load = ['states']




augment_negative_demos = False
num_parallel_envs = 1

ds = get_dataset_fast(env_name, task, augment_negative_demos, keys_to_load)
# prof = Profiler()
# batch, prof = ds.sample_sequence(batch_size=256, sequence_length=5, discount=0.99, profiler=prof, augment_on_device=True)
# print(prof.report())

pf = Prefetcher(ds, batch_size=256, seq_len=5, discount=0.99, max_prefetch=32, augment_on_device=True)
pf.start()
num_steps = 10_000
start_time = time.time()
for step in tqdm(range(num_steps), desc="Iterating over dataset"):
    batch, prof = pf.get()   # blocks if no prefetch available
    # optionally aggregate profiler data
    # push batch to device and train
    # e.g., jax.device_put(batch['full_observations']) or convert to pytorch tensors
    # if step % 100 == 0:
        # print(prof.report())

pf.stop()
end_time = time.time()
print(f"Time taken to iterate over {num_steps} steps: {end_time - start_time} seconds") 
# creation_start_time = time.time()
# env, eval_env, train_dataset, val_dataset = make_env_and_datasets(env_name, task_name=task, keys_to_load=keys_to_load, augment_negative_demos=augment_negative_demos, num_parallel_envs=num_parallel_envs)

# handle dataset
# def process_train_dataset(ds):
#     """
#     Process the train dataset to 
#         - handle dataset proportion
#         - handle sparse reward
#         - convert to action chunked dataset
#     """

#     ds = Dataset.create(**ds)
#     return ds

# train_dataset = process_train_dataset(train_dataset)
# example_batch = train_dataset.sample(())

# creation_end_time = time.time()
# print(f"Time taken to create env and datasets: {creation_end_time - creation_start_time} seconds")

# start_time = time.time()
# for i in tqdm(range(10_000)):
#     batch = train_dataset.sample_sequence(256, sequence_length=5, discount=0.99)
# end_time = time.time()
# print(f"Time taken to iterate over 10,000 batches: {end_time - start_time} seconds")


# # now test the multi-eval

# for i, eval_i in enumerate(eval_env):
#     env = eval_i.get_eval_env()
#     env.reset()
#     dummy_action = np.array([0.0] * 6 + [1.0])
#     for i in tqdm(range(500), desc=f"Evaluating env {i} "):
#         parallel_actions = [dummy_action] * num_parallel_envs
#         obs, reward, done, info = env.step(parallel_actions)