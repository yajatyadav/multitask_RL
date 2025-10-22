from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset
import time

suite = "libero_90"
scene = "study_scene1"
task = "pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy"
env_name = f"{suite}-{scene}-{task}"
keys_to_load = ['obs/ee_pos', 'obs/ee_ori', 'obs/gripper_states', 'states'] # 'states' is a vector of eef_state + flattened mujoco simulator state; right now we are keeping the timestep entry in the simullator-state vector!

env, eval_env, train_dataset, val_dataset = make_env_and_datasets(env_name, keys_to_load=keys_to_load)

# handle dataset
def process_train_dataset(ds):
    """
    Process the train dataset to 
        - handle dataset proportion
        - handle sparse reward
        - convert to action chunked dataset
    """

    ds = Dataset.create(**ds)
    return ds

train_dataset = process_train_dataset(train_dataset)
example_batch = train_dataset.sample(())

start_time = time.time()
for i in range(10_000):
    batch = train_dataset.sample_sequence(256, sequence_length=5, discount=0.99)
end_time = time.time()
print(f"Time taken to iterate over 10,000 batches: {end_time - start_time} seconds")