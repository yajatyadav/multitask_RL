import torch
from torch.utils.data import IterableDataset, DataLoader
import jax.numpy as jnp
import numpy as np
from typing import List, Optional, Iterator
import tqdm
import time
import os
import h5py

class LiberoDataset(IterableDataset):
    """Simple HDF5 IterableDataset or PyTorch."""
    
    def __init__(
        self,
        h5_files: List[str],
        dataset_key: str = 'data',
        transform=None,
        shuffle=True,
        buffer_size=50_000,
    ):
        """
        Args:
            h5_files: List of paths to h5 files
            dataset_key: Key for data arrays in h5 files
            transform: Optional transform to apply to data
        """
        super().__init__()
        self.h5_files = h5_files
        self.dataset_key = dataset_key
        self.transform = transform
        self.shuffle = shuffle
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator:
        """Apply shuffle buffer to iterator."""
        buffer = []
        iterator = self.unshuffled_iterator()
        
        for item in self.unshuffled_iterator():
            buffer.append(item)
            if len(buffer) >= self.buffer_size:
                # Shuffle and yield random item from buffer
                idx = np.random.randint(0, len(buffer))
                yield buffer.pop(idx)
    
    
    def unshuffled_iterator(self) -> Iterator:
        files_to_process = self.h5_files
        epoch = 0
        while True:
            epoch += 1
            print(f"Epoch {epoch}")
            current_files = list(files_to_process)
            if self.shuffle:
                np.random.shuffle(current_files)
            for h5_path in current_files:
                with h5py.File(h5_path, 'r') as f:
                    demo_group = f[self.dataset_key]
                    demo_keys = list(demo_group.keys())
                    for demo_key in demo_keys:
                        demo = demo_group[demo_key]
                        
                        ee_states = demo['obs']['ee_states'][()]
                        gripper_states = demo['obs']['gripper_states'][()]
                        joint_states = demo['obs']['joint_states'][()]
                        images = demo['obs']['agentview_rgb'][()]
                        wrist_images = demo['obs']['eye_in_hand_rgb'][()]
                        
                        
                        sim_states = demo['states'][()]
                        actions = demo['actions'][()]
                        rewards = demo['rewards'][()]
                        dones = demo['dones'][()]
                        
                        
                        raw_file_string = os.path.basename(h5_path).split('/')[-1]
                        words = raw_file_string[:-10].split("_")
                        command = ''
                        for w in words:
                            if "SCENE" in w:
                                command = ''
                                continue
                        command = command[:-1]

                        traj_len = actions.shape[0]
                        indices = np.arange(traj_len)
                        if self.shuffle:
                            np.random.shuffle(indices)

                        for i in indices:
                            obs = {
                                'image': images[i][::-1,::],
                                'wrist_image': wrist_images[i][::-1,::],
                                'state': np.concatenate((ee_states[i], gripper_states[i]), axis=-1),
                                'joint_state': joint_states[i],
                                'sim_state': sim_states[i],
                            }
                            if i == (actions.shape[0] - 1):
                                next_obs = obs
                            else:
                                next_obs = {
                                    'image': images[i+1][::-1,::],
                                    'wrist_image': wrist_images[i+1][::-1,::],
                                    'state': np.concatenate((ee_states[i+1], gripper_states[i+1]), axis=-1),
                                    'joint_state': joint_states[i+1],
                                    'sim_state': sim_states[i+1],
                                }
                            transition = {
                                'observations': obs,
                                'next_observations': next_obs,
                                'actions': np.array(actions[i], dtype=np.float32),
                                'discount': 1.0,
                                'rewards': np.where(rewards[i] == 0, -1, 0),
                                'is_first': i == 0,
                                'is_last': i == (actions.shape[0] - 1),
                                'is_terminal': bool(dones[i]),
                                'masks': np.where(dones[i], 0, 1),
                                'language_instruction': command,
                            }
                            if self.transform:
                                transition = self.transform(transition)
                            yield transition


def create_dataloader(
    h5_files: List[str],
    batch_size: int,
    dataset_key: str = 'data',
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    transform=None,
    shuffle: bool = True,
    buffer_size: int = 50_000,
) -> DataLoader:
    """
    Create a DataLoader for HDF5 files using IterableDataset.
    
    Args:
        h5_files: List of paths to h5 files
        batch_size: Batch size
        dataset_key: Key for data arrays in h5 files
        num_workers: Number of worker processes for parallel loading
        pin_memory: Pin memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        transform: Optional transform function
    
    Returns:
        DataLoader instance
    """
    dataset = LiberoDataset(
        h5_files=h5_files,
        dataset_key=dataset_key,
        transform=transform,
        shuffle=shuffle,
        buffer_size=buffer_size,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=collate_fn,
    )
    
    return dataloader


import numpy as np

def collate_fn(list_of_transitions: List[dict]) -> dict:
    """
    Collates a list of transition dictionaries into a single dictionary of NumPy arrays,
    handling nested observation dictionaries.
    """
    if not list_of_transitions:
        return {}
    return list_of_transitions
    
    # Get observation keys once
    obs_keys = list_of_transitions[0]['observations'].keys()
    
    # Pre-allocate lists
    obs_lists = {key: [] for key in obs_keys}
    next_obs_lists = {key: [] for key in obs_keys}
    
    actions, rewards, discount = [], [], []
    is_first, is_last, is_terminal, masks = [], [], [], []
    language_instructions = []
    
    # Single pass through the data
    for t in list_of_transitions:
        for key in obs_keys:
            obs_lists[key].append(t['observations'][key])
            next_obs_lists[key].append(t['next_observations'][key])
        
        actions.append(t['actions'])
        rewards.append(t['rewards'])
        discount.append(t['discount'])
        is_first.append(t['is_first'])
        is_last.append(t['is_last'])
        is_terminal.append(t['is_terminal'])
        masks.append(t['masks'])
        language_instructions.append(t['language_instruction'])
    
    # Stack everything as NumPy arrays
    batch_obs = {key: np.stack(obs_lists[key]) for key in obs_keys}
    batch_next_obs = {key: np.stack(next_obs_lists[key]) for key in obs_keys}
    
    return {
        'observations': batch_obs,
        'next_observations': batch_next_obs,
        'actions': np.stack(actions),
        'rewards': np.stack(rewards),
        'discount': np.stack(discount),
        'is_first': np.stack(is_first),
        'is_last': np.stack(is_last),
        'is_terminal': np.stack(is_terminal),
        'masks': np.stack(masks),
        'language_instruction': language_instructions,
    }
    


def create_libero_dataloader(
    data_root_dir: str,
    suite_name:str,
    batch_size: int,
    single_task_name:Optional[list[str]] = None,
    dataset_key: str = 'data',
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for HDF5 files using IterableDataset.
    """
    suite_path = os.path.join(data_root_dir, suite_name)
    if single_task_name is not None:
        h5_files = [os.path.join(suite_path, task_name) for task_name in single_task_name]
    else:
        h5_files = [os.path.join(suite_path, file_name) for file_name in os.listdir(suite_path)]

    dataloader = create_dataloader(
        h5_files=h5_files,
        batch_size=batch_size,
        dataset_key=dataset_key,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )
    return dataloader



# Example usage
if __name__ == "__main__":
    dataloader = create_libero_dataloader(
        data_root_dir='/raid/users/yajatyadav/datasets/raw_libero',
        suite_name='LIBERO_90',
        batch_size=256,
        single_task_name=['KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_demo.hdf5'],
    )
    # List of h5 files
    # root_dir = '/raid/users/yajatyadav/datasets/raw_libero/LIBERO_90'
    # file_names = ['KITCHEN_SCENE1_put_the_black_bowl_on_the_plate_demo.hdf5']
    # h5_files = [os.path.join(root_dir, file_name) for file_name in file_names]
    
    # # Create dataloader
    # dataloader = create_dataloader(
    #     h5_files=h5_files,
    #     batch_size=32,
    #     dataset_key='data',
    #     num_workers=0,
    #     pin_memory=True,
    #     prefetch_factor=None,
    #     persistent_workers=True,
    #     shuffle=True,
    #     buffer_size=20_000,
    # )

    # # time 1000 batches
    start_time = time.time()
    for i, batch in tqdm.tqdm(enumerate(dataloader)):
        if i == 100_000:
            break
        continue
    end_time = time.time()
    print(f"Time taken to iterate over 100_000 batches: {end_time - start_time} seconds")