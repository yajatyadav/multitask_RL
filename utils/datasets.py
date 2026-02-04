from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Dataset class."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None  # Number of frames to stack; set outside the class.
        self.p_aug = None  # Image augmentation probability; set outside the class.
        self.return_next_actions = False  # Whether to additionally return next actions; set outside the class.

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        if self.frame_stack is not None:
            # Stack frames.
            initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
            obs = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]].
            next_obs = []  # Will be [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]].
            for i in reversed(range(self.frame_stack)):
                # Use the initial state if the index is out of bounds.
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
                if i != self.frame_stack - 1:
                    next_obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
            next_obs.append(jax.tree_util.tree_map(lambda arr: arr[idxs], self['next_observations']))

            batch['observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *obs)
            batch['next_observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *next_obs)
        if self.p_aug is not None:
            # Apply random-crop image augmentation.
            if np.random.rand() < self.p_aug:
                self.augment(batch, ['observations', 'next_observations'])
        return batch

    def sample_sequence_at_trajectory_position(self, batch_size, sequence_length, discount, position):
        """Sample sequences from distinct episodes, each starting at the given position.
        
        Args:
            batch_size: Number of distinct trajectories' subsequences to sample.
            sequence_length: Length of each sequence.
            discount: Discount factor for cumulative rewards.
            position: Starting position within each trajectory (0 = start of episode).
        
        Returns:
            Batch dictionary with the same structure as sample_sequence.
        """
        initial_locs = self.initial_locs
        terminal_locs = self.terminal_locs
        
        # Find episodes long enough: need at least (position + sequence_length) transitions
        episode_lengths = terminal_locs - initial_locs + 1
        valid_episodes = np.where(episode_lengths >= position + sequence_length)[0]

        if batch_size == -1:
            batch_size = len(valid_episodes)
        
        if len(valid_episodes) < batch_size:
            raise ValueError(
                f"Only {len(valid_episodes)} episodes have length >= {position + sequence_length}, "
                f"but batch_size={batch_size}"
            )
        selected_episodes = np.random.choice(valid_episodes, size=batch_size, replace=False)
        
        # Compute starting indices: initial_loc + position for each selected episode
        idxs = initial_locs[selected_episodes] + position
        
        # --- Rest is identical to sample_sequence ---
        data = jax.tree_util.tree_map(lambda v: v[idxs], self._dict)

        # Pre-compute all required indices
        all_idxs = idxs[:, None] + np.arange(sequence_length)[None, :]  # (batch_size, sequence_length)
        all_idxs = all_idxs.flatten()
        
        # Batch fetch data - handle both dict and array observations
        def fetch_and_reshape(arr):
            fetched = arr[all_idxs]
            return fetched.reshape(batch_size, sequence_length, *arr.shape[1:])
        
        batch_observations = jax.tree_util.tree_map(fetch_and_reshape, self['observations'])
        batch_next_observations = jax.tree_util.tree_map(fetch_and_reshape, self['next_observations'])
        batch_actions = self['actions'][all_idxs].reshape(batch_size, sequence_length, *self['actions'].shape[1:])
        batch_rewards = self['rewards'][all_idxs].reshape(batch_size, sequence_length, *self['rewards'].shape[1:])
        batch_masks = self['masks'][all_idxs].reshape(batch_size, sequence_length, *self['masks'].shape[1:])
        batch_terminals = self['terminals'][all_idxs].reshape(batch_size, sequence_length, *self['terminals'].shape[1:])
        
        # Calculate next_actions
        next_action_idxs = np.minimum(all_idxs + 1, self.size - 1)
        batch_next_actions = self['actions'][next_action_idxs].reshape(batch_size, sequence_length, *self['actions'].shape[1:])
        
        # Use vectorized operations to calculate cumulative rewards and masks
        rewards = np.zeros((batch_size, sequence_length), dtype=float)
        masks = np.ones((batch_size, sequence_length), dtype=float)
        terminals = np.zeros((batch_size, sequence_length), dtype=float)
        valid = np.ones((batch_size, sequence_length), dtype=float)
        
        # Vectorized calculation
        rewards[:, 0] = batch_rewards[:, 0].squeeze()
        masks[:, 0] = batch_masks[:, 0].squeeze()
        terminals[:, 0] = batch_terminals[:, 0].squeeze()
        
        discount_powers = discount ** np.arange(sequence_length)
        for i in range(1, sequence_length):
            rewards[:, i] = rewards[:, i-1] + batch_rewards[:, i].squeeze() * discount_powers[i]
            masks[:, i] = np.minimum(masks[:, i-1], batch_masks[:, i].squeeze())
            terminals[:, i] = np.maximum(terminals[:, i-1], batch_terminals[:, i].squeeze())
            valid[:, i] = 1.0 - terminals[:, i-1]
        
        # Reorganize observations data format - handle both dict and array
        def transpose_if_visual(arr):
            if len(arr.shape) == 5:  # Visual data: (batch, seq, h, w, c)
                return arr.transpose(0, 2, 3, 1, 4)  # -> (batch, h, w, seq, c)
            else:  # State data: maintain (batch, seq, state_dim)
                return arr
        
        observations = jax.tree_util.tree_map(transpose_if_visual, batch_observations)
        next_observations = jax.tree_util.tree_map(transpose_if_visual, batch_next_observations)
        
        actions = batch_actions  # (batch_size, sequence_length, action_dim)
        next_actions = batch_next_actions  # (batch_size, sequence_length, action_dim)
        
        return dict(
            observations=jax.tree_util.tree_map(lambda arr: arr.copy(), data['observations']),
            full_observations=observations,
            actions=actions,
            masks=masks,
            rewards=rewards,
            terminals=terminals,
            valid=valid,
            next_observations=next_observations,
            next_actions=next_actions,
        )

    
    def sample_sequence(self, batch_size, sequence_length, discount, idxs_to_use=None):
        # TODO(YY): hack so we can use sample_sequence to controllably sample from start/middle/end of trajectories
        if idxs_to_use is None:
            # Build list of valid starting indices that don't cross episode boundaries
            initial_locs = self.initial_locs
            terminal_locs = self.terminal_locs
            
            valid_starts = []
            for ep_start, ep_end in zip(initial_locs, terminal_locs):
                # Valid starts for this episode: can start from ep_start up to ep_end - sequence_length + 1
                max_start = ep_end - sequence_length + 1
                if max_start >= ep_start:
                    valid_starts.append(np.arange(ep_start, max_start + 1))
            
            valid_starts = np.concatenate(valid_starts)
            replace = batch_size > len(valid_starts) # only do replacement if we need more samples than the number of valid starts
            idxs = np.random.choice(valid_starts, size=batch_size, replace=replace)
        else:
            idxs = idxs_to_use
        
        data = jax.tree_util.tree_map(lambda v: v[idxs], self._dict)

        # Pre-compute all required indices
        all_idxs = idxs[:, None] + np.arange(sequence_length)[None, :]  # (batch_size, sequence_length)
        all_idxs = all_idxs.flatten()
        
        # Batch fetch data - handle both dict and array observations
        def fetch_and_reshape(arr):
            fetched = arr[all_idxs]
            return fetched.reshape(batch_size, sequence_length, *arr.shape[1:])
        
        batch_observations = jax.tree_util.tree_map(fetch_and_reshape, self['observations'])
        batch_next_observations = jax.tree_util.tree_map(fetch_and_reshape, self['next_observations'])
        batch_actions = self['actions'][all_idxs].reshape(batch_size, sequence_length, *self['actions'].shape[1:])
        batch_rewards = self['rewards'][all_idxs].reshape(batch_size, sequence_length, *self['rewards'].shape[1:])
        batch_masks = self['masks'][all_idxs].reshape(batch_size, sequence_length, *self['masks'].shape[1:])
        batch_terminals = self['terminals'][all_idxs].reshape(batch_size, sequence_length, *self['terminals'].shape[1:])
        
        # Calculate next_actions
        next_action_idxs = np.minimum(all_idxs + 1, self.size - 1)
        batch_next_actions = self['actions'][next_action_idxs].reshape(batch_size, sequence_length, *self['actions'].shape[1:])
        
        # Use vectorized operations to calculate cumulative rewards and masks
        rewards = np.zeros((batch_size, sequence_length), dtype=float)
        masks = np.ones((batch_size, sequence_length), dtype=float)
        terminals = np.zeros((batch_size, sequence_length), dtype=float)
        valid = np.ones((batch_size, sequence_length), dtype=float)
        
        # Vectorized calculation
        rewards[:, 0] = batch_rewards[:, 0].squeeze()
        masks[:, 0] = batch_masks[:, 0].squeeze()
        terminals[:, 0] = batch_terminals[:, 0].squeeze()
        
        discount_powers = discount ** np.arange(sequence_length)
        for i in range(1, sequence_length):
            rewards[:, i] = rewards[:, i-1] + batch_rewards[:, i].squeeze() * discount_powers[i]
            masks[:, i] = np.minimum(masks[:, i-1], batch_masks[:, i].squeeze())
            terminals[:, i] = np.maximum(terminals[:, i-1], batch_terminals[:, i].squeeze())
            valid[:, i] = 1.0 - terminals[:, i-1]
        
        # Reorganize observations data format - handle both dict and array
        def transpose_if_visual(arr):
            if len(arr.shape) == 5:  # Visual data: (batch, seq, h, w, c)
                return arr.transpose(0, 2, 3, 1, 4)  # -> (batch, h, w, seq, c)
            else:  # State data: maintain (batch, seq, state_dim)
                return arr
        
        observations = jax.tree_util.tree_map(transpose_if_visual, batch_observations)
        next_observations = jax.tree_util.tree_map(transpose_if_visual, batch_next_observations)
        
        actions = batch_actions  # (batch_size, sequence_length, action_dim)
        next_actions = batch_next_actions  # (batch_size, sequence_length, action_dim)
        
        return dict(
            observations=jax.tree_util.tree_map(lambda arr: arr.copy(), data['observations']),
            full_observations=observations,
            actions=actions,
            masks=masks,
            rewards=rewards,
            terminals=terminals,
            valid=valid,
            next_observations=next_observations,
            next_actions=next_actions,
        )
        

    def sample_sequence_old(self, batch_size, sequence_length, discount):
        idxs = np.random.randint(self.size - sequence_length + 1, size=batch_size)
        
        data = {k: v[idxs] for k, v in self.items()}

        # Pre-compute all required indices
        all_idxs = idxs[:, None] + np.arange(sequence_length)[None, :]  # (batch_size, sequence_length)
        all_idxs = all_idxs.flatten()
        
        # Batch fetch data to avoid loops
        batch_observations = self['observations'][all_idxs].reshape(batch_size, sequence_length, *self['observations'].shape[1:])
        batch_next_observations = self['next_observations'][all_idxs].reshape(batch_size, sequence_length, *self['next_observations'].shape[1:])
        batch_actions = self['actions'][all_idxs].reshape(batch_size, sequence_length, *self['actions'].shape[1:])
        batch_rewards = self['rewards'][all_idxs].reshape(batch_size, sequence_length, *self['rewards'].shape[1:])
        batch_masks = self['masks'][all_idxs].reshape(batch_size, sequence_length, *self['masks'].shape[1:])
        batch_terminals = self['terminals'][all_idxs].reshape(batch_size, sequence_length, *self['terminals'].shape[1:])
        
        # Calculate next_actions
        next_action_idxs = np.minimum(all_idxs + 1, self.size - 1)
        batch_next_actions = self['actions'][next_action_idxs].reshape(batch_size, sequence_length, *self['actions'].shape[1:])
        
        # Use vectorized operations to calculate cumulative rewards and masks
        rewards = np.zeros((batch_size, sequence_length), dtype=float)
        masks = np.ones((batch_size, sequence_length), dtype=float)
        terminals = np.zeros((batch_size, sequence_length), dtype=float)
        valid = np.ones((batch_size, sequence_length), dtype=float)
        
        # Vectorized calculation
        rewards[:, 0] = batch_rewards[:, 0].squeeze()
        masks[:, 0] = batch_masks[:, 0].squeeze()
        terminals[:, 0] = batch_terminals[:, 0].squeeze()
        
        discount_powers = discount ** np.arange(sequence_length)
        for i in range(1, sequence_length):
            rewards[:, i] = rewards[:, i-1] + batch_rewards[:, i].squeeze() * discount_powers[i]
            masks[:, i] = np.minimum(masks[:, i-1], batch_masks[:, i].squeeze())
            terminals[:, i] = np.maximum(terminals[:, i-1], batch_terminals[:, i].squeeze())
            valid[:, i] = 1.0 - terminals[:, i-1]
        
        # Reorganize observations data format - maintain the exact same shape as the original function
        if len(batch_observations.shape) == 5:  # Visual data: (batch, seq, h, w, c)
            # Transpose to (batch, h, w, seq, c) format, consistent with the original function
            observations = batch_observations.transpose(0, 2, 3, 1, 4)  # (batch_size, h, w, sequence_length, c)
            next_observations = batch_next_observations.transpose(0, 2, 3, 1, 4)  # (batch_size, h, w, sequence_length, c)
        else:  # State data: maintain (batch, seq, state_dim) shape
            observations = batch_observations  # (batch_size, sequence_length, state_dim)
            next_observations = batch_next_observations  # (batch_size, sequence_length, state_dim)
        
        # Maintain the 3D shape of actions and next_actions, consistent with the original function
        actions = batch_actions  # (batch_size, sequence_length, action_dim)
        next_actions = batch_next_actions  # (batch_size, sequence_length, action_dim)
        
        return dict(
            observations=data['observations'].copy(),
            full_observations=observations,
            actions=actions,
            masks=masks,
            rewards=rewards,
            terminals=terminals,
            valid=valid,
            next_observations=next_observations,
            next_actions=next_actions,
        )

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            # WARNING: This is incorrect at the end of the trajectory. Use with caution.
            result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]
        return result

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0

def add_history(dataset, history_length):

    size = dataset.size
    (terminal_locs,) = np.nonzero(dataset['terminals'] > 0)
    initial_locs = np.concatenate([[0], terminal_locs[:-1] + 1])
    assert terminal_locs[-1] == size - 1

    idxs = np.arange(size)
    initial_state_idxs = initial_locs[np.searchsorted(initial_locs, idxs, side='right') - 1]
    obs_rets = []
    acts_rets = []
    for i in reversed(range(1, history_length)):
        cur_idxs = np.maximum(idxs - i, initial_state_idxs)
        outside = (idxs - i < initial_state_idxs)[..., None]
        obs_rets.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs] * (~outside) + jnp.zeros_like(arr[cur_idxs]) * outside, 
            dataset['observations']))
        acts_rets.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs] * (~outside) + jnp.zeros_like(arr[cur_idxs]) * outside, 
            dataset['actions']))
    observation_history, action_history = jax.tree_util.tree_map(lambda *args: np.stack(args, axis=-2), *obs_rets),\
        jax.tree_util.tree_map(lambda *args: np.stack(args, axis=-2), *acts_rets)

    dataset = Dataset(dataset.copy(dict(
        observation_history=observation_history,
        action_history=action_history)))
    
    return dataset


import numpy as np
import jax.tree_util

import numpy as np
import jax.tree_util


class MultiDatasetWrapper:
    """Wrapper for sampling from multiple Dataset objects with various strategies."""

    def __init__(
        self,
        datasets,
        batch_level_sampling,
        weights=None,
        return_dataset_indices=True,
        shuffle_merged=True,
    ):
        """Initialize the multi-dataset wrapper.

        Args:
            datasets: List of Dataset objects.
            weights: Optional list of weights for each dataset. If None, uses uniform weights.
                     Weights are automatically normalized to sum to 1.
            batch_level_sampling: If True, each batch maintains the exact proportions specified
                                  by weights (e.g., 50/50 split means each batch is half from
                                  each dataset). If False, samples are drawn independently
                                  based on weights (proportions hold in expectation).
            return_dataset_indices: If True, returned batches include a 'dataset_indices' key
                                    indicating which dataset each sample came from.
            shuffle_merged: If True, shuffle samples after merging batches from different
                            datasets. Recommended to avoid dataset clustering in batches.
        """
        assert len(datasets) > 0, "Must provide at least one dataset"
        self.datasets = datasets
        self.num_datasets = len(datasets)
        self.batch_level_sampling = batch_level_sampling
        self.return_dataset_indices = return_dataset_indices
        self.shuffle_merged = shuffle_merged

        # Set up weights (normalized)
        if weights is None:
            self.weights = np.ones(self.num_datasets) / self.num_datasets
        else:
            weights = np.array(weights, dtype=np.float64)
            assert len(weights) == self.num_datasets, "Must provide one weight per dataset"
            assert np.all(weights >= 0), "Weights must be non-negative"
            assert np.sum(weights) > 0, "Weights must sum to a positive value"
            self.weights = weights / np.sum(weights)

        # Compute total size
        self.dataset_sizes = np.array([d.size for d in datasets])
        self.size = np.sum(self.dataset_sizes)

        # Propagate attributes from first dataset (assume all datasets share these)
        self.frame_stack = datasets[0].frame_stack
        self.p_aug = datasets[0].p_aug
        self.return_next_actions = datasets[0].return_next_actions

    def set_weights(self, weights):
        """Update sampling weights."""
        weights = np.array(weights, dtype=np.float64)
        assert len(weights) == self.num_datasets
        self.weights = weights / np.sum(weights)

    def _compute_samples_per_dataset(self, batch_size):
        """Compute how many samples to draw from each dataset for batch-level sampling."""
        fractional_samples = self.weights * batch_size
        samples_per_dataset = np.floor(fractional_samples).astype(int)

        remainder = batch_size - np.sum(samples_per_dataset)
        if remainder > 0:
            fractional_parts = fractional_samples - samples_per_dataset
            
            # Randomly select among datasets, weighted by their fractional parts
            # (handles ties naturally)
            probs = fractional_parts / fractional_parts.sum()
            top_indices = np.random.choice(
                self.num_datasets, size=int(remainder), replace=False, p=probs
            )
            samples_per_dataset[top_indices] += 1

        return samples_per_dataset

    def _get_batch_size(self, batch):
        """Get batch size from a batch dictionary."""
        first_value = next(iter(batch.values()))
        if isinstance(first_value, dict):
            # Nested dict (e.g., observations with multiple keys)
            return len(next(iter(first_value.values())))
        else:
            return len(first_value)

    def _shuffle_batch(self, batch):
        """Shuffle samples within a batch."""
        batch_size = self._get_batch_size(batch)
        perm = np.random.permutation(batch_size)
        return jax.tree_util.tree_map(lambda x: x[perm], batch)

    def _merge_batches(self, batches, dataset_indices_list=None):
        """Merge multiple batch dictionaries into one."""
        if len(batches) == 0:
            raise ValueError("No batches to merge")

        if len(batches) == 1:
            merged = batches[0]
            if self.return_dataset_indices and dataset_indices_list is not None:
                merged['dataset_indices'] = np.concatenate(dataset_indices_list)
            return merged

        # Concatenate all batches
        merged = {}
        for key in batches[0].keys():
            values = [b[key] for b in batches]
            merged[key] = jax.tree_util.tree_map(
                lambda *arrs: np.concatenate(arrs, axis=0),
                *values
            )

        if self.return_dataset_indices and dataset_indices_list is not None:
            merged['dataset_indices'] = np.concatenate(dataset_indices_list)

        # Shuffle to interleave samples from different datasets
        if self.shuffle_merged:
            merged = self._shuffle_batch(merged)

        return merged

    def sample(self, batch_size, idxs=None):
        """Sample a batch of transitions from the datasets.

        Args:
            batch_size: Number of samples to draw.
            idxs: Not supported for multi-dataset wrapper (must be None).

        Returns:
            Batch dictionary with concatenated samples from all datasets.
        """
        if idxs is not None:
            raise NotImplementedError("Custom indices not supported for MultiDatasetWrapper")

        if self.batch_level_sampling:
            # Each batch maintains exact proportions
            samples_per_dataset = self._compute_samples_per_dataset(batch_size)

            batches = []
            dataset_indices_list = []
            for i, (dataset, n_samples) in enumerate(zip(self.datasets, samples_per_dataset)):
                if n_samples > 0:
                    batches.append(dataset.sample(n_samples))
                    dataset_indices_list.append(np.full(n_samples, i, dtype=np.int32))

            return self._merge_batches(batches, dataset_indices_list)
        else:
            # Sample dataset indices based on weights (proportions hold in expectation)
            dataset_choices = np.random.choice(
                self.num_datasets, size=batch_size, p=self.weights
            )

            # Count samples needed from each dataset
            unique, counts = np.unique(dataset_choices, return_counts=True)

            batches = []
            dataset_indices_list = []
            for dataset_idx, count in zip(unique, counts):
                batches.append(self.datasets[dataset_idx].sample(int(count)))
                dataset_indices_list.append(np.full(count, dataset_idx, dtype=np.int32))

            return self._merge_batches(batches, dataset_indices_list)

    def sample_sequence(self, batch_size, sequence_length, discount, idxs_to_use=None):
        """Sample sequences from the datasets.

        Args:
            batch_size: Number of sequences to sample.
            sequence_length: Length of each sequence.
            discount: Discount factor for cumulative rewards.
            idxs_to_use: Not supported for multi-dataset wrapper (must be None).

        Returns:
            Batch dictionary with concatenated sequences from all datasets.
        """
        if idxs_to_use is not None:
            raise NotImplementedError("Custom indices not supported for MultiDatasetWrapper")

        if self.batch_level_sampling:
            samples_per_dataset = self._compute_samples_per_dataset(batch_size)

            batches = []
            dataset_indices_list = []
            for i, (dataset, n_samples) in enumerate(zip(self.datasets, samples_per_dataset)):
                if n_samples > 0:
                    batches.append(dataset.sample_sequence(n_samples, sequence_length, discount))
                    dataset_indices_list.append(np.full(n_samples, i, dtype=np.int32))

            return self._merge_batches(batches, dataset_indices_list)
        else:
            dataset_choices = np.random.choice(
                self.num_datasets, size=batch_size, p=self.weights
            )
            unique, counts = np.unique(dataset_choices, return_counts=True)

            batches = []
            dataset_indices_list = []
            for dataset_idx, count in zip(unique, counts):
                batches.append(
                    self.datasets[dataset_idx].sample_sequence(int(count), sequence_length, discount)
                )
                dataset_indices_list.append(np.full(count, dataset_idx, dtype=np.int32))

            return self._merge_batches(batches, dataset_indices_list)

    def sample_sequence_at_trajectory_position(self, batch_size, sequence_length, discount, position):
        """Sample sequences starting at a specific trajectory position from the datasets.

        Args:
            batch_size: Number of sequences to sample.
            sequence_length: Length of each sequence.
            discount: Discount factor for cumulative rewards.
            position: Starting position within each trajectory (0 = start of episode).

        Returns:
            Batch dictionary with concatenated sequences from all datasets.
        """
        if self.batch_level_sampling:
            samples_per_dataset = self._compute_samples_per_dataset(batch_size)

            batches = []
            dataset_indices_list = []
            for i, (dataset, n_samples) in enumerate(zip(self.datasets, samples_per_dataset)):
                if n_samples > 0:
                    batches.append(
                        dataset.sample_sequence_at_trajectory_position(
                            n_samples, sequence_length, discount, position
                        )
                    )
                    dataset_indices_list.append(np.full(n_samples, i, dtype=np.int32))

            return self._merge_batches(batches, dataset_indices_list)
        else:
            dataset_choices = np.random.choice(
                self.num_datasets, size=batch_size, p=self.weights
            )
            unique, counts = np.unique(dataset_choices, return_counts=True)

            batches = []
            dataset_indices_list = []
            for dataset_idx, count in zip(unique, counts):
                batches.append(
                    self.datasets[dataset_idx].sample_sequence_at_trajectory_position(
                        int(count), sequence_length, discount, position
                    )
                )
                dataset_indices_list.append(np.full(count, dataset_idx, dtype=np.int32))

            return self._merge_batches(batches, dataset_indices_list)

    def __len__(self):
        return self.size

    def __repr__(self):
        return (
            f"MultiDatasetWrapper(num_datasets={self.num_datasets}, "
            f"sizes={list(self.dataset_sizes)}, weights={list(np.round(self.weights, 3))}, "
            f"batch_level_sampling={self.batch_level_sampling}, "
            f"shuffle_merged={self.shuffle_merged})"
        )
    """Wrapper for sampling from multiple Dataset objects with various strategies."""