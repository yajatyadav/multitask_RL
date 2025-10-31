from functools import partial
import time
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
    """Randomly crop an image."""
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Dataset class with optimized sample_sequence."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields."""
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.frame_stack = None
        self.p_aug = None
        self.return_next_actions = False

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        
        # Cache for optimizations
        self._is_image_obs = None
        self.profile = False
        self.profile_times = {}

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        if self.frame_stack is not None:
            initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
            obs = []
            next_obs = []
            for i in reversed(range(self.frame_stack)):
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
                if i != self.frame_stack - 1:
                    next_obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
            next_obs.append(jax.tree_util.tree_map(lambda arr: arr[idxs], self['next_observations']))

            batch['observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *obs)
            batch['next_observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *next_obs)
        if self.p_aug is not None:
            if np.random.rand() < self.p_aug:
                self.augment(batch, ['observations', 'next_observations'])
        return batch

    def sample_sequence_optimized(self, batch_size, sequence_length, discount):
        """Optimized version of sample_sequence.
        
        Key optimizations:
        1. Use advanced indexing more efficiently
        2. Pre-allocate arrays with correct shape
        3. Avoid unnecessary reshapes
        4. Vectorize reward/mask calculations
        5. Minimize memory copies
        """
        times = {} if self.profile else None
        
        if self.profile:
            t0 = time.perf_counter()
        
        # Generate starting indices
        idxs = np.random.randint(self.size - sequence_length + 1, size=batch_size)
        
        if self.profile:
            times['idx_generation'] = time.perf_counter() - t0
            t0 = time.perf_counter()
        
        # Create index array for sequences - shape (batch_size, sequence_length)
        seq_idxs = idxs[:, None] + np.arange(sequence_length)[None, :]
        
        if self.profile:
            times['idx_computation'] = time.perf_counter() - t0
            t0 = time.perf_counter()
        
        # OPTIMIZATION 1: Use fancy indexing directly with 2D array instead of flattening
        # This is more memory efficient and can be faster for contiguous sequences
        batch_observations = self['observations'][seq_idxs]
        
        if self.profile:
            times['fetch_observations'] = time.perf_counter() - t0
            t0 = time.perf_counter()
        
        batch_next_observations = self['next_observations'][seq_idxs]
        batch_actions = self['actions'][seq_idxs]
        batch_rewards = self['rewards'][seq_idxs]
        batch_masks = self['masks'][seq_idxs]
        batch_terminals = self['terminals'][seq_idxs]
        
        if self.profile:
            times['fetch_all_data'] = time.perf_counter() - t0
            t0 = time.perf_counter()
        
        # OPTIMIZATION 2: Calculate next_actions more efficiently
        next_action_idxs = np.minimum(seq_idxs + 1, self.size - 1)
        batch_next_actions = self['actions'][next_action_idxs]
        
        if self.profile:
            times['fetch_next_actions'] = time.perf_counter() - t0
            t0 = time.perf_counter()
        
        # OPTIMIZATION 3: Vectorize cumulative calculations more efficiently
        # Squeeze rewards/masks/terminals upfront
        batch_rewards_squeezed = batch_rewards.squeeze(-1) if batch_rewards.shape[-1] == 1 else batch_rewards
        batch_masks_squeezed = batch_masks.squeeze(-1) if batch_masks.shape[-1] == 1 else batch_masks
        batch_terminals_squeezed = batch_terminals.squeeze(-1) if batch_terminals.shape[-1] == 1 else batch_terminals
        
        # Calculate cumulative rewards using cumsum (more efficient)
        discount_powers = discount ** np.arange(sequence_length)
        discounted_rewards = batch_rewards_squeezed * discount_powers[None, :]
        rewards = np.cumsum(discounted_rewards, axis=1)
        
        # Calculate masks and terminals using cummin/cummax
        masks = np.minimum.accumulate(batch_masks_squeezed, axis=1)
        terminals = np.maximum.accumulate(batch_terminals_squeezed, axis=1)
        
        # Calculate valid mask
        valid = np.ones((batch_size, sequence_length), dtype=float)
        if sequence_length > 1:
            valid[:, 1:] = 1.0 - terminals[:, :-1]
        
        if self.profile:
            times['compute_rewards_masks'] = time.perf_counter() - t0
            t0 = time.perf_counter()
        
        # OPTIMIZATION 4: Only transpose if needed, and cache the check
        if self._is_image_obs is None:
            self._is_image_obs = len(batch_observations.shape) == 5
        
        if self._is_image_obs:
            # Visual data: transpose to (batch, h, w, seq, c)
            observations = np.transpose(batch_observations, (0, 2, 3, 1, 4))
            next_observations = np.transpose(batch_next_observations, (0, 2, 3, 1, 4))
        else:
            # State data: keep as is
            observations = batch_observations
            next_observations = batch_next_observations
        
        if self.profile:
            times['transpose_observations'] = time.perf_counter() - t0
            t0 = time.perf_counter()
        
        # OPTIMIZATION 5: Avoid unnecessary copy by directly indexing
        result = dict(
            observations=self['observations'][idxs],  # Direct indexing instead of copy
            full_observations=observations,
            actions=batch_actions,
            masks=masks,
            rewards=rewards,
            terminals=terminals,
            valid=valid,
            next_observations=next_observations,
            next_actions=batch_next_actions,
        )
        
        if self.profile:
            times['create_result_dict'] = time.perf_counter() - t0
            self.profile_times = times
        
        return result

    def sample_sequence(self, batch_size, sequence_length, discount):
        """Original version kept for compatibility."""
        idxs = np.random.randint(self.size - sequence_length + 1, size=batch_size)
        
        data = {k: v[idxs] for k, v in self.items()}

        # Pre-compute all required indices
        all_idxs = idxs[:, None] + np.arange(sequence_length)[None, :]
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
        
        # Reorganize observations data format
        if len(batch_observations.shape) == 5:
            observations = batch_observations.transpose(0, 2, 3, 1, 4)
            next_observations = batch_next_observations.transpose(0, 2, 3, 1, 4)
        else:
            observations = batch_observations
            next_observations = batch_next_observations
        
        actions = batch_actions
        next_actions = batch_next_actions
        
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
    
    def print_profile(self):
        """Print profiling results."""
        if not self.profile_times:
            print("No profiling data available. Set dataset.profile = True before sampling.")
            return
        
        print("\n=== Sample Sequence Profiling ===")
        total = sum(self.profile_times.values())
        for key, value in sorted(self.profile_times.items(), key=lambda x: x[1], reverse=True):
            print(f"{key:30s}: {value*1000:8.2f} ms ({value/total*100:5.1f}%)")
        print(f"{'TOTAL':30s}: {total*1000:8.2f} ms")
        print("=" * 50)

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
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
    """Replay buffer class."""

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition."""
        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset."""
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