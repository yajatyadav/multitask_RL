# fast_datasets.py
from functools import partial
import threading
import queue
import time
import numpy as np
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

# -------------------------------------------------------------------
# Small helpers (kept/ported from your original module)
# -------------------------------------------------------------------
def get_size(data):
    """Return the size of the dataset (max length among arrays)."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    # jax.tree_util.tree_leaves returns python list of leaves; we want max
    leaves = jax.tree_util.tree_leaves(sizes)
    return max(leaves) if leaves else 0


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


# -------------------------------------------------------------------
# Profiler
# -------------------------------------------------------------------
class Profiler:
    """Lightweight profiler that accumulates time per labeled section."""

    def __init__(self):
        self._acc = {}
        self._count = {}
        self._stack = []

    def start(self, label):
        self._stack.append((label, time.perf_counter()))

    def stop(self):
        label, t0 = self._stack.pop()
        dt = time.perf_counter() - t0
        self._acc[label] = self._acc.get(label, 0.0) + dt
        self._count[label] = self._count.get(label, 0) + 1

    def timeit(self, label):
        """Context manager: with profiler.timeit('sample'): ..."""
        class _Ctx:
            def __enter__(inner_self):
                self.start(label)
            def __exit__(inner_self, exc_type, exc, tb):
                self.stop()
        return _Ctx()

    def report(self):
        lines = []
        for k in sorted(self._acc.keys()):
            tot = self._acc[k]
            cnt = self._count[k]
            lines.append(f"{k}: total {tot:.3f}s over {cnt} calls, avg {tot/cnt*1000:.3f} ms")
        return "\n".join(lines)

# -------------------------------------------------------------------
# FastDataset class
# -------------------------------------------------------------------
class FastDataset(FrozenDict):
    """
    Fast dataset that preserves the flattened arrays (for backward compatibility)
    and also stores per-episode lists for fast, contiguous sampling.
    """
    @classmethod
    def create(cls, freeze=True, **fields):
        data = fields
        # Set write flags off if freeze requested
        if freeze:
            jax.tree_util.tree_map(lambda arr: np.asarray(arr).setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # preserve old API size
        self.size = get_size(self._dict)
        self.frame_stack = None
        self.p_aug = None
        self.return_next_actions = False

        # Build per-episode structures if the input contains episode boundaries metadata.
        # We try to reconstruct episodes by looking for terminal signals (like original code).
        # If terminals are not informative, we fallback to whole-array as single episode.
        self._build_episode_index_if_possible()

        # Compute terminal and initial locations (for compatibility)
        if 'terminals' in self._dict:
            self.terminal_locs = np.nonzero(self._dict['terminals'] > 0)[0]
            self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1]) if len(self.terminal_locs) > 0 else np.array([0])
        else:
            self.terminal_locs = np.array([], dtype=np.int64)
            self.initial_locs = np.array([0], dtype=np.int64)

    def _build_episode_index_if_possible(self):
        """
        Construct per-episode lists:
          - ep_observations_list: list of arrays shaped (ep_len, H, W, C) or (ep_len, state_dim)
          - ep_actions_list, ep_rewards_list, ep_masks_list, ep_terminals_list
        Also keep the original concatenated arrays for API compatibility.
        """
        # If the user already supplied per-episode lists (like list-of-dicts), let's detect and keep.
        # Otherwise try to split the concatenated arrays by terminals.
        d = self._dict

        # If there is no 'terminals', treat entire arrays as single-episode
        if 'terminals' not in d or len(d['terminals']) == 0:
            # single episode fallback
            L = len(d['actions']) if 'actions' in d else get_size(d)
            self.ep_lengths = np.array([L], dtype=np.int32)
            self.ep_count = 1
            self.episode_starts = np.array([0], dtype=np.int64)
            # create single-element lists
            self._make_episode_lists_from_flat_arrays()
            return

        # find terminal indices (where terminal == 1)
        term_idxs = np.nonzero(np.asarray(d['terminals']).squeeze() > 0)[0]
        if term_idxs.size == 0:
            # no terminals: one big episode
            self.ep_lengths = np.array([len(next(iter(d.values())))])
            self.ep_count = 1
            self.episode_starts = np.array([0], dtype=np.int64)
            self._make_episode_lists_from_flat_arrays()
            return

        # Build episode start indices (0 and terminal+1)
        starts = np.concatenate([[0], term_idxs[:-1] + 1]) if len(term_idxs) > 0 else np.array([0])
        starts = starts.astype(np.int64)
        lens = []
        for i, s in enumerate(starts):
            end = (term_idxs[i] + 1) if i < len(term_idxs) else len(d['actions'])
            lens.append(end - s)
        self.ep_lengths = np.array(lens, dtype=np.int32)
        self.ep_count = len(self.ep_lengths)
        self.episode_starts = starts

        # Now slice into lists
        self._make_episode_lists_from_flat_arrays()

    def _make_episode_lists_from_flat_arrays(self):
        """Slice the concatenated arrays into per-episode lists using episode_starts and ep_lengths."""
        d = self._dict
        starts = self.episode_starts
        lens = self.ep_lengths
        N = len(starts)

        # helper to slice safely when key may be dict (observation dict) or array
        if 'observations' in d and isinstance(d['observations'], dict):
            # If observations is already a dict-of-ndarrays concatenated per-episode
            # we build lists of dicts for each ep, then convert to arrays with consistent shapes.
            # But the original get_dataset produced observations as a dict with arrays stacked along axis 0 for each key.
            self.ep_observations_list = []
            self.ep_next_observations_list = []
            for i in range(N):
                s = starts[i]
                l = lens[i]
                obs_piece = {}
                next_obs_piece = {}
                for k, arr in d['observations'].items():
                    # arr is concatenated across episodes in the original code
                    obs_piece[k] = arr[s: s + l]
                    next_obs_piece[k] = d['next_observations'][k][s: s + l]
                # If only visual keys, convert to single-array views if possible
                self.ep_observations_list.append(obs_piece)
                self.ep_next_observations_list.append(next_obs_piece)
        else:
            # observations is an ndarray shaped (T, H, W, C) or (T, state_dim)
            self.ep_observations_list = [d['observations'][starts[i]: starts[i] + lens[i]] for i in range(len(starts))]
            self.ep_next_observations_list = [d['next_observations'][starts[i]: starts[i] + lens[i]] for i in range(len(starts))]

        # actions/rewards/masks/terminals lists
        self.ep_actions_list = [d['actions'][starts[i]: starts[i] + lens[i]] for i in range(len(starts))]
        self.ep_rewards_list = [d['rewards'][starts[i]: starts[i] + lens[i]] for i in range(len(starts))]
        self.ep_masks_list = [d['masks'][starts[i]: starts[i] + lens[i]] for i in range(len(starts))]
        self.ep_terminals_list = [d['terminals'][starts[i]: starts[i] + lens[i]] for i in range(len(starts))]

    # -----------------------------
    # Compatibility methods (API preserved)
    # -----------------------------
    def get_random_idxs(self, num_idxs):
        return np.random.randint(self.size, size=num_idxs)

    def get_subset(self, idxs):
        # returns same as original implementation: jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]
        return result

    def sample(self, batch_size: int, idxs=None):
        """Keep original behavior but use contiguous slicing when we can (per index map)."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        if self.frame_stack is not None:
            # We reuse original stacking logic (kept identical)
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

    # -----------------------------
    # Fast contiguous sequence sampler
    # -----------------------------
    def sample_sequence(self, batch_size, sequence_length, discount, profiler: Profiler = None, augment_on_device=False):
        """
        Fast contiguous sampler: sample (episode, start) pairs then slice contiguous sequences.
        Returns same dict as original sample_sequence.
        `profiler` is optional; if provided, use profiler.timeit('label') contexts.
        If augment_on_device True -> attempts to run augmentations via jax (device) to avoid host copies.
        """
        # Precompute eligible episodes
        if profiler is None:
            profiler = Profiler()

        with profiler.timeit('select_ep_start'):
            valid_eps = np.where(self.ep_lengths >= sequence_length)[0]
            # sample episodes uniformly among valid episodes
            chosen_eps = np.random.choice(valid_eps, size=batch_size)
            starts = np.array([np.random.randint(0, self.ep_lengths[e] - sequence_length + 1) for e in chosen_eps], dtype=np.int64)

        with profiler.timeit('gather_slices'):
            # For arrays that are simple ndarrays, stack contiguous slices quickly.
            # The original returned shapes:
            # batch_observations: (batch_size, sequence_length, *obs_shape[1:])
            # then later transposed to (batch, h, w, seq, c) if visual.
            def stack_list_slices(list_or_arr):
                # if list_of_arrays
                if isinstance(list_or_arr, list):
                    return np.stack([list_or_arr[e][s:s + sequence_length] for e, s in zip(chosen_eps, starts)], axis=0)
                else:
                    # list_or_arr might be a big concatenated ndarray with first axis T
                    # use episode start offsets to translate to global start index
                    # i.e., global_start = episode_start + s
                    global_starts = self.episode_starts[chosen_eps] + starts
                    return np.stack([list_or_arr[gs:gs + sequence_length] for gs in global_starts], axis=0)

            # Observations handling: they might be dict-of-arrays (many keys) or single ndarray
            if isinstance(self._dict['observations'], dict):
                batch_observations = {}
                batch_next_observations = {}
                for k in self._dict['observations'].keys():
                    # if we have per-episode list-of-dicts (ep_observations_list contains dicts)
                    if isinstance(self.ep_observations_list[0], dict):
                        batch_observations[k] = np.stack([self.ep_observations_list[e][k][s:s+sequence_length] for e, s in zip(chosen_eps, starts)], axis=0)
                        batch_next_observations[k] = np.stack([self.ep_next_observations_list[e][k][s:s+sequence_length] for e, s in zip(chosen_eps, starts)], axis=0)
                    else:
                        # fallback to slicing the big concatenated arrays
                        batch_observations[k] = stack_list_slices(self._dict['observations'][k])
                        batch_next_observations[k] = stack_list_slices(self._dict['next_observations'][k])
            else:
                batch_observations = stack_list_slices(self.ep_observations_list if isinstance(self.ep_observations_list[0], np.ndarray) else self._dict['observations'])
                batch_next_observations = stack_list_slices(self.ep_next_observations_list if isinstance(self.ep_next_observations_list[0], np.ndarray) else self._dict['next_observations'])

            batch_actions = np.stack([self.ep_actions_list[e][s:s + sequence_length] for e, s in zip(chosen_eps, starts)], axis=0)
            batch_rewards = np.stack([self.ep_rewards_list[e][s:s + sequence_length] for e, s in zip(chosen_eps, starts)], axis=0)
            batch_masks = np.stack([self.ep_masks_list[e][s:s + sequence_length] for e, s in zip(chosen_eps, starts)], axis=0)
            batch_terminals = np.stack([self.ep_terminals_list[e][s:s + sequence_length] for e, s in zip(chosen_eps, starts)], axis=0)

        # next_actions: sample next indices but done per-episode boundary
        with profiler.timeit('compute_next_actions'):
            # next actions are simply shifted along time, but need to cap at end of episode
            batch_next_actions = np.zeros_like(batch_actions)
            for i, (e, start) in enumerate(zip(chosen_eps, starts)):
                ep_act = self.ep_actions_list[e]
                # take actions[start+1 : start+sequence_length+1], but cap at ep_end-1
                ep_end = ep_act.shape[0]
                # compute indices
                idxs = np.clip(np.arange(start + 1, start + 1 + sequence_length), 0, ep_end - 1)
                batch_next_actions[i] = ep_act[idxs]

        # compute discounted cumulative rewards, masks, terminals, valid (vectorized)
        with profiler.timeit('compute_rewards_masks'):
            B = batch_size
            S = sequence_length
            rewards = np.zeros((B, S), dtype=float)
            masks = np.ones((B, S), dtype=float)
            terminals = np.zeros((B, S), dtype=float)
            valid = np.ones((B, S), dtype=float)

            rewards[:, 0] = batch_rewards[:, 0].squeeze()
            masks[:, 0] = batch_masks[:, 0].squeeze()
            terminals[:, 0] = batch_terminals[:, 0].squeeze()

            discount_powers = discount ** np.arange(S)
            for i in range(1, S):
                rewards[:, i] = rewards[:, i-1] + batch_rewards[:, i].squeeze() * discount_powers[i]
                masks[:, i] = np.minimum(masks[:, i-1], batch_masks[:, i].squeeze())
                terminals[:, i] = np.maximum(terminals[:, i-1], batch_terminals[:, i].squeeze())
                valid[:, i] = 1.0 - terminals[:, i-1]

        # Reorganize observation shapes to match old API (visual: (batch, h, w, seq, c); state: (batch, seq, dim))
        with profiler.timeit('transpose'):
            def transpose_if_visual(arr):
                if arr is None:
                    return None
                if isinstance(arr, dict):
                    # produce dict of transposed arrays where necessary
                    out = {}
                    for k, a in arr.items():
                        if a.ndim == 5:  # (B, S, H, W, C)
                            out[k] = a.transpose(0, 2, 3, 1, 4)  # (B, H, W, S, C)
                        else:
                            out[k] = a
                    return out
                else:
                    if arr.ndim == 5:  # (B, S, H, W, C)
                        return arr.transpose(0, 2, 3, 1, 4)
                    else:
                        return arr

            observations_transposed = transpose_if_visual(batch_observations)
            next_observations_transposed = transpose_if_visual(batch_next_observations)

        # Optionally apply augmentation. We provide two modes:
        # - augment_on_device: move batch to device and run jax-compiled batched_random_crop (recommended)
        # - otherwise: run the augment() in-place on CPU (original code path)
        if self.p_aug is not None:
            with profiler.timeit('augment'):
                if augment_on_device:
                    # convert to float32 numpy, then to jax array and run batched_random_crop
                    # WARNING: assumes visuals are a single ndarray; if dict-of-arrays you'll need to adjust
                    # Here we handle ndarray visual case only.
                    if not isinstance(batch_observations, dict) and batch_observations.ndim == 5:
                        # flattened into (B*S, H, W, C) for cropping via vmap
                        B, S, H, W, C = batch_observations.shape
                        crop_froms = np.random.randint(0, 7, size=(B * S, 2))  # padding=3 -> 0..6
                        crop_froms = np.concatenate([crop_froms, np.zeros((B * S, 1), dtype=np.int64)], axis=1)
                        imgs = jnp.array(batch_observations.reshape(B * S, H, W, C))
                        cropped = batched_random_crop(imgs, crop_froms, 3)
                        cropped = np.array(cropped).reshape(B, S, H, W, C)
                        batch_observations = cropped
                        # do the same for next_observations if present
                        if batch_next_observations is not None and not isinstance(batch_next_observations, dict):
                            imgs2 = jnp.array(batch_next_observations.reshape(B * S, H, W, C))
                            cropped2 = batched_random_crop(imgs2, crop_froms, 3)
                            batch_next_observations = np.array(cropped2).reshape(B, S, H, W, C)
                    else:
                        # fallback to original CPU augmentation
                        self.augment({'observations': batch_observations, 'next_observations': batch_next_observations}, ['observations', 'next_observations'])
                else:
                    self.augment({'observations': batch_observations, 'next_observations': batch_next_observations}, ['observations', 'next_observations'])

        # Build return dict while preserving the old 'observations' flattened copy semantics
        with profiler.timeit('finalize_return'):
            # old API returned observations=data['observations'].copy()
            flattened_observations_copy = jax.tree_util.tree_map(lambda arr: arr.copy(), self._dict['observations'])
            ret = dict(
                observations=flattened_observations_copy,
                full_observations=observations_transposed,
                actions=batch_actions,
                masks=masks,
                rewards=rewards,
                terminals=terminals,
                valid=valid,
                next_observations=next_observations_transposed,
                next_actions=batch_next_actions,
            )
        return ret, profiler

    # keep old sample_sequence_old for safety/back-compatibility
    def sample_sequence_old(self, batch_size, sequence_length, discount):
        # you can route to original implementation if needed
        idxs = np.random.randint(self.size - sequence_length + 1, size=batch_size)
        data = jax.tree_util.tree_map(lambda v: v[idxs], self._dict)

        # Pre-compute all required indices
        all_idxs = idxs[:, None] + np.arange(sequence_length)[None, :]  # (batch_size, sequence_length)
        all_idxs = all_idxs.flatten()

        # Batch fetch data to avoid loops
        batch_observations = self._dict['observations'][all_idxs].reshape(batch_size, sequence_length, *self._dict['observations'].shape[1:])
        batch_next_observations = self._dict['next_observations'][all_idxs].reshape(batch_size, sequence_length, *self._dict['next_observations'].shape[1:])
        batch_actions = self._dict['actions'][all_idxs].reshape(batch_size, sequence_length, *self._dict['actions'].shape[1:])
        batch_rewards = self._dict['rewards'][all_idxs].reshape(batch_size, sequence_length, *self._dict['rewards'].shape[1:])
        batch_masks = self._dict['masks'][all_idxs].reshape(batch_size, sequence_length, *self._dict['masks'].shape[1:])
        batch_terminals = self._dict['terminals'][all_idxs].reshape(batch_size, sequence_length, *self._dict['terminals'].shape[1:])

        # Calculate next_actions
        next_action_idxs = np.minimum(all_idxs + 1, self.size - 1)
        batch_next_actions = self._dict['actions'][next_action_idxs].reshape(batch_size, sequence_length, *self._dict['actions'].shape[1:])

        # Compute the rewards/masks/terminals/valid (identical to original)
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
        if len(batch_observations.shape) == 5:  # Visual data: (batch, seq, h, w, c)
            observations = batch_observations.transpose(0, 2, 3, 1, 4)
            next_observations = batch_next_observations.transpose(0, 2, 3, 1, 4)
        else:
            observations = batch_observations
            next_observations = batch_next_observations

        return dict(
            observations=self._dict['observations'].copy(),
            full_observations=observations,
            actions=batch_actions,
            masks=masks,
            rewards=rewards,
            terminals=terminals,
            valid=valid,
            next_observations=next_observations,
            next_actions=batch_next_actions,
        )

    # augment kept mostly identical but avoid unnecessary conversions
    def augment(self, batch, keys):
        padding = 3
        # detect batch size from first key
        first_key = keys[0]
        if isinstance(batch[first_key], dict):
            # handle dict-of-arrays case; iterate keys
            # In this mode we expect e.g. batch['observations'] -> {'agentview': (B,S,H,W,C), 'proprio': ...}
            for topk, arr in batch[first_key].items():
                B = arr.shape[0]
                crop_froms = np.random.randint(0, 2 * padding + 1, (B, 2))
                crop_froms = np.concatenate([crop_froms, np.zeros((B, 1), dtype=np.int64)], axis=1)
                batch[first_key][topk] = np.array(batched_random_crop(arr.reshape(-1, *arr.shape[2:]), np.repeat(crop_froms, arr.shape[1], axis=0), padding)).reshape(arr.shape)
        else:
            # simple ndarray case (B, S, H, W, C) or (N, H, W, C) etc.
            for key in keys:
                arr = batch[key]
                if len(arr.shape) == 4:
                    B = arr.shape[0]
                    crop_froms = np.random.randint(0, 2 * padding + 1, (B, 2))
                    crop_froms = np.concatenate([crop_froms, np.zeros((B, 1), dtype=np.int64)], axis=1)
                    batch[key] = np.array(batched_random_crop(arr, crop_froms, padding))
                elif len(arr.shape) == 5:
                    B = arr.shape[0] * arr.shape[1]
                    # flatten B*S into batch
                    flattened = arr.reshape(B, *arr.shape[2:])
                    crop_froms = np.random.randint(0, 2 * padding + 1, (B, 2))
                    crop_froms = np.concatenate([crop_froms, np.zeros((B, 1), dtype=np.int64)], axis=1)
                    cropped = np.array(batched_random_crop(flattened, crop_froms, padding)).reshape(arr.shape)
                    batch[key] = cropped
                else:
                    # non-visual fields unchanged
                    batch[key] = arr

# -------------------------------------------------------------------
# Prefetcher: background producer thread that fills a queue with ready batches
# -------------------------------------------------------------------
class Prefetcher:
    """
    Prefetcher produces sample_sequence batches in a background thread/process.
    Usage:
        pf = Prefetcher(dataset, batch_size, seq_len, discount, max_prefetch=4)
        pf.start()
        batch = pf.get()   # get one prepared batch (and optionally profiler)
        ...
        pf.stop()
    """
    def __init__(self, dataset: FastDataset, batch_size, seq_len, discount, max_prefetch=4, augment_on_device=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.discount = discount
        self.queue = queue.Queue(maxsize=max_prefetch)
        self._stop_event = threading.Event()
        self._thread = None
        self.augment_on_device = augment_on_device
        self._profilers = []  # keep last few profilers if needed

    def _producer(self):
        while not self._stop_event.is_set():
            p = Profiler()
            batch, profiler = self.dataset.sample_sequence(self.batch_size, self.seq_len, self.discount, profiler=p, augment_on_device=self.augment_on_device)
            # push tuple(batch, profiler)
            try:
                self.queue.put((batch, profiler), timeout=1.0)
            except queue.Full:
                # allow gracefully exit if queue is full and stop requested
                continue

    def start(self):
        if self._thread is None:
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._producer, daemon=True)
            self._thread.start()

    def get(self, timeout=None):
        item = self.queue.get(timeout=timeout)
        return item  # returns (batch, profiler)

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        # drain the queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

# -------------------------------------------------------------------
# Utility: a drop-in get_dataset() that mirrors your previous behavior but returns FastDataset
# -------------------------------------------------------------------
import glob
import h5py
import os

def stack_dict_list(dict_list):
    """Stack a list of dictionaries into a dictionary of stacked arrays (like your original helper)."""
    if not dict_list:
        return {}
    keys = dict_list[0].keys()
    return {k: np.concatenate([d[k] for d in dict_list], axis=0) for k in keys}

def get_dataset_fast(env_name, task_name, augment_negative_demos, keys_to_load):
    """
    Replacement for your original get_dataset function.
    It reads HDF5 files, builds per-episode lists and also builds the flattened arrays
    to preserve the old Dataset.create() interface (so previous code using data['observations'] still works).
    """
    observations = []
    actions = []
    next_observations = []
    terminals = []
    rewards = []
    masks = []

    # inner helper largely same logic as original but appends per-episode arrays to lists
    def process_task(rm_dataset, flip_rewards, this_task_name):
        demos = list(rm_dataset["data"].keys())
        inds = np.argsort([int(elem[5:]) for elem in demos])
        demos = [demos[i] for i in inds]
        # task embedding snippet kept for compatibility: user should define OneHotEmbedding_Libero externally
        # For brevity here, assume language key is handled outside
        this_task_num_timesteps = 0
        for ep in demos:
            a = np.array(rm_dataset[f"data/{ep}/actions"])
            this_task_num_timesteps += a.shape[0]
            obs, next_obs = {}, {}
            for k in keys_to_load:
                if k == 'states':
                    obs[k] = np.array(rm_dataset[f"data/{ep}/{k}"])[:, 1:]
                elif k == 'language':
                    # placeholder: users should supply OneHot embedding function
                    # replicate previous behavior: we will create an array per timestep externally if needed
                    raise NotImplementedError("language key requires OneHotEmbedding_Libero in scope; adapt caller.")
                elif k == 'proprio':
                    obs[k] = np.concatenate(
                        [
                            np.array(rm_dataset[f"data/{ep}/obs/ee_pos"]),
                            np.array(rm_dataset[f"data/{ep}/obs/ee_ori"]),
                            np.array(rm_dataset[f"data/{ep}/obs/gripper_states"]),
                        ],
                        axis=-1,
                    )
                else:
                    obs[k] = np.array(rm_dataset[f"data/{ep}/obs/{k}"])
            for k in keys_to_load:
                if k == 'states':
                    obs_array = np.array(rm_dataset[f"data/{ep}/{k}"])[:, 1:]
                elif k == 'language':
                    raise NotImplementedError("language key requires OneHotEmbedding_Libero in scope; adapt caller.")
                elif k == 'proprio':
                    obs_array = np.concatenate(
                        [
                            np.array(rm_dataset[f"data/{ep}/obs/ee_pos"]),
                            np.array(rm_dataset[f"data/{ep}/obs/ee_ori"]),
                            np.array(rm_dataset[f"data/{ep}/obs/gripper_states"]),
                        ],
                        axis=-1,
                    )
                else:
                    obs_array = np.array(rm_dataset[f"data/{ep}/obs/{k}"])

                next_obs[k] = np.concatenate([obs_array[1:], obs_array[-1:]], axis=0)

            dones = np.array(rm_dataset[f"data/{ep}/dones"])
            r = np.array(rm_dataset[f"data/{ep}/rewards"], dtype=np.float32)
            if flip_rewards:
                r = np.full_like(r, -1.0)

            # append per-episode arrays (to lists) — these are used by FastDataset for contiguous reads
            observations.append(obs if isinstance(obs, dict) else obs)
            next_observations.append(next_obs if isinstance(next_obs, dict) else next_obs)
            actions.append(a.astype(np.float32))
            rewards.append(r.astype(np.float32))
            terminals.append(dones.astype(np.float32))
            masks.append(1.0 - dones.astype(np.float32))
        return this_task_num_timesteps

    # locate hdf5 files like your previous get_dataset (use same path logic)
    libero_dataset_dir = os.path.join(os.path.dirname(os.getcwd()), 'datasets/raw_libero')

    suites = []
    scenes = []
    if env_name.startswith("all_libero"):
        suites = ["libero_spatial", "libero_object", "libero_goal", "libero_90"]
        scenes = ['*', '*', '*', '*', '*']
    else:
        suite = env_name.split("-")[0]
        scene = ''
        if suite == "libero_90":
            scene = env_name.split("-")[1]
        suites = [suite]
        scenes = [scene]
    suites = sorted(suites)
    scenes = sorted(scenes)

    num_timesteps = 0
    j = 0
    for i, (suite, scene) in enumerate(zip(suites, scenes)):
        suite = suite.upper()
        pattern = os.path.join(libero_dataset_dir, suite, f"{scene.upper()}*.hdf5")
        for filepath in sorted(glob.glob(pattern)):
            rm_dataset = h5py.File(filepath, "r")
            # fallback: decide flip_rewards based on caller args; here we set False for clarity
            flip_rewards = False
            tsteps = process_task(rm_dataset, flip_rewards, os.path.basename(filepath))
            num_timesteps += tsteps
            j += 1

    # Build the concatenated flattened arrays (old API)
    flat_observations = stack_dict_list(observations) if isinstance(observations[0], dict) else np.concatenate(observations, axis=0)
    flat_next_observations = stack_dict_list(next_observations) if isinstance(next_observations[0], dict) else np.concatenate(next_observations, axis=0)
    flat_actions = np.concatenate(actions, axis=0)
    flat_rewards = np.concatenate(rewards, axis=0)
    flat_terminals = np.concatenate(terminals, axis=0)
    flat_masks = np.concatenate(masks, axis=0)

    # Create the FastDataset. We still keep the flattened arrays in the dict for backward compatibility.
    ds = FastDataset.create(
        observations=flat_observations,
        next_observations=flat_next_observations,
        actions=flat_actions,
        rewards=flat_rewards,
        terminals=flat_terminals,
        masks=flat_masks,
    )

    # But attach the per-episode lists to the dataset so sample_sequence uses them.
    # The FastDataset already tried to build per-episode lists in __init__, but since we created
    # it from the flattened arrays above, we now explicitly set the ep_* lists based on our pre-read lists.
    ds.ep_observations_list = observations
    ds.ep_next_observations_list = next_observations
    ds.ep_actions_list = actions
    ds.ep_rewards_list = rewards
    ds.ep_masks_list = masks
    ds.ep_terminals_list = terminals
    ds.ep_lengths = np.array([a.shape[0] for a in actions], dtype=np.int32)
    ds.ep_count = len(actions)
    ds.episode_starts = np.cumsum(np.concatenate([[0], ds.ep_lengths[:-1]]), dtype=np.int64)

    ds.size = sum(ds.ep_lengths)
    return ds