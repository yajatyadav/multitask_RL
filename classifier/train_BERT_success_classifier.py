import os
import sys

import copy
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
import tqdm
import numpy as np
import math
import random, json, pickle
import time
import wandb

# Enable persistent compilation cache
jax.config.update('jax_compilation_cache_dir', '/tmp/jax_cache')
jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field, restore_agent_with_file
import pickle
from utils.networks import MLP
from typing import Sequence
from collections import defaultdict

from envs.libero_utils import get_dataset as get_libero_dataset, make_env as make_libero_env
from evaluation_libero import evaluate
from utils.log_utils import get_wandb_video

from utils.datasets import Dataset, MultiDatasetWrapper

from envs.libero_utils import get_single_dataset
from libero.libero.benchmark.libero_suite_task_map import libero_task_map
from utils.log_utils import build_network_tree
import os
import pickle
import h5py
import numpy as np
from pathlib import Path
from jax import tree_util
import tqdm
import re

from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed


def _parse_trajs_suffix_date(pkl_path):
    """
    Extract date from trajs_*.pkl filename suffix.
    Suffix format from collect_rollout_from_bcactor: gpu{N}_{YYYYMMDD}_{HHMMSS}{random_digits}.
    Returns string YYYYMMDD_HHMMSS (15 chars) or None if unparseable.
    """
    stem = pkl_path.stem
    if not stem.startswith('trajs_'):
        return None
    suffix = stem[6:]
    m = re.match(r'gpu\d+_(\d{8}_\d{6})', suffix)
    if m is None:
        return None
    return m.group(1)


def _filter_pkl_files_by_date(pkl_files, data_from_date_before):
    """
    Keep only pkl files whose suffix date is <= data_from_date_before.
    data_from_date_before: YYYYMMDD (8 chars) or YYYYMMDD_HHMMSS (15 chars). Comparison is string lexicographic.
    Files with unparseable suffixes are included (not filtered out).
    """
    if data_from_date_before is None or data_from_date_before == '':
        return list(pkl_files)
    cutoff = data_from_date_before.strip()
    if len(cutoff) == 8:
        cutoff_15 = cutoff + '_235959'
    elif len(cutoff) == 15:
        cutoff_15 = cutoff
    else:
        raise ValueError(
            f"data_from_date_before must be YYYYMMDD (8 chars) or YYYYMMDD_HHMMSS (15 chars), got {len(cutoff)} chars: {cutoff!r}"
        )
    out = []
    for p in pkl_files:
        file_dt = _parse_trajs_suffix_date(p)
        if file_dt is None:
            out.append(p)
        elif file_dt <= cutoff_15:
            out.append(p)
    
    # if files filtered out, print
    print(f"⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️ Filtered out {len(pkl_files) - len(out)}/{len(pkl_files)} pkl files with date before {data_from_date_before!r}")
    return out


class TransClassifier_BERT(nn.Module):
    vision_encoder: nn.Module = None    
    layer_norm: bool = True
    p_drop_state: float = 0.5
    embed_dim: int = 128

    def setup(self):
        # Action encoder: action_dim -> 128
        self.action_encoder = MLP((self.embed_dim, self.embed_dim), activate_final=True, layer_norm=self.layer_norm)
        
        # Language encoder: 768 (BERT) -> 128
        self.lang_encoder = MLP((self.embed_dim, self.embed_dim), activate_final=True, layer_norm=self.layer_norm)
        
        # Final classifier: 384 (128*3) -> 128 -> 128 -> 1
        self.classifier = MLP((self.embed_dim, self.embed_dim, 1), activate_final=False, layer_norm=self.layer_norm)
    
    def __call__(self, observations, actions, language_embedding, train=True, rng=None):
        """
        Args:
            observations: (batch, obs_dim)
            actions: (batch, action_dim)
            language_embedding: (batch, 768) - BERT embedding
            train: whether to apply dropout
            rng: random key for dropout
            
        Returns:
            logits: (batch, 1) - scalar logit for P(lang | obs, action)
        """
        assert actions is not None, "Actions must be provided to the classifier"
        assert self.vision_encoder is not None, "Encoder must be provided to the classifier"
        
        # Encode observations -> (batch, 128)
        obs_encoded = self.vision_encoder(observations)

        # Dropout on obs encoding
        if rng is None:
            rng = jax.random.PRNGKey(0)
        mask = jax.random.bernoulli(rng, 1 - self.p_drop_state, shape=(obs_encoded.shape[0], 1))
        obs_encoded = jax.lax.cond(
            train,
            lambda x: mask * x,
            lambda x: x,
            obs_encoded
        )
        
        # Encode actions -> (batch, 128)
        action_encoded = self.action_encoder(actions)
        
        # Encode language -> (batch, 768)
        lang_encoded = self.lang_encoder(language_embedding)
        
        # Concatenate all three -> (batch, 384)
        inputs = jnp.concatenate([obs_encoded, action_encoded, lang_encoded], axis=-1)
        
        # Final classification -> (batch, 1)
        return self.classifier(inputs)


def save_classifier(network, step, save_dir, hparams):
    """Save everything needed to restore the classifier."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save parameters
    with open(save_dir / f'params_{step}.pkl', 'wb') as f:
        pickle.dump(network.params, f)
    
    # Save hyperparameters
    with open(save_dir / f'hparams_{step}.json', 'w') as f:
        json.dump(hparams, f, indent=2)
    
    print(f"Saved classifier to {save_dir}")


def load_classifier(save_dir, ckpt_number):
    """Load classifier from saved directory."""
    save_dir = Path(save_dir)
    
    # Load hyperparameters
    with open(save_dir / f'hparams_{ckpt_number}.json', 'r') as f:
        hparams = json.load(f)
    
    # Load parameters
    with open(save_dir / f'params_{ckpt_number}.pkl', 'rb') as f:
        params = pickle.load(f)
    
    # Recreate model definition
    classifier_def = TransClassifier_BERT(
        vision_encoder=encoder_modules[hparams['encoder']](),
        embed_dim=hparams['embed_dim'],
        layer_norm=hparams['layer_norm'],
        p_drop_state=hparams['p_drop_state'],
    )
    
    networks = {'classifier': classifier_def}
    network_def = ModuleDict(networks)
    network_tx = optax.adam(learning_rate=hparams['lr'])
    network = TrainState.create(network_def, params, tx=network_tx)
    
    return network, hparams

def batch_dicts(dicts):
    return tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *dicts)

def stack_dict_list(dict_list):
    """Stack a list of dictionaries into a dictionary of stacked arrays."""
    if not dict_list:
        return {}
    keys = dict_list[0].keys()
    return {k: np.concatenate([d[k] for d in dict_list], axis=0) for k in keys}


def load_pickle_file(path):
    """Load a single pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_single_task_thread(args):
    """
    Load a single task - designed for ThreadPoolExecutor.
    Returns Dataset objects directly (no serialization needed with threads).
    Uses train_fraction and val_fraction to split demos so that train and val
    have equal proportion of success (max_rew==1) and failure (max_rew==0) trajectories.
    """
    task_dir, task_name, train_fraction, val_fraction, action_clip_eps, seed, rollouts_per_task, data_from_date_before = args
    
    assert abs((train_fraction + val_fraction) - 1.0) < 1e-6, (
        f"train_fraction + val_fraction must equal 1.0, got {train_fraction} + {val_fraction}"
    )
    
    pkl_files = sorted(Path(task_dir).glob('trajs_*.pkl'))
    if data_from_date_before is not None and data_from_date_before != '':
        n_before = len(pkl_files)
        pkl_files = _filter_pkl_files_by_date(pkl_files, data_from_date_before)
        if len(pkl_files) < n_before:
            print(f"  [{task_name}] Using {len(pkl_files)}/{n_before} pkl files with date before {data_from_date_before!r}")
    if not pkl_files:
        raise ValueError(f"No pickle files (after filtering) for {task_name}")
    print(f"Loading {len(pkl_files)} pickle files for {task_name}")
    
    # Load all pickle files
    all_datasets = [load_pickle_file(p) for p in pkl_files]
    
    t0 = time.perf_counter()
    # First pass: process all demos and classify by success (max_rew==1) vs failure (max_rew==0)
    success_episodes = []  # list of (dataset_idx, ep_idx, timesteps)
    failure_episodes = []

    ep_counter = 0    
    for dataset_idx, dataset in enumerate(all_datasets):
        for ep_idx, ep in enumerate(dataset):
            if ep_counter >= rollouts_per_task and rollouts_per_task != -1:
                break
            timesteps = len(ep['actions'])
            rew_list = ep['rewards']
            assert len(rew_list) == timesteps, f"Reward list length {len(rew_list)} does not match timesteps {timesteps}"
            max_rew = max(rew_list)
            is_success = (max_rew == 1)
            entry = (dataset_idx, ep_idx, timesteps)
            if is_success:
                success_episodes.append(entry)
            else:
                failure_episodes.append(entry)
            ep_counter += 1
    print(f"Loaded { ep_counter} = {len(success_episodes)} success episodes and {len(failure_episodes)} failure episodes for {task_name}")
    
    # Deterministic shuffle using provided seed
    rng = random.Random(seed)
    rng.shuffle(success_episodes)
    rng.shuffle(failure_episodes)
    
    # Split each group by fractions: train_fraction to train, val_fraction to val
    n_success = len(success_episodes)
    n_failure = len(failure_episodes)
    n_train_success = int(round(n_success * train_fraction))
    n_train_failure = int(round(n_failure * train_fraction))
    
    train_successes = success_episodes[:n_train_success]
    val_successes = success_episodes[n_train_success:]
    train_failures = failure_episodes[:n_train_failure]
    val_failures = failure_episodes[n_train_failure:]
    
    train_demos = train_successes + train_failures
    val_demos = val_successes + val_failures
    
    train_total = sum(ts for (_, _, ts) in train_demos)
    val_total = sum(ts for (_, _, ts) in val_demos)
    elapsed = time.perf_counter() - t0
    print(f"  [{task_name}] Built train/val demo lists in {elapsed:.3f}s ({len(train_demos)} train, {len(val_demos)} val)")
    # print succ/fail counts given to train and val sets
    print(f"  [{task_name}] Train Successes: {len(train_successes)}, Train Failures: {len(train_failures)}, Val Successes: {len(val_successes)}, Val Failures: {len(val_failures)}")
    # print succes rate in rollout data
    print(f"  [{task_name}] Rollout Success Rate: {len(success_episodes) / (len(success_episodes) + len(failure_episodes))}")
    
    if train_total == 0:
        return task_name, None, None, f"No training data for {task_name}"
    if val_total == 0:
        return task_name, None, None, f"No validation data for {task_name}"
    
    # Get shapes from first episode
    first_ep = all_datasets[0][0]
    first_obs = first_ep['observations'][0]
    obs_keys = list(first_obs.keys())
    action_dim = len(first_ep['actions'][0])
    
    def fill_arrays(demos_info, total_timesteps):
        """Pre-allocate and fill arrays for a set of demos."""
        obs_arrays = {}
        next_obs_arrays = {}
        for k in obs_keys:
            shape = first_obs[k].shape
            dtype = first_obs[k].dtype
            obs_arrays[k] = np.empty((total_timesteps, *shape), dtype=dtype)
            next_obs_arrays[k] = np.empty((total_timesteps, *shape), dtype=dtype)
        
        actions = np.empty((total_timesteps, action_dim), dtype=np.float32)
        rewards = np.empty((total_timesteps,), dtype=np.float32)
        dones = np.empty((total_timesteps,), dtype=np.float32)
        masks = np.empty((total_timesteps,), dtype=np.float32)
        successes = np.empty((total_timesteps,), dtype=np.float32)
        
        offset = 0
        for dataset_idx, ep_idx, timesteps in demos_info:
            ep = all_datasets[dataset_idx][ep_idx]
            end = offset + timesteps
            
            obs_list = ep['observations']
            next_obs_list = ep['next_observations']
            
            for k in obs_keys:
                arr = obs_arrays[k]
                next_arr = next_obs_arrays[k]
                for t, (o, no) in enumerate(zip(obs_list, next_obs_list)):
                    arr[offset + t] = o[k]
                    next_arr[offset + t] = no[k]
            
            act_list = ep['actions']
            rew_list = ep['rewards']
            done_list = ep['dones']
            
            for t in range(timesteps):
                actions[offset + t] = act_list[t]
                rewards[offset + t] = rew_list[t]
                dones[offset + t] = done_list[t]
            
            max_rew = max(rew_list)
            successes[offset:end] = max_rew
            offset = end
        
        np.clip(actions, -1 + action_clip_eps, 1 - action_clip_eps, out=actions)
        np.subtract(1.0, dones, out=masks)
        
        return obs_arrays, next_obs_arrays, actions, rewards, dones, masks, successes
    
    # Fill train arrays
    train_obs, train_next_obs, train_actions, train_rewards, train_dones, train_masks, train_successes = \
        fill_arrays(train_demos, train_total)
    
    # Fill val arrays  
    val_obs, val_next_obs, val_actions, val_rewards, val_dones, val_masks, val_successes = \
        fill_arrays(val_demos, val_total)
    
    # Create Dataset objects directly in thread (shares memory with main process)
    train_ds = Dataset.create(
        observations=train_obs,
        next_observations=train_next_obs,
        actions=train_actions,
        rewards=train_rewards,
        terminals=train_dones,
        masks=train_masks,
        successes=train_successes,
    )
    
    val_ds = Dataset.create(
        observations=val_obs,
        next_observations=val_next_obs,
        actions=val_actions,
        rewards=val_rewards,
        terminals=val_dones,
        masks=val_masks,
        successes=val_successes,
    )
    
    return task_name, train_ds, val_ds, None


# def load_single_task_hdf5(args):
#     """
#     Load a single task from HDF5 file - designed for ThreadPoolExecutor.
#     HDF5 format is much faster since data is already in array form.
#     """
#     task_dir, task_name, train_demo_nums, val_demo_nums, action_clip_eps = args
    
#     hdf5_path = Path(task_dir) / "demos.hdf5"
#     if not hdf5_path.exists():
#         return task_name, None, None, f"No demos.hdf5 for {task_name}"
    
#     train_indices = set(train_demo_nums)
#     val_indices = set(val_demo_nums)
    
#     with h5py.File(hdf5_path, 'r') as hdf_file:
#         data_group = hdf_file['data']
#         num_demos = hdf_file.attrs['num_demos']
        
#         # First pass: count timesteps for train/val
#         train_demos = []
#         val_demos = []
#         train_total = 0
#         val_total = 0
        
#         for demo_idx in range(num_demos):
#             if demo_idx not in train_indices and demo_idx not in val_indices:
#                 continue
#             demo_key = f'demo_{demo_idx}'
#             if demo_key not in data_group:
#                 continue
#             timesteps = data_group[demo_key]['actions'].shape[0]
#             if demo_idx in train_indices:
#                 train_total += timesteps
#                 train_demos.append((demo_key, timesteps))
#             elif demo_idx in val_indices:
#                 val_total += timesteps
#                 val_demos.append((demo_key, timesteps))
        
#         if train_total == 0:
#             return task_name, None, None, f"No training data for {task_name}"
#         if val_total == 0:
#             return task_name, None, None, f"No validation data for {task_name}"
        
#         # Get shapes from first demo
#         first_demo = data_group['demo_0']
#         obs_keys = list(first_demo['obs'].keys())
#         action_dim = first_demo['actions'].shape[1]
        
#         def fill_arrays_hdf5(demos_info, total_timesteps):
#             """Pre-allocate and fill arrays from HDF5 data."""
#             # Get shapes/dtypes from first demo
#             first_demo_key = demos_info[0][0]
#             fd = data_group[first_demo_key]
            
#             obs_arrays = {}
#             next_obs_arrays = {}
#             for k in obs_keys:
#                 shape = fd[f'obs/{k}'].shape[1:]  # Remove batch dim
#                 dtype = fd[f'obs/{k}'].dtype
#                 obs_arrays[k] = np.empty((total_timesteps, *shape), dtype=dtype)
#                 next_obs_arrays[k] = np.empty((total_timesteps, *shape), dtype=dtype)
            
#             actions = np.empty((total_timesteps, action_dim), dtype=np.float32)
#             rewards = np.empty((total_timesteps,), dtype=np.float32)
#             dones = np.empty((total_timesteps,), dtype=np.float32)
#             masks = np.empty((total_timesteps,), dtype=np.float32)
#             successes = np.empty((total_timesteps,), dtype=np.float32)
            
#             offset = 0
#             for demo_key, timesteps in demos_info:
#                 demo = data_group[demo_key]
#                 end = offset + timesteps
                
#                 # Direct array copy from HDF5 (very fast!)
#                 for k in obs_keys:
#                     obs_arrays[k][offset:end] = demo[f'obs/{k}'][:]
#                     next_obs_arrays[k][offset:end] = demo[f'next_obs/{k}'][:]
                
#                 actions[offset:end] = demo['actions'][:]
#                 rewards[offset:end] = demo['rewards'][:]
#                 dones[offset:end] = demo['dones'][:].astype(np.float32)
                
#                 max_rew = rewards[offset:end].max()
#                 successes[offset:end] = max_rew
#                 offset = end
            
#             np.clip(actions, -1 + action_clip_eps, 1 - action_clip_eps, out=actions)
#             np.subtract(1.0, dones, out=masks)
            
#             return obs_arrays, next_obs_arrays, actions, rewards, dones, masks, successes
        
#         # Fill train arrays
#         train_obs, train_next_obs, train_actions, train_rewards, train_dones, train_masks, train_successes = \
#             fill_arrays_hdf5(train_demos, train_total)
        
#         # Fill val arrays
#         val_obs, val_next_obs, val_actions, val_rewards, val_dones, val_masks, val_successes = \
#             fill_arrays_hdf5(val_demos, val_total)
    
#     # Create Dataset objects (outside the h5py context)
#     train_ds = Dataset.create(
#         observations=train_obs,
#         next_observations=train_next_obs,
#         actions=train_actions,
#         rewards=train_rewards,
#         terminals=train_dones,
#         masks=train_masks,
#         successes=train_successes,
#     )
    
#     val_ds = Dataset.create(
#         observations=val_obs,
#         next_observations=val_next_obs,
#         actions=val_actions,
#         rewards=val_rewards,
#         terminals=val_dones,
#         masks=val_masks,
#         successes=val_successes,
#     )
    
#     return task_name, train_ds, val_ds, None


def load_all_tasks_parallel(rollouts_dir, task_names, train_fraction, val_fraction,
                            seed, rollouts_per_task, num_workers=16, action_clip_eps=1e-5, data_format='pickle',
                            data_from_date_before=None):
    """
    Load all tasks in parallel using ThreadPoolExecutor.
    
    Args:
        data_format: 'pickle' or 'hdf5'
        seed: used for deterministic shuffle when splitting success/failure into train/val
        data_from_date_before: only use trajs_*.pkl with suffix date on or before this (YYYYMMDD or YYYYMMDD_HHMMSS)
    """
    task_args = [
        (rollouts_dir / task_name, task_name, train_fraction, val_fraction, action_clip_eps, seed, rollouts_per_task, data_from_date_before)
        for task_name in task_names
    ]
    
    # Choose loader based on format
    # HDF5 with gzip is CPU-bound (decompression), so fewer workers helps
    if data_format == 'hdf5':
        raise ValueError("HDF5 format is not supported yet")
        loader_fn = load_single_task_hdf5
        effective_workers = min(num_workers, 8)  # Limit for CPU-bound decompression
        print(f"Loading {len(task_names)} tasks from HDF5 with {effective_workers} threads...")
    else:
        loader_fn = load_single_task_thread
        effective_workers = num_workers
        print(f"Loading {len(task_names)} tasks from pickle with {effective_workers} threads...")
    
    train_datasets = {}
    val_datasets = {}
    errors = []
    completed = 0
    
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {executor.submit(loader_fn, arg): arg[1] for arg in task_args}
        
        for future in as_completed(futures):
            task_name = futures[future]
            completed += 1
            try:
                name, train_ds, val_ds, error = future.result()
                if error:
                    errors.append(error)
                    print(f"[{completed}/{len(task_names)}] Error loading {task_name}: {error}")
                else:
                    train_datasets[name] = train_ds
                    val_datasets[name] = val_ds
                    if completed % 10 == 0 or completed == len(task_names):
                        print(f"[{completed}/{len(task_names)}] Loaded {len(train_datasets)} tasks...")
            except Exception as e:
                errors.append(str(e))
                print(f"[{completed}/{len(task_names)}] Exception loading {task_name}: {e}")
                # re-raise the exception
                raise e
    
    print(f"Successfully loaded {len(train_datasets)} tasks, {len(errors)} errors")
    return train_datasets, val_datasets


def get_tasks_from_env(env_name):
    if env_name == 'libero_90':
        x =  libero_task_map["libero_90"]
        x = [f"libero_90-{env}" for env in x]
        return x
    else:
        envs =  env_name.split('|')
        formatted = []
        for env in envs:
            pieces = env.split('-')
            if len(pieces) == 2:
                formatted.append(f"{pieces[0]}-{pieces[1]}")
            elif len(pieces) == 3:
                formatted.append(f"{pieces[0]}-{pieces[1].upper()}_{pieces[2]}")
            else:
                raise ValueError(f"Invalid env string: {env}")
        if not formatted:
            raise ValueError(f"No tasks found for env {env_name}")
        return formatted


# def process_task(task_paths, task_str, train_demo_nums, val_demo_nums, action_clip_eps=1e-5):
#     # initialize everything needed for train
#     train_observations = []
#     train_actions = []
#     train_next_observations = []
#     train_terminals = []
#     train_rewards = []
#     train_masks = []
#     train_successes = []

#     # initialize everything needed for val
#     val_observations = []
#     val_actions = []
#     val_next_observations = []
#     val_terminals = []
#     val_rewards = []
#     val_masks = []
#     val_successes = []

#     num_train_timesteps = 0
#     num_val_timesteps = 0
    
#     # Convert to sets for O(1) lookup
#     train_indices = set(train_demo_nums)
#     val_indices = set(val_demo_nums)
    
#     demo_num = -1
    
#     # REMOVE nested progress bar - just one
#     for path in tqdm.tqdm(sorted(task_paths), total=len(task_paths), desc=f"Processing {task_str}"):
#         with open(path, 'rb') as f:
#             dataset = pickle.load(f)
        
#         # REMOVE inner progress bar
#         for ep in dataset:
#             demo_num += 1
#             if demo_num not in train_indices and demo_num not in val_indices:
#                 continue
            
#             # COMBINE batch_dicts + astype into single operation
#             obs = ep['observations']
#             next_obs = ep['next_observations']
            
#             # If obs is a list of dicts, batch them efficiently
#             if isinstance(obs, list):
#                 # Stack and convert to float32 in one go
#                 obs_f32 = tree_util.tree_map(
#                     lambda *xs: np.stack(xs, axis=0).astype(np.float32), 
#                     *obs
#                 )
#                 next_obs_f32 = tree_util.tree_map(
#                     lambda *xs: np.stack(xs, axis=0).astype(np.float32), 
#                     *next_obs
#                 )
#             else:
#                 # If already batched, just convert
#                 obs_f32 = tree_util.tree_map(lambda x: x.astype(np.float32), obs)
#                 next_obs_f32 = tree_util.tree_map(lambda x: x.astype(np.float32), next_obs)
            
#             # Process actions - convert to float32 during clipping
#             a = np.clip(
#                 np.array(ep['actions']).astype(np.float32), 
#                 -1 + action_clip_eps, 
#                 1 - action_clip_eps
#             )
            
#             # Process other arrays - convert to float32 immediately
#             r = np.array(ep['rewards']).astype(np.float32)
#             dones = np.array(ep['dones']).astype(np.float32)
#             masks_f32 = 1.0 - dones
            
#             success = np.full(a.shape[0], np.max(r), dtype=np.float32)
            
#             # Append to either train or val
#             if demo_num in train_indices:
#                 num_train_timesteps += a.shape[0]
#                 train_observations.append(obs_f32)
#                 train_actions.append(a)
#                 train_rewards.append(r)
#                 train_next_observations.append(next_obs_f32)
#                 train_terminals.append(dones)
#                 train_masks.append(masks_f32)
#                 train_successes.append(success)
#             else:  # Must be in val_indices
#                 num_val_timesteps += a.shape[0]
#                 val_observations.append(obs_f32)
#                 val_actions.append(a)
#                 val_rewards.append(r)
#                 val_next_observations.append(next_obs_f32)
#                 val_terminals.append(dones)
#                 val_masks.append(masks_f32)
#                 val_successes.append(success)
    
#     print(f"Train timesteps: {num_train_timesteps}, Val timesteps: {num_val_timesteps}")
    
#     train_dataset = Dataset.create(
#         observations=stack_dict_list(train_observations),
#         next_observations=stack_dict_list(train_next_observations),
#         actions=np.concatenate(train_actions, axis=0),
#         rewards=np.concatenate(train_rewards, axis=0),
#         terminals=np.concatenate(train_terminals, axis=0),
#         masks=np.concatenate(train_masks, axis=0),
#         successes=np.concatenate(train_successes, axis=0),
#     )
    
#     val_dataset = Dataset.create(
#         observations=stack_dict_list(val_observations),
#         next_observations=stack_dict_list(val_next_observations),
#         actions=np.concatenate(val_actions, axis=0),
#         rewards=np.concatenate(val_rewards, axis=0),
#         terminals=np.concatenate(val_terminals, axis=0),
#         masks=np.concatenate(val_masks, axis=0),
#         successes=np.concatenate(val_successes, axis=0),
#     )
    
#     return train_dataset, val_dataset

# def process_task_hdf5(task_dir, task_str, train_demo_nums, val_demo_nums, action_clip_eps=1e-5):
#     """
#     Fast version that reads from a single HDF5 file per task.
    
#     Args:
#         task_dir: Path to task directory containing demos.hdf5
#         task_str: Task name for logging
#         train_demo_nums: List of demo indices for training
#         val_demo_nums: List of demo indices for validation
#     """
#     train_observations = []
#     train_actions = []
#     train_next_observations = []
#     train_terminals = []
#     train_rewards = []
#     train_masks = []
#     train_successes = []

#     val_observations = []
#     val_actions = []
#     val_next_observations = []
#     val_terminals = []
#     val_rewards = []
#     val_masks = []
#     val_successes = []

#     num_train_timesteps = 0
#     num_val_timesteps = 0
    
#     train_indices = set(train_demo_nums)
#     val_indices = set(val_demo_nums)
    
#     # Read from single HDF5 file
#     hdf5_path = Path(task_dir) / "demos.hdf5"
    
#     if not hdf5_path.exists():
#         raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    
#     with h5py.File(hdf5_path, 'r') as hdf_file:
#         data_group = hdf_file['data']
#         total_demos = hdf_file.attrs['num_demos']
        
#         # Process each demo
#         for demo_idx in range(total_demos):
#             if demo_idx not in train_indices and demo_idx not in val_indices:
#                 continue
            
#             demo_key = f'demo_{demo_idx}'
#             if demo_key not in data_group:
#                 print(f"Warning: {demo_key} not found in {hdf5_path}")
#                 continue
                
#             demo = data_group[demo_key]
            
#             # Load observations (already in dict of arrays format!)
#             obs_f32 = {k: np.array(demo[f'obs/{k}']).astype(np.float32) 
#                       for k in demo['obs'].keys()}
#             next_obs_f32 = {k: np.array(demo[f'next_obs/{k}']).astype(np.float32) 
#                            for k in demo['next_obs'].keys()}
            
#             # Load other arrays
#             a = np.clip(
#                 np.array(demo['actions']).astype(np.float32),
#                 -1 + action_clip_eps,
#                 1 - action_clip_eps
#             )
#             r = np.array(demo['rewards']).astype(np.float32)
#             dones = np.array(demo['dones']).astype(np.float32)
#             masks_f32 = 1.0 - dones
#             success = np.full(a.shape[0], np.max(r), dtype=np.float32)
            
#             # Append to appropriate dataset
#             if demo_idx in train_indices:
#                 num_train_timesteps += a.shape[0]
#                 train_observations.append(obs_f32)
#                 train_actions.append(a)
#                 train_rewards.append(r)
#                 train_next_observations.append(next_obs_f32)
#                 train_terminals.append(dones)
#                 train_masks.append(masks_f32)
#                 train_successes.append(success)
#             else:
#                 num_val_timesteps += a.shape[0]
#                 val_observations.append(obs_f32)
#                 val_actions.append(a)
#                 val_rewards.append(r)
#                 val_next_observations.append(next_obs_f32)
#                 val_terminals.append(dones)
#                 val_masks.append(masks_f32)
#                 val_successes.append(success)
    
#     print(f"Task {task_str}: Train timesteps: {num_train_timesteps}, Val timesteps: {num_val_timesteps}")
    
#     # Handle empty datasets
#     if not train_observations:
#         raise ValueError(f"No training data found for task {task_str}")
#     if not val_observations:
#         raise ValueError(f"No validation data found for task {task_str}")
    
#     train_dataset = Dataset.create(
#         observations=stack_dict_list(train_observations),
#         next_observations=stack_dict_list(train_next_observations),
#         actions=np.concatenate(train_actions, axis=0),
#         rewards=np.concatenate(train_rewards, axis=0),
#         terminals=np.concatenate(train_terminals, axis=0),
#         masks=np.concatenate(train_masks, axis=0),
#         successes=np.concatenate(train_successes, axis=0),
#     )
    
#     val_dataset = Dataset.create(
#         observations=stack_dict_list(val_observations),
#         next_observations=stack_dict_list(val_next_observations),
#         actions=np.concatenate(val_actions, axis=0),
#         rewards=np.concatenate(val_rewards, axis=0),
#         terminals=np.concatenate(val_terminals, axis=0),
#         masks=np.concatenate(val_masks, axis=0),
#         successes=np.concatenate(val_successes, axis=0),
#     )
    
#     return train_dataset, val_dataset

def get_loss_fn(network, batch, train, rng):    
    def loss_fn(grad_params):
        masked_actions = batch['actions'] * batch['masks'][..., None]
        batch_actions = jnp.reshape(masked_actions, (masked_actions.shape[0], -1))
        lang = batch['observations']['language']  # (batch, 768)
        
        # Positive examples: correct (obs, action, lang) triplets
        logits = network.select('classifier')(
            batch['observations'], batch_actions, lang,
            params=grad_params, train=train, rng=rng
        )  # (batch, 1)

        targets = batch['successes'][..., 0]
        # targets have shape(batch, seq_len), but the latter is redundant, since all transitions in a demo will have the same success, so just take the first one!
        
        # Binary cross-entropy: positives -> 1, negatives -> 0
        logits = logits.squeeze()
        targets = targets.squeeze()
        batch_loss  = optax.sigmoid_binary_cross_entropy(logits, targets)
        loss = jnp.mean(batch_loss)
        
        return loss, {
            'classifier_loss': loss,
        }
    return loss_fn

# JIT THE ACCURACY FUNCTION - CRITICAL FIX!
@jax.jit
def accuracy(network, batch, thresh=0.0):
    lang = batch['observations']['language']
    batch_masked_actions = batch['actions'] * batch['masks'][..., None]
    batch_actions = jnp.reshape(batch_masked_actions, (batch_masked_actions.shape[0], -1))
    
    # Forward pass
    logits = network.select('classifier')(
        batch['observations'], batch_actions, lang,
        train=False, params=network.params
    )
    
    targets = batch['successes'][..., 0]
    
    # Ensure shapes match - flatten both to 1D
    logits = logits.squeeze()
    targets = targets.squeeze()

    predictions = (logits >= 0).astype(jnp.float32)  # Use >= instead of >
    num_correct = jnp.sum(predictions == targets)
    num_total = logits.shape[0]   
    return num_correct, num_total
    

@jax.jit
def update(network, batch, rng):
    new_rng, rng = jax.random.split(rng)
    loss_fn = get_loss_fn(network, batch, True, rng)
    new_network, info = network.apply_loss_fn(loss_fn=loss_fn)
    network, rng = new_network, new_rng
    return network, rng, info

def main(flags):
    SEED = flags.seed
    random.seed(SEED)
    np.random.seed(SEED)
    
    # make save_dir
    time_suffix = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{flags.run_prefix}_{flags.task_name}_h{flags.horizon_length}_drop{flags.p_drop_state}_lr{flags.lr}_train{flags.train_fraction}_seed{SEED}_{time_suffix}"
    save_dir = Path(flags.save_dir) / flags.wandb_group / run_name
    if save_dir.exists():
        print(f"Saving classifier to {save_dir}, but it already exists. Deleting it...")
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'flags.json', 'w') as f:
        json.dump(flags.__dict__, f)
    
    wandb.init(
        entity="yajatyadav",
        project="multitask_RL",
        group=flags.wandb_group,
        name=run_name,
        config={
            "env_name": flags.env_name,
            "task_name": flags.task_name,
            "horizon_length": flags.horizon_length,
            "p_drop_state": flags.p_drop_state,
            "lr": flags.lr,
            "train_fraction": flags.train_fraction,
            "val_fraction": flags.val_fraction,
            "seed": SEED,
            "num_workers": flags.num_workers,
            "data_format": flags.data_format,
            "num_epochs": flags.num_epochs,
        }
    )

    
    rollouts = Path(flags.rollouts_dir)
    task_names = get_tasks_from_env(flags.env_name)
    
    # train_demo_nums = list(range(0, flags.num_train_demos))
    # val_demo_nums = list(range(flags.num_train_demos, flags.num_train_demos + flags.num_val_demos))

    # Use parallel loading for speed
    load_start = time.time()
    train_datasets, val_datasets = load_all_tasks_parallel(
        rollouts, task_names, flags.train_fraction, flags.val_fraction,
        SEED,
        flags.rollouts_per_task,
        num_workers=flags.num_workers,
        data_format=flags.data_format,
        data_from_date_before=flags.data_from_date_before,
    )
    load_time = time.time() - load_start
    print(f"Dataset loading completed in {load_time:.1f}s ({load_time/len(task_names):.2f}s per task)")
    
    if not train_datasets:
        raise ValueError("No training datasets loaded! Check your data directory.")
    train_dataset = MultiDatasetWrapper(list(train_datasets.values()), batch_level_sampling=True)


    # all hparams
    batch_size = 256
    horizon_length = flags.horizon_length
    discount = 0.99
    encoder = 'image_only_tiny'
    embed_dim = encoder_modules[encoder]().mlp_hidden_dims[-1]
    layer_norm = True
    lr  = flags.lr
    p_drop_state = flags.p_drop_state

    example_batch = train_dataset.sample_sequence(batch_size, sequence_length=horizon_length, discount=discount)
    ex_observations = example_batch['observations']
    ex_actions = example_batch['actions']
    ex_lang_embedding = ex_observations['language']  # Don't pop here
    full_actions = jnp.reshape(ex_actions, (ex_actions.shape[0], -1))
    lang_embedding_dim = ex_lang_embedding.shape[-1]
    print(f"lang_embedding_dim: {lang_embedding_dim}, action_dim: {full_actions.shape[-1]}")

    rng = jax.random.PRNGKey(SEED)
    val_rng = jax.random.PRNGKey(SEED + 100)
    rng, init_rng = jax.random.split(rng, 2)
    classifier_def = TransClassifier_BERT(
        vision_encoder=encoder_modules[encoder](),
        embed_dim=embed_dim,
        layer_norm=layer_norm,
        p_drop_state=p_drop_state,
    )

    network_info = dict(
        classifier=(classifier_def, (ex_observations, full_actions, ex_lang_embedding, True, init_rng)),
    )
    networks = {k: v[0] for k, v in network_info.items()}
    network_args = {k: v[1] for k, v in network_info.items()}
    network_def = ModuleDict(networks)
    network_tx = optax.adam(learning_rate=lr)
    network_params = network_def.init(init_rng, **network_args)['params']
    network = TrainState.create(network_def, network_params, tx=network_tx)

    
    build_network_tree(network.params)

    # Update wandb config with training details
    wandb.config.update({
        "batch_size": batch_size,
        "horizon_length": horizon_length,
        "discount": discount,
        "encoder": encoder,
        "embed_dim": embed_dim,
        "layer_norm": layer_norm,
        "lr": lr,
    })


    # start the main train loop!
    train_losses, val_losses = [], []
    per_task_val_losses = defaultdict(list)
    val_accuracies = []
    per_task_val_accuracies = defaultdict(list)
    grad_max, grad_min, grad_norm = [], [], []


    print(train_dataset.size)
    NUM_EPOCHS = flags.num_epochs
    VAL_INTERVAL = flags.val_interval  # INCREASED FROM 25 - less frequent validation
    NUM_VAL_TASKS = flags.num_val_tasks  # ONLY VALIDATE ON 10 RANDOM TASKS PER STEP
    if flags.save_every is None:
        SAVE_EVERY = train_dataset.size // (2 * batch_size)  # save roughly every half epoch
    else:
        SAVE_EVERY = flags.save_every
    num_train_steps = math.ceil(NUM_EPOCHS * train_dataset.size / batch_size)
    print(f"num_train_steps: {num_train_steps}")
    
    # Sample which tasks to use for validation
    val_task_names = list(val_datasets.keys())
    
    # for saving
    hparams = {
        'batch_size': batch_size,
        'horizon_length': horizon_length,
        'discount': discount,
        'encoder': encoder,
        'embed_dim': embed_dim,
        'layer_norm': layer_norm,
        'lr': lr,
        'p_drop_state': p_drop_state,
    }

    for step in tqdm.tqdm(range(1, num_train_steps+1), total=num_train_steps, desc="Training"):
        batch = train_dataset.sample_sequence(batch_size, sequence_length=horizon_length, discount=discount)
        network, rng, info = update(network, batch, rng)
        train_losses.append((step, info['classifier_loss']))
        grad_max.append((step, info['grad/max']))
        grad_min.append((step, info['grad/min']))
        grad_norm.append((step, info['grad/norm']))

        if step == 1:
            targets = batch['successes'][..., 0]
            print(f"Target mean: {np.mean(targets)}, shape: {targets.shape}")
            print(f"Actions norm: {np.mean(np.linalg.norm(batch['actions'], axis=-1))}")
            masks = batch['masks']
            print(f"Mask mean: {np.mean(masks)} (fraction non-terminal)")
            print(f"Loss value: {info['classifier_loss']}")
        
        # Log training metrics to wandb - LESS FREQUENTLY
        if (step == 1 or step % VAL_INTERVAL == 0 or step == num_train_steps):
            # Batch wandb logs to reduce API calls
            train_log = {
                'train/loss': float(info['classifier_loss']),
                'train/grad_max': float(info['grad/max']),
                'train/grad_min': float(info['grad/min']),
                'train/grad_norm': float(info['grad/norm']),
                'step': step,
            }
            
            val_losses_this_iter, val_accuracies_this_iter = [], []
            val_log = {}  # Batch all val logs together

            # ONLY VALIDATE ON A SUBSET OF TASKS, if len(val_task_names) < NUM_VAL_TASKS, validate on all tasks
            if len(val_task_names) < NUM_VAL_TASKS:
                sampled_val_tasks = val_task_names
            else:
                sampled_val_tasks = random.sample(val_task_names, NUM_VAL_TASKS)
            # print(f"Validating on {len(sampled_val_tasks)} tasks: {sampled_val_tasks}")

            for task_name in sampled_val_tasks:
                val_dataset = val_datasets[task_name]
                val_batch = val_dataset.sample_sequence(batch_size, sequence_length=horizon_length, discount=discount)
                val_rng, val_loss_rng, val_acc_rng = jax.random.split(val_rng, 3)
                loss_fn = get_loss_fn(network, val_batch, False, val_loss_rng)
                loss, val_info = loss_fn(network.params)

                # USE JITTED ACCURACY FUNCTION
                num_correct, num_total = accuracy(network, val_batch, thresh=0.0)
                
                # Convert to Python scalars
                task_loss = float(val_info['classifier_loss'])
                task_acc = float(num_correct / num_total)
                
                per_task_val_losses[task_name].append((step, task_loss))
                per_task_val_accuracies[task_name].append((step, task_acc))

                val_losses_this_iter.append(task_loss)
                val_accuracies_this_iter.append(task_acc)
                
                # Add to batch log
                val_log[f'val_per_task/{task_name}/loss'] = task_loss
                val_log[f'val_per_task/{task_name}/accuracy'] = task_acc
            
            avg_val_loss = np.mean(val_losses_this_iter)
            avg_val_acc = np.mean(val_accuracies_this_iter)
            val_losses.append((step, avg_val_loss))
            val_accuracies.append((step, avg_val_acc))
            
            # Add aggregate metrics to batch log
            val_log['val/loss'] = float(avg_val_loss)
            val_log['val/accuracy'] = float(avg_val_acc)
            val_log['step'] = step
            
            # SINGLE WANDB LOG CALL FOR ALL METRICS
            wandb.log({**train_log, **val_log}, step=step)

        # save every SAVE_EVERY steps or at the end of training
        if (SAVE_EVERY > 0 and (step % SAVE_EVERY == 0)) or (step == num_train_steps):
            print(f"Saving classifier at step {step}")
            save_classifier(network, step, save_dir, hparams)


    # save plotting data
    plotting_data = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'per_task_val_losses': per_task_val_losses,
        'per_task_val_accuracies': per_task_val_accuracies,
        'grad_max': grad_max,
        'grad_min': grad_min,
        'grad_norm': grad_norm,
    }

    with open(save_dir / 'plotting_data.pkl', 'wb') as f:
        pickle.dump(plotting_data, f)
    
    wandb.finish()

def setup_env_vars():
    sys.path.insert(0, os.getcwd())
    os.chdir('/home/yajatyadav/multitask_reinforcement_learning/multitask_RL')
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

    # ADD COMPILATION CACHE
    os.environ['XLA_FLAGS'] = (
        '--xla_gpu_triton_gemm_any=True '
        '--xla_gpu_autotune_level=2'  # Enable autotuning but cache results
    )

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
        os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--task_name', type=str, default='') # unused for now
    parser.add_argument('--run_prefix', type=str, required=True)
    parser.add_argument('--wandb_group', type=str, required=True)
    parser.add_argument('--rollouts_per_task', type=int, required=False, default=-1, help='Number of rollouts to use per task')

    
    parser.add_argument('--save_dir', type=str, required=False, default='/home/yajatyadav/multitask_reinforcement_learning/checkpoints/CLASSIFIERS/')
    parser.add_argument('--rollouts_dir', type=str, required=False, default='/home/yajatyadav/multitask_reinforcement_learning/multitask_RL/bcactor_collected_rollouts/bcflow_libero_90_25_demo_ckpt_80k')
    parser.add_argument('--language_embedder', type=str, default='bert')
    parser.add_argument('--batch_level_sampling', type=bool, default=True)
    parser.add_argument('--horizon_length', type=int, default=5)
    parser.add_argument('--p_drop_state', type=float, default=0.5)
    parser.add_argument('--num_train_demos', type=int, default=50)
    parser.add_argument('--num_val_demos', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=16, help='Number of parallel workers for data loading')
    parser.add_argument('--data_format', type=str, default='pickle', choices=['pickle', 'hdf5'],
                        help='Data format: pickle or hdf5')
   
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--train_fraction', type=float, default=0.8)
    parser.add_argument('--val_fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_from_date_before', type=str, default=None,
                        help='Only use trajs_*.pkl files with suffix date on or before this. Format: YYYYMMDD or YYYYMMDD_HHMMSS (matches collect_rollout_from_bcactor suffix).')
    parser.add_argument('--save_every', type=int, default=None)
    parser.add_argument('--val_interval', type=int, default=100)
    parser.add_argument('--num_val_tasks', type=int, default=10)
    flags = parser.parse_args()

    setup_env_vars()
    main(flags)