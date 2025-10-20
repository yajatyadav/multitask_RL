"""
traj_transforms.py

Contains trajectory transforms used in the orca data pipeline. Trajectory transforms operate on a dictionary
that represents a single trajectory, meaning each tensor has the same leading dimension (the trajectory length).
"""

import logging
from typing import Dict

import tensorflow as tf


def chunk_act_obs(traj: Dict, window_size: int, future_action_window_size: int = 0) -> Dict:
    """
    Chunks actions and observations into the given window_size.

    Creates sliding windows where each window contains:
    - `window_size` observations: [s_t, s_{t+1}, ..., s_{t+window_size-1}]
    - `window_size - 1` actions: [a_t, a_{t+1}, ..., a_{t+window_size-2}]
    
    When windows extend past the trajectory end, the last state-action pair is repeated
    and marked as padding via "pad_mask".
    """
    traj_len = tf.shape(traj["action"])[0]
    action_dim = traj["action"].shape[-1]
    obs_chunk_indices = tf.broadcast_to(tf.range(0, window_size), [traj_len, window_size]) + tf.broadcast_to(
        tf.range(traj_len)[:, None], [traj_len, window_size]
    )

    action_chunk_indices = tf.broadcast_to(
        tf.range(0, window_size -1),
        [traj_len, window_size - 1],
    ) + tf.broadcast_to(
        tf.range(traj_len)[:, None],
        [traj_len, window_size - 1],
    )

    reward_chunk_indices = action_chunk_indices # reward field is also no longer a scalar, but a list of scalars

    capped_obs_indices = tf.minimum(obs_chunk_indices, traj_len - 1)
    capped_action_indices = tf.minimum(
        tf.maximum(action_chunk_indices, 0),
        traj_len - 1
    )
    capped_reward_indices = tf.minimum(
        tf.maximum(reward_chunk_indices, 0),
        traj_len - 1
    )
    # capped_chunk_indices = tf.minimum(chunk_indices, traj_len - 1)

    # if "timestep" in traj["task"]:
    #     goal_timestep = traj["task"]["timestep"]
    # else:
    #     goal_timestep = tf.fill([traj_len], traj_len - 1)

    # capped_action_chunk_indices = tf.minimum(
    #     tf.maximum(action_chunk_indices, 0), 
    #     goal_timestep[:, None]
    # )


    traj["observation"] = tf.nest.map_structure(lambda x: tf.gather(x, capped_obs_indices), traj["observation"])
    chunked_actions = tf.gather(traj["action"], capped_action_indices)

    actions_past_end = action_chunk_indices  >= traj_len

    if "absolute_action_mask" not in traj:
        raise ValueError(
            "No absolute_action_mask was provided. "
            "Cannot make neutral actions for actions past the end of the trajectory."
        )
    absolute_action_mask = traj.get("absolute_action_mask")

    # Gather the absolute_action_mask for chunked actions
    chunked_absolute_mask = tf.gather(absolute_action_mask, capped_action_indices)
    
    # Create neutral actions: absolute dims repeat last action, relative dims are zeroed
    neutral_actions = tf.where(
        chunked_absolute_mask,
        chunked_actions,  # absolute actions are repeated (already done during chunking)
        tf.zeros_like(chunked_actions),  # relative actions are zeroed
    )
    actions_past_end = action_chunk_indices >= traj_len
    traj["action"] = tf.where(actions_past_end[:, :, None], neutral_actions, chunked_actions)

    # setting padding
    traj["observation"]["pad_mask"] = obs_chunk_indices < traj_len
    traj["action_pad_mask"] = action_chunk_indices < traj_len

    return traj

    ## old implementation with goal timesteps

    # # if no absolute_action_mask was provided, assume all actions are relative
    # if "absolute_action_mask" not in traj and future_action_window_size > 0:
    #     logging.warning(
    #         "future_action_window_size > 0 but no absolute_action_mask was provided. "
    #         "Assuming all actions are relative for the purpose of making neutral actions."
    #     )
    # absolute_action_mask = traj.get("absolute_action_mask", tf.zeros([traj_len, action_dim], dtype=tf.bool))
    # neutral_actions = tf.where(
    #     absolute_action_mask[:, None, :],
    #     traj["action"],  # absolute actions are repeated (already done during chunking)
    #     tf.zeros_like(traj["action"]),  # relative actions are zeroed
    # )

    # # actions past the goal timestep become neutral
    # action_past_goal = action_chunk_indices > goal_timestep[:, None]
    # traj["action"] = tf.where(action_past_goal[:, :, None], neutral_actions, traj["action"])

    # return traj


def subsample(traj: Dict, subsample_length: int) -> Dict:
    """Subsamples trajectories to the given length."""
    traj_len = tf.shape(traj["action"])[0]
    if traj_len > subsample_length:
        indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
        traj = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)

    return traj


def add_pad_mask_dict(traj: Dict) -> Dict:
    """
    Adds a dictionary indicating which elements of the observation/task should be treated as padding.
        =>> traj["observation"|"task"]["pad_mask_dict"] = {k: traj["observation"|"task"][k] is not padding}
    """
    traj_len = tf.shape(traj["action"])[0]

    for key in ["observation", "task"]:
        pad_mask_dict = {}
        for subkey in traj[key]:
            # Handles "language_instruction", "image_*", and "depth_*"
            if traj[key][subkey].dtype == tf.string:
                pad_mask_dict[subkey] = tf.strings.length(traj[key][subkey]) != 0

            # All other keys should not be treated as padding
            else:
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)

        traj[key]["pad_mask_dict"] = pad_mask_dict

    return traj
