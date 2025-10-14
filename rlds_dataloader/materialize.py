"""
materialize.py

Factory class for initializing Open-X Embodiment dataset kwargs and other parameters; provides and exports functions for
clear control flow.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple
from functools import partial
import tensorflow as tf
from enum import IntEnum
# from prismatic.overwatch import initialize_overwatch
# from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS, ActionEncoding
# from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS

from rlds_dataloader.data_utils import NormalizationType
from rlds_dataloader.data_utils import binarize_gripper_actions
# Initialize Overwatch =>> Wraps `logging.Logger`
# overwatch = initialize_overwatch(__name__)

# Defines Proprioceptive State Encoding Schemes
class StateEncoding(IntEnum):
    # fmt: off
    NONE = -1               # No Proprioceptive State
    POS_EULER = 1           # EEF XYZ (3) + Roll-Pitch-Yaw (3) + <PAD> (1) + Gripper Open/Close (1)
    POS_QUAT = 2            # EEF XYZ (3) + Quaternion (4) + Gripper Open/Close (1)
    JOINT = 3               # Joint Angles (7, <PAD> if fewer) + Gripper Open/Close (1)
    JOINT_BIMANUAL = 4      # Joint Angles (2 x [ Joint Angles (6) + Gripper Open/Close (1) ])
    # fmt: on


# Defines Action Encoding Schemes
class ActionEncoding(IntEnum):
    # fmt: off
    EEF_POS = 1             # EEF Delta XYZ (3) + Roll-Pitch-Yaw (3) + Gripper Open/Close (1)
    JOINT_POS = 2           # Joint Delta Position (7) + Gripper Open/Close (1)
    JOINT_POS_BIMANUAL = 3  # Joint Delta Position (2 x [ Joint Delta Position (6) + Gripper Open/Close (1) ])
    EEF_R6 = 4              # EEF Delta XYZ (3) + R6 (6) + Gripper Open/Close (1)
    # fmt: on


## YY: this method is used with .filter(). Must return True for trajectories to be kept, False for trajectories to be filtered out
def zero_action_filter(traj, threshold=1e-3):
    """checks through traj[action] and filters out transitions whose action-chunk is near-idle"""
    actions = traj["action"]
    diffs = actions[1:] - actions[:-1]
    small_step = tf.reduce_all(tf.abs(diffs) < threshold, axis=1)
    is_idle = tf.reduce_any(small_step)
    return not is_idle

def flip_images_horizontally(traj):
    """flips the images horizontally"""
    traj["observation"]["image_primary"] = traj["observation"]["image_primary"][:, :, ::-1]
    traj["observation"]["image_wrist"] = traj["observation"]["image_wrist"][:, :, ::-1]
    return traj

OXE_DATASET_CONFIGS = {

    "libero_90_put_the_black_bowl_on_the_plate": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "action_encoding": ActionEncoding.EEF_POS,
    },

     "libero_90": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "action_encoding": ActionEncoding.EEF_POS,
    },

    "libero_10": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "action_encoding": ActionEncoding.EEF_POS,
    },

    "libero_object": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "action_encoding": ActionEncoding.EEF_POS,
    },

    "libero_spatial": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "action_encoding": ActionEncoding.EEF_POS,
    },

    "libero_goal": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "action_encoding": ActionEncoding.EEF_POS,
    },
}


def rand_swap_exterior_images(img1, img2):
    """
    Randomly swaps the two exterior images (for training with single exterior input).
    """
    return tf.cond(tf.random.uniform(shape=[]) > 0.5, lambda: (img1, img2), lambda: (img2, img1))

import random
def rand_pick_language_instruction(lang1, lang2, lang3):
    """
    Randomly picks one of the three language instructions.
    """
    return random.choice([lang1, lang2, lang3])

## TODO(YY): edit this DROID standardization function to set trajectory["language_instruction"] to randomly be one of the 3 language instructions
def droid_finetuning_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    DROID dataset transformation for actions expressed in *base* frame of the robot.
    """
    joint_velocity = trajectory["action_dict"]["joint_velocity"]
    gripper_position = trajectory["action_dict"]["gripper_position"]
    trajectory["action"] = tf.concat(
        (
            joint_velocity,
            gripper_position, ## TODO(YY)!!: not copying the OpenVLA code (1 - gripper_position), so that we remain consistent with the demo dataset.... but not sure if this is correct...
        ),
        axis=-1,
    )
    trajectory["observation"]["exterior_image_1_left"], trajectory["observation"]["exterior_image_2_left"] = (
        rand_swap_exterior_images(
            trajectory["observation"]["exterior_image_1_left"],
            trajectory["observation"]["exterior_image_2_left"],
        )
    )
    trajectory["language_instruction"] = rand_pick_language_instruction(
        trajectory["language_instruction"],
        trajectory["language_instruction_2"],
        trajectory["language_instruction_3"],
    )
    return trajectory



## YY: needed to rename "actions" to "action" LOL
def make_standardize_fn(dataset_name, action_key: str, binarize_gripper: bool = False):
    if dataset_name in ["droid_alt", "droid"]:
        return droid_finetuning_transform
    
    def standardize_fn(traj):
        if action_key in traj:
            traj["action"] = traj.pop(action_key)

        if binarize_gripper:
            traj["action"] = tf.concat(
            [
                traj["action"][:, :6],
                binarize_gripper_actions(traj["action"][:, -1], normalize=True)[:, None],
            ],
            axis=1,
            )
        return traj

    return standardize_fn


def make_oxe_dataset_kwargs(
    dataset_name: str,
    data_root_dir: Path,
    action_key: str,
    binarize_gripper: bool = False,
    load_camera_views: Tuple[str] = ("exterior_image_1", "exterior_image_2", "wrist_image"),
    load_depth: bool = True,
    load_proprio: bool = True,
    load_language: bool = True,
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
) -> Dict[str, Any]:
    """Generates config (kwargs) for given dataset from Open-X Embodiment."""
    dataset_kwargs = deepcopy(OXE_DATASET_CONFIGS[dataset_name])
    # if dataset_kwargs["action_encoding"] not in [ActionEncoding.EEF_POS, ActionEncoding.EEF_R6]:
    #     raise ValueError(f"Cannot load `{dataset_name}`; only EEF_POS & EEF_R6 actions supported!")

    # [Contract] For EEF_POS & EEF_R6 actions, only the last action dimension (gripper) is absolute!
    # Normalize all action dimensions *except* the gripper
    if dataset_kwargs["action_encoding"] is ActionEncoding.EEF_POS:
        dataset_kwargs["absolute_action_mask"] = [False] * 6 + [True]
        dataset_kwargs["action_normalization_mask"] = [True] * 6 + [False]
    # elif dataset_kwargs["action_encoding"] is ActionEncoding.EEF_R6:
        # dataset_kwargs["absolute_action_mask"] = [False] * 9 + [True]
        # dataset_kwargs["action_normalization_mask"] = [True] * 9 + [False]
    dataset_kwargs["action_proprio_normalization_type"] = action_proprio_normalization_type

    # Adjust Loaded Camera Views
    if len(missing_keys := (set(load_camera_views) - set(dataset_kwargs["image_obs_keys"]))) > 0:
        raise ValueError(f"Cannot load `{dataset_name}`; missing camera views `{missing_keys}`")

    # Filter
    dataset_kwargs["image_obs_keys"] = {
        k: v for k, v in dataset_kwargs["image_obs_keys"].items() if k in load_camera_views
    }
    # dataset_kwargs["depth_obs_keys"] = {
    #     k: v for k, v in dataset_kwargs["depth_obs_keys"].items() if k in load_camera_views
    # }

    # Eliminate Unnecessary Keys
    dataset_kwargs.pop("action_encoding")

    if not load_depth:
        dataset_kwargs.pop("depth_obs_keys")
    if not load_proprio:
        dataset_kwargs.pop("state_obs_keys")

    # Load Language
    if load_language:
        dataset_kwargs["language_key"] = "language_instruction"

    

    # Specify Standardization Transform
    dataset_kwargs["standardize_fn"] = make_standardize_fn(dataset_name,action_key, binarize_gripper)

    # Add any aux arguments
    if "aux_kwargs" in dataset_kwargs:
        dataset_kwargs.update(dataset_kwargs.pop("aux_kwargs"))

    return {"name": dataset_name, "data_dir": str(data_root_dir), **dataset_kwargs}


def get_oxe_dataset_kwargs_and_weights(
    data_root_dir: Path,
    mixture_spec: List[Tuple[str, float]],
    action_key_list: List[str],
    binarize_gripper: bool = False,
    load_camera_views: Tuple[str] = ("exterior_image_1", "exterior_image_2", "wrist_image"),
    load_depth: bool = True,
    load_proprio: bool = True,
    load_language: bool = True,
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
) -> Tuple[Dict[str, Any], List[float]]:
    """
    Generates dataset kwargs for a given dataset mix from the Open X-Embodiment dataset. The returned kwargs
    (per-dataset configs) and weights can be passed directly to `make_interleaved_dataset`.

    :param data_root_dir: Base directory containing RLDS/TFDS-formatted datasets (from Open-X)
    :param mixture_spec: List of (dataset_name, sampling_weight) from `oxe.mixtures.OXE_NAMED_MIXTURES`
    :param load_camera_views: Camera views to load; see `oxe.dataset_configs.py` for available views.
    :param load_depth: Load depth information in addition to camera RGB.
    :param load_proprio: Load proprioceptive state.
    :param load_language: Load language instructions.
    :param action_proprio_normalization_type: Normalization scheme to use for proprioceptive actions.

    return: Tuple of (per_dataset_kwargs, sampling_weights)
    """
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight in mixture_spec:
        if d_name in included_datasets:
            # overwatch.warning(f"Skipping Duplicate Dataset: `{(d_name, d_weight)}`")
            continue

        included_datasets.add(d_name)
        filtered_mixture_spec.append((d_name, d_weight))

    # Assemble Dataset Config (kwargs) and Weights
    per_dataset_kwargs, sampling_weights = [], []
    for i, (d_name, d_weight) in enumerate(filtered_mixture_spec):
        try:
            per_dataset_kwargs.append(
                make_oxe_dataset_kwargs(
                    d_name,
                    data_root_dir,
                    action_key_list[i],
                    binarize_gripper,
                    load_camera_views,
                    load_depth,
                    load_proprio,
                    load_language,
                    action_proprio_normalization_type,
                )
            )
            sampling_weights.append(d_weight)

        except ValueError as e:
            # overwatch.warning(f"Skipping `{d_name}` due to Error: {e}")
            pass
    return per_dataset_kwargs, sampling_weights