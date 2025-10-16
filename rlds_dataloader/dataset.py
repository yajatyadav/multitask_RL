"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""


from pathlib import Path
from typing import Any, Dict, Tuple

from torch.utils.data import IterableDataset
from typing import List

from rlds_dataloader.rlds_dataset import make_interleaved_dataset
from rlds_dataloader.materialize import get_oxe_dataset_kwargs_and_weights
from rlds_dataloader.data_utils import NormalizationType

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100
WORKER_SCALE_FACTOR = 4
from utils.data_utils import LIBERO_ENV_RESOLUTION

class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        mixture_spec: List[Tuple[str, float]],
        action_key_list: List[str],
        binarize_gripper: bool,
        balance_datasets: bool,
        batch_size: int,
        num_workers: int,
        action_horizon: int,
        window_size: int,
        valid_episodes: List[int] = [],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        infinite_dataset: bool = True,
        image_aug: bool = True,
        skip_norm_stats: bool = True,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir = data_root_dir
        self.action_horizon = action_horizon
        self.batch_size = batch_size
        self.num_workers = num_workers

        action_proprio_normalization_type = None if skip_norm_stats else NormalizationType.NORMAL



        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            action_key_list=action_key_list,
            binarize_gripper=binarize_gripper,
            load_camera_views=("primary", "secondary", "wrist"),
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=action_proprio_normalization_type,
        )

        
        
        rlds_config = dict(
            batch_size=self.batch_size,
            traj_transform_kwargs=dict(
                window_size=window_size,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=0,                        # For action chunking
                skip_unlabeled=False,                                # Skip trajectories without language labels
                goal_relabeling_strategy=None,                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size={"image_primary": (LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION), "image_wrist": (LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION)},
                num_parallel_calls=self.num_workers,                       # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=balance_datasets,
            # traj_transform_threads= self.num_workers * len(mixture_spec),
            # traj_read_threads= WORKER_SCALE_FACTOR * len(mixture_spec),
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
            infinite_dataset=infinite_dataset,
        )
        
        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        
        # fmt: on
        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield rlds_batch

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")

