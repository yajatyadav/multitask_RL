import jax
import numpy as np
import torch
from torch.utils.data import IterableDataset


from rlds_dataloader.dataset import RLDSDataset


class RLDSDataLoader():
    def __init__(self, config: dict, dataset: IterableDataset, num_batches: int | None = None):
            self._num_batches = num_batches
            generator = torch.Generator()
            generator.manual_seed(config["seed"])
            self._data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config["batch_size"] // jax.process_count(),
                shuffle=False, # not allowing shuffling here, since TFDS will handle it!!
                num_workers=0,  # causes conflicts with tfds's own multiprocessing logic...
                collate_fn=_collate_fn,
                generator=generator,
            )
    
    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            print(f" 🧟‍♂️🧟‍♂️🧟‍♂️🧟‍♂️ Dataloader Exhuasted/First Epoch: creating a new iterator over the dataloader. 🧟‍♂️🧟‍♂️🧟‍♂️🧟‍♂️")
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield batch


def create_data_loader(
    config: dict,
    *,
    skip_norm_stats: bool = False,
    num_batches: int | None = None,
) -> RLDSDataLoader:

    mixture_spec = [(repo_id, weight) for repo_id, weight in config["dataset_mix"].items()]
    action_key_list = [config.get("action_key", "action") for _ in range(len(mixture_spec))]
    dataset =  RLDSDataset(
        data_root_dir = config["data_root_dir"],
        mixture_spec = mixture_spec,
        action_key_list = action_key_list,
        binarize_gripper=config["binarize_gripper"],
        batch_size = config["batch_size"] // jax.process_count(),
        num_workers = config["num_workers"],
        action_horizon = 1,
        window_size = 2,
        shuffle_buffer_size = 100_000,
        skip_norm_stats = skip_norm_stats,
        train = config["train"],
        image_aug = config["do_image_aug"],
    )
    dataloader = RLDSDataLoader(config, dataset, num_batches)

    return dataloader # data_loader = TorchDataLoader(


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *items)