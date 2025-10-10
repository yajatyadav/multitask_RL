import jax
import numpy as np
import torch
from torch.utils.data import IterableDataset
import tensorflow as tf

from rlds_dataloader.dataset import RLDSDataset

import tensorflow_hub
import tensorflow_text

class MuseEmbedding:
    def __init__(self):
        self.muse_model = tensorflow_hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    
    def encode(self, strings):
        embeddings = self.muse_model(strings).numpy()
        return embeddings

class RLDSDataLoader():
    def __init__(self, config: dict, dataset: IterableDataset):
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
            self.text_encoder = MuseEmbedding()
    
    ## fix up the batch in a fromat downstream agents can more easily use
    def postprocess_batch(self, batch):
        task = batch["task"]
        action = batch["action"]
        reward = batch["reward"]
        is_terminal = batch["is_terminal"]
        curr_and_next_observation = batch["observation"]

        observation = tf.nest.map_structure(lambda x: x[:, 0], curr_and_next_observation)
        next_observation = tf.nest.map_structure(lambda x: x[:, 1], curr_and_next_observation)        
       

        # let's add task under observation and next_observation to be used in FiLM modulation of visual observations
        observation['task_embedding'] = self.text_encoder.encode(task['language_instruction'])
        next_observation['task_embedding'] = self.text_encoder.encode(task['language_instruction'])

        # squeeze out the window dimension from actions
        action = np.squeeze(action, axis=1)

        # make masks using is_terminals: 0 if terminal, 1 if not
        masks = np.where(is_terminal, 0, 1)

        ## map rewards: 0-> -1, 1 -> 0
        reward = np.where(reward == 0, -1, 0)

  

        batch = {
        "actions": action,
        "rewards": reward,
        "observations": observation,
        "next_observations": next_observation,
        "masks": masks,
        }

        return batch

    def example_batch(self):
        # creates a fresh iterator, and returns the first batch
        data_iter = iter(self._data_loader)
        return self.postprocess_batch(next(data_iter))

    
    def __iter__(self):
        while True:
            data_iter = iter(self._data_loader)
            print(f" 🧟‍♂️🧟‍♂️🧟‍♂️🧟‍♂️ Dataloader Exhuasted/First Epoch: creating a iterator over the dataloader. 🧟‍♂️🧟‍♂️🧟‍♂️🧟‍♂️")
            while True:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                yield self.postprocess_batch(batch)


def create_data_loader(
    config: dict,
    *,
    skip_norm_stats: bool = False,
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
    dataloader = RLDSDataLoader(config, dataset)

    return dataloader # data_loader = TorchDataLoader(


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *items)