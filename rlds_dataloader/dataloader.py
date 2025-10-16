import numpy as np
from torch.utils.data import IterableDataset
import tensorflow as tf

from rlds_dataloader.dataset import RLDSDataset
from utils.data_utils import normalize_libero_batch
from utils.data_utils import MuseEmbedding


class RLDSDataLoader():
    def __init__(self, config: dict, dataset: IterableDataset, normalize_batches: bool = True):
            self.dataset = dataset
            self.text_encoder = MuseEmbedding
            self.config = config
            self.normalize_batches = normalize_batches
    
    ## fix up the batch in a fromat downstream agents can more easily use
    def postprocess_batch(self, batch):
        task = batch["task"]
        action = batch["action"]
        reward = batch["reward"]
        is_terminal = batch["is_terminal"]
        curr_and_next_observation = batch["observation"]

        observation = tf.nest.map_structure(lambda x: x[:, 0], curr_and_next_observation)
        next_observation = tf.nest.map_structure(lambda x: x[:, 1], curr_and_next_observation)      
        observation['task_embedding'] = self.text_encoder.encode(task['language_instruction'])
        next_observation['task_embedding'] = self.text_encoder.encode(task['language_instruction'])  

        # squeeze out the window dimension from actions
        action = np.squeeze(action, axis=1)
        # action = normalize_action(action)
        # normalize each action in batch
        # action = np.apply_along_axis(normalize_action, axis=1, arr=action)        

        # make masks using is_terminals: 0 if terminal, 1 if not
        masks = np.where(is_terminal, 0, 1)

        ## map rewards: 0-> -1, 1 -> 0
        reward = np.where(reward == 0, -1, 0)
       

        # let's add task under observation and next_observation to be used in FiLM modulation of visual observations
        # observation['image_primary'] = normalize_image(observation['image_primary'])
        # observation['image_wrist'] = normalize_image(observation['image_wrist'])
        # observation['proprio'] = normalize_proprio(observation['proprio'])
        # next_observation['image_primary'] = normalize_image(next_observation['image_primary'])
        # next_observation['image_wrist'] = normalize_image(next_observation['image_wrist'])
        # next_observation['proprio'] = normalize_proprio(next_observation['proprio'])

        

        # let's explicitly convert obs image to uint8, and also proprio normalize
        # observation['image_primary'] = np.apply_along_axis(normalize_image, axis=1, arr=observation['image_primary'])
        # observation['image_wrist'] = np.apply_along_axis(normalize_image, axis=1, arr=observation['image_wrist'])
        # observation['proprio'] = np.apply_along_axis(normalize_proprio, axis=1, arr=observation['proprio'])
        
        # next_observation['image_primary'] = np.apply_along_axis(normalize_image, axis=1, arr=next_observation['image_primary'])
        # next_observation['image_wrist'] = np.apply_along_axis(normalize_image, axis=1, arr=next_observation['image_wrist'])
        # next_observation['proprio'] = np.apply_along_axis(normalize_proprio, axis=1, arr=next_observation['proprio'])

  

        batch = {
        "actions": action,
        "rewards": reward,
        "observations": observation,
        "next_observations": next_observation,
        "masks": masks,
        }
        if self.normalize_batches:
            batch = normalize_libero_batch(batch, dataset_name=list(self.config["dataset_mix"].keys())[0]) # right now, we only support one dataset in the mixture
        return batch

    # def example_batch(self):
    #     # creates a fresh iterator, and returns the first batch
    #     data_iter = iter(self.dataset)
    #     return self.postprocess_batch(next(data_iter))

    
    def __iter__(self):
        print(f" 🧟‍♂️🧟‍♂️🧟‍♂️🧟‍♂️ Creating a iterator over the dataloader. 🧟‍♂️🧟‍♂️🧟‍♂️🧟‍♂️") # we don't need the infinite logic, since the underlying RLDS dataset is already infinite!
        data_iter = iter(self.dataset)
        for batch in data_iter:
            yield self.postprocess_batch(batch)


def create_data_loader(
    config: dict,
    *,
    skip_norm_stats: bool = True,
    infinite_dataset: bool = True,
    normalize_batches: bool = True,
) -> RLDSDataLoader:

    mixture_spec = [(repo_id, weight) for repo_id, weight in config["dataset_mix"].items()]
    action_key_list = [config.get("action_key", "action") for _ in range(len(mixture_spec))]
    dataset =  RLDSDataset(
        data_root_dir = config["data_root_dir"],
        mixture_spec = mixture_spec,
        action_key_list = action_key_list,
        binarize_gripper=config["binarize_gripper"],
        balance_datasets=config["balance_datasets"],
        batch_size = config["batch_size"],
        num_workers = config["num_workers"],
        action_horizon = 1,
        window_size = 2,
        shuffle_buffer_size = 100_000,
        skip_norm_stats = skip_norm_stats,
        train = config["train"],
        image_aug = config["do_image_aug"],
        infinite_dataset=infinite_dataset,
    )
    dataloader = RLDSDataLoader(config, dataset, normalize_batches=normalize_batches)

    return dataloader # data_loader = TorchDataLoader(


# def _collate_fn(items):
#     """Collate the batch elements into batched numpy arrays."""
#     # Make sure to convert to numpy arrays before stacking since some of the incoming elements
#     # may be JAX arrays.
#     return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *items)