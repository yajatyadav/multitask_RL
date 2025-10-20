import numpy as np
from torch.utils.data import IterableDataset
import tensorflow as tf

from rlds_dataloader.dataset import RLDSDataset
from utils.data_utils import normalize_libero_batch
from utils.data_utils import get_language_encoder


class RLDSDataLoader():
    def __init__(self, config: dict, dataset: IterableDataset, load_images: bool, load_proprio: bool, load_language: bool, normalize_batches: bool = True):
        self.dataset = dataset
        self.text_encoder = get_language_encoder(config["text_encoder"]) if load_language else None
        self.config = config
        self.normalize_batches = normalize_batches
        self.load_images = load_images
        self.load_proprio = load_proprio
        self.load_language = load_language
    ## fix up the batch in a fromat downstream agents can more easily use
    def postprocess_batch(self, batch):
        task = batch["task"]
        action = batch["action"]
        reward = batch["reward"]
        # is_terminal = batch["is_terminal"]
        states_window = batch["observation"]

        # get the first and last observation from our sliding window
        observation = tf.nest.map_structure(lambda x: x[:, 0], states_window)
        next_observation = tf.nest.map_structure(lambda x: x[:, -1], states_window)

        if self.load_language:
            observation['task_embedding'] = self.text_encoder.encode(task['language_instruction'])
            next_observation['task_embedding'] = self.text_encoder.encode(task['language_instruction'])  

       

        # the mask is used in the loss-function to signal whether next_observation is truly a next_observation or just padding
        # thus, we can just set mask by using next_observation['pad_mask']
        mask = np.where(next_observation['pad_mask'], 1, 0)

        ## map rewards: 0-> -1, 1 -> 0
        reward = np.where(reward == 0, -1, 0)  

        batch = {
        "actions": action, # shape (B, W - 1, A)
        "rewards": reward, # shape (B, W - 1)
        "observations": observation, # ex: sim_state has shape (B, 45)
        "next_observations": next_observation, # same thing, (B, 45)
        "masks": mask, # shape (B,), used to mask next_observation in TD learning
        }
        
        if self.normalize_batches:
            batch = normalize_libero_batch(batch, dataset_name=list(self.config["dataset_mix"].keys())[0]) # TODO(YY): normalization done by picking the FIRST dataset in the mixture for now
        return batch

    
    def __iter__(self):
        print(f" 🧟‍♂️🧟‍♂️🧟‍♂️🧟‍♂️ Creating a iterator over the RLDS dataset. 🧟‍♂️🧟‍♂️🧟‍♂️🧟‍♂️") # we don't need the infinite logic, since the underlying RLDS dataset is already infinite!
        data_iter = iter(self.dataset)
        for batch in data_iter:
            yield self.postprocess_batch(batch)


def create_data_loader(
    config: dict,
    load_images: bool,
    load_proprio: bool,
    load_language: bool,
    *,
    normalize_images: bool = True,
    normalize_batches: bool = True,
    infinite_dataset: bool = True,
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
        prefetch_factor=config["prefetch_factor"],
        action_horizon = 1,
        window_size = config["window_size"],
        shuffle_buffer_size = 50_000,
        skip_norm_stats = True,
        train = config["train"],
        image_aug = config["do_image_aug"],
        infinite_dataset=infinite_dataset,
        load_images=load_images,
        load_proprio=load_proprio,
        load_language=load_language,
        normalize_images=normalize_images,
    )
    dataloader = RLDSDataLoader(config, dataset, load_images=load_images, load_proprio=load_proprio, load_language=load_language, normalize_batches=normalize_batches)

    return dataloader