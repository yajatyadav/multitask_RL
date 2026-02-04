import os
import sys
sys.path.insert(0, os.getcwd())
os.chdir('/home/yajatyadav/multitask_reinforcement_learning/multitask_RL')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MUJOCO_GL'] = 'egl'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

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

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field, restore_agent_with_file
import pickle
from utils.networks import MLP
from typing import Sequence


from envs.libero_utils import get_dataset as get_libero_dataset, make_env as make_libero_env
from evaluation_libero import evaluate
from utils.log_utils import get_wandb_video

from envs.libero_utils import get_single_dataset
from libero.libero.benchmark.libero_suite_task_map import libero_task_map
from utils.log_utils import build_network_tree

SEED = 0
random.seed(SEED)
np.random.seed(SEED)

ENV_NAME_TO_EXPECTED_NUM_TASKS = {
    'libero_90': 74,
    'libero_90-living_room_scene1-pick_up_the_alphabet_soup_and_put_it_in_the_basket|libero_90-living_room_scene1-pick_up_the_ketchup_and_put_it_in_the_basket': 2,
}


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
        
        # Encode language -> (batch, 128)
        lang_encoded = self.lang_encoder(language_embedding)
        
        # Concatenate all three -> (batch, 384)
        inputs = jnp.concatenate([obs_encoded, action_encoded, lang_encoded], axis=-1)
        
        # Final classification -> (batch, 1)
        return self.classifier(inputs)


# train_percent, val_percent, test_percent = 0.9, 0.05, 0.05
# tot_num_demos_per_task = 50
# train_num_demos_per_task = int(train_percent * tot_num_demos_per_task)
# val_num_demos_per_task = int(val_percent * tot_num_demos_per_task)
# test_num_demos_per_task = tot_num_demos_per_task - train_num_demos_per_task - val_num_demos_per_task




# shuffled_demo_nums = np.random.permutation(tot_num_demos_per_task)
# train_demo_nums_to_use = list(shuffled_demo_nums[:train_num_demos_per_task])
# val_demo_nums_to_use = list(shuffled_demo_nums[train_num_demos_per_task:train_num_demos_per_task + val_num_demos_per_task])
# test_demo_nums_to_use = list(shuffled_demo_nums[train_num_demos_per_task + val_num_demos_per_task:])

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

def sample_negative_lang(batch, rng):
    """Sample negative embeddings - call BEFORE JIT-compiled functions."""
    lang_embedding = batch['observations']['language']
    batch_size = lang_embedding.shape[0]
    
    neg_embeddings = []
    for i in range(batch_size):
        key = tuple(np.array(lang_embedding[i]))
        candidates = other_lang_embeddings_map[key]
        idx = int(jax.random.randint(rng, (), 0, len(candidates)))
        rng, _ = jax.random.split(rng)
        neg_embeddings.append(candidates[idx])
    return jnp.array(neg_embeddings), rng

def get_loss_fn(batch, neg_lang, train, rng):    
    def loss_fn(grad_params):
        masked_actions = batch['actions'] * batch['masks'][..., None]
        batch_actions = jnp.reshape(masked_actions, (masked_actions.shape[0], -1))
        pos_lang = batch['observations'].pop('language')  # (batch, 768)
        
        pos_rng, neg_rng = jax.random.split(rng)
        # Positive examples: correct (obs, action, lang) triplets
        pos_logits = network.select('classifier')(
            batch['observations'], batch_actions, pos_lang,
            params=grad_params, train=train, rng=pos_rng
        )  # (batch, 1)
        
        # Negative examples: use pre-sampled embeddings
        neg_logits = network.select('classifier')(
            batch['observations'], batch_actions, neg_lang,
            params=grad_params, train=train, rng=neg_rng
        )  # (batch, 1)
        
        # Binary cross-entropy: positives -> 1, negatives -> 0
        pos_loss = optax.sigmoid_binary_cross_entropy(pos_logits, jnp.ones_like(pos_logits))
        neg_loss = optax.sigmoid_binary_cross_entropy(neg_logits, jnp.zeros_like(neg_logits))
        classifier_loss = jnp.mean(pos_loss) + jnp.mean(neg_loss)
        
        return classifier_loss, {
            'classifier_loss': classifier_loss,
            'pos_loss': jnp.mean(pos_loss),
            'neg_loss': jnp.mean(neg_loss),
        }
    return loss_fn

def accuracy(network, batch, rng):
    neg_lang, rng = sample_negative_lang(batch, rng)
    
    pos_lang = batch['observations'].pop('language')
    batch_masked_actions = batch['actions'] * batch['masks'][..., None]
    batch_actions = jnp.reshape(batch_masked_actions, (batch_masked_actions.shape[0], -1))
    
    # Positive examples
    pos_logits = network.select('classifier')(
        batch['observations'], batch_actions, pos_lang,
        train=False, params=network.params
    )
    
    # Negative examples
    neg_logits = network.select('classifier')(
        batch['observations'], batch_actions, neg_lang,
        train=False, params=network.params
    )
    
    # Correct if model assigns higher score to true lang than sampled neg lang
    num_correct = jnp.sum(pos_logits > neg_logits)
    num_total = pos_logits.shape[0]
    
    return num_correct, num_total
    

@jax.jit
def update(network, batch, neg_lang, rng):
    new_rng, rng = jax.random.split(rng)
    loss_fn = get_loss_fn(batch, neg_lang, True, rng)
    new_network, info = network.apply_loss_fn(loss_fn=loss_fn)
    network, rng = new_network, new_rng
    return network, rng, info


from pathlib import Path
import shutil
def main(flags):
    # make save_dir
    save_dir = Path(flags.save_dir)
    if save_dir.exists():
        print(f"Saving classifier to {save_dir}, but it already exists. Deleting it...")
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'flags.json', 'w') as f:
        json.dump(flags.__dict__, f)

    NUM_TRAIN_DEMOS = flags.NUM_TRAIN_DEMOS
    NUM_VAL_DEMOS = 50 - NUM_TRAIN_DEMOS
    train_demo_nums_to_use = list(range(NUM_TRAIN_DEMOS))
    val_demo_nums_to_use = list(range(NUM_TRAIN_DEMOS, NUM_TRAIN_DEMOS + NUM_VAL_DEMOS))
    keys_to_load = ['agentview_rgb', 'eye_in_hand_rgb', 'language']
    env_name = flags.env_name
    task_name = ''
    augmentation_type = 'none'
    augmentation_reward = False
    language_embedder = flags.language_embedder
    batch_level_sampling = flags.batch_level_sampling

    train_dataset = get_libero_dataset(None, env_name, task_name, language_embedder, augmentation_type, augmentation_reward, keys_to_load, batch_level_sampling=batch_level_sampling, demo_nums_to_use_per_task=train_demo_nums_to_use, augmentation_dict=None)
    print(f"main.py:Made env and datasets.Train dataset size: {train_dataset.size}", flush=True)
    val_dataset = get_libero_dataset(None, env_name, task_name, language_embedder, augmentation_type, augmentation_reward, keys_to_load, batch_level_sampling=batch_level_sampling, demo_nums_to_use_per_task=val_demo_nums_to_use, augmentation_dict=None)
    print(f"main.py:Made env and datasets.Val dataset size: {val_dataset.size}", flush=True)


    # all hparams
    batch_size = 256
    horizon_length = flags.horizon_length ## TODO(YY): change this to edit action chunking for the classifier...
    discount = 0.99
    encoder = 'image_only_tiny'
    embed_dim = encoder_modules[encoder]().mlp_hidden_dims[-1]
    layer_norm = True
    lr  = 3e-4
    p_drop_state = flags.p_drop_state

    example_batch = train_dataset.sample_sequence(batch_size, sequence_length=horizon_length, discount=discount)
    ex_observations = example_batch['observations']
    ex_actions = example_batch['actions']
    ex_lang_embedding = ex_observations.pop('language')
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


    BERT_HIDDEN_DIM = 768
    all_possible_lang_embeddings = set()
    for i in tqdm.tqdm(range(50), desc="Sampling lang embeddings"):
        batch = train_dataset.sample_sequence(batch_size, sequence_length=horizon_length, discount=discount)
        for lang_embedding in batch['observations']['language']:
            lang_embedding = tuple(lang_embedding)
            all_possible_lang_embeddings.add(lang_embedding)
    all_possible_lang_embeddings = [np.array(lang_embedding) for lang_embedding in all_possible_lang_embeddings]

    other_lang_embeddings_map = {tuple(lang_embedding): [] for lang_embedding in all_possible_lang_embeddings}
    for lang_embedding in tqdm.tqdm(other_lang_embeddings_map, total=len(other_lang_embeddings_map)):
        for other_lang_embedding in all_possible_lang_embeddings:
            if np.any(lang_embedding != other_lang_embedding):
                other_lang_embeddings_map[lang_embedding].append(other_lang_embedding)

    NUM_TASKS = ENV_NAME_TO_EXPECTED_NUM_TASKS[env_name]
    assert len(other_lang_embeddings_map) == NUM_TASKS, f"Expected {NUM_TASKS} lang embeddings, got {len(other_lang_embeddings_map)}"
    assert [len(x) == NUM_TASKS - 1 for x in other_lang_embeddings_map.values()], f"Expected {NUM_TASKS - 1} negative samples for each lang embedding, got {len(x)} for {lang_embedding}"



    # instantiate per-task data-iterators using val+test demo_nums_to_use, in 1 dataset object
    per_task_val_datasets = {}
    if env_name == 'libero_90':   
        envs_list = libero_task_map["libero_90"]
        envs_list = [f"libero_90-{env}" for env in envs_list]
    else:
        envs_list = env_name.split('|')
    print(f"env_name_list: {envs_list}")
    for env_name_i in envs_list:
        per_task_val_datasets[env_name_i] = get_single_dataset(None, env_name_i, task_name, language_embedder, augmentation_type, augmentation_reward, keys_to_load, demo_nums_to_use_per_task=val_demo_nums_to_use, augmentation_dict=None)



    train_losses, val_losses = [], []
    from collections import defaultdict
    per_task_train_losses, per_task_val_losses = defaultdict(list), defaultdict(list)
    val_accuracies = []
    per_task_val_accuracies = defaultdict(list)
    grad_max, grad_min, grad_norm = [], [], []
    per_task_grad_max, per_task_grad_min, per_task_grad_norm = defaultdict(list), defaultdict(list), defaultdict(list)


    print(train_dataset.size)
    NUM_EPOCHS = 10
    VAL_INTERVAL = 25
    SAVE_EVERY = 655
    num_train_steps = NUM_EPOCHS * math.ceil(train_dataset.size / batch_size)
    print(f"num_train_steps: {num_train_steps}")


    # Save
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
        neg_lang, rng = sample_negative_lang(batch, rng)
        network, rng, info = update(network, batch, neg_lang, rng)
        train_losses.append((step, info['classifier_loss'], info['pos_loss'], info['neg_loss']))
        grad_max.append((step, info['grad/max']))
        grad_min.append((step, info['grad/min']))
        grad_norm.append((step, info['grad/norm']))
        
        if (VAL_INTERVAL > 0 and (step == 1 or step % VAL_INTERVAL == 0)):
            val_batch = val_dataset.sample_sequence(batch_size, sequence_length=horizon_length, discount=discount)
            val_neg_lang, val_rng = sample_negative_lang(val_batch, val_rng)
            val_batch_copy = copy.deepcopy(val_batch)
            val_rng, val_loss_rng, val_acc_rng = jax.random.split(val_rng, 3)
            loss_fn = get_loss_fn(val_batch, val_neg_lang, False, val_loss_rng)
            loss, info = loss_fn(network.params)
            num_correct, num_total = accuracy(network, val_batch_copy, val_acc_rng)
            val_losses.append((step, info['classifier_loss'], info['pos_loss'], info['neg_loss']))
            val_accuracies.append((step, num_correct / num_total))


            # per-task val logging: we iterate through each val dataset, take a batch from it, compute loss / acc, and move on
            for eval_name in per_task_val_datasets:
                val_batch = per_task_val_datasets[eval_name].sample_sequence(batch_size, sequence_length=horizon_length, discount=discount)
                val_neg_lang, val_rng = sample_negative_lang(val_batch, val_rng)
                val_batch_copy = copy.deepcopy(val_batch)
                val_rng, val_loss_rng, val_acc_rng = jax.random.split(val_rng, 3)
                loss_fn = get_loss_fn(val_batch, val_neg_lang, False, val_loss_rng)
                loss, info = loss_fn(network.params)

                num_correct, num_total = accuracy(network, val_batch_copy, val_acc_rng)
                per_task_val_losses[eval_name].append((step, info['classifier_loss'], info['pos_loss'], info['neg_loss']))
                per_task_val_accuracies[eval_name].append((step, num_correct / num_total))

        # save every SAVE_EVERY steps or at the end of training
        if (SAVE_EVERY > 0 and ( step % SAVE_EVERY == 0)) or (step == num_train_steps):
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--language_embedder', type=str, default='bert')
    parser.add_argument('--batch_level_sampling', type=bool, default=True)
    parser.add_argument('--horizon_length', type=int, default=5)
    parser.add_argument('--p_drop_state', type=float, default=0.5)
    parser.add_argument('--NUM_TRAIN_DEMOS', type=int, default=25)
    flags = parser.parse_args()
    main(flags)