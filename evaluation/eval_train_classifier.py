import os, sys
os.chdir('/home/yajatyadav/multitask_reinforcement_learning/multitask_RL')
sys.path.insert(0, os.getcwd())
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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
import random

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field, restore_agent_with_file
import pickle
from utils.networks import MLP
from typing import Sequence


from envs.libero_utils import get_dataset as get_libero_dataset, make_env as make_libero_env
from evaluation_libero import evaluate
from utils.log_utils import get_wandb_video


class TransClassifier(nn.Module):
    hidden_dims: Sequence[int]
    encoder: nn.Module = None    
    layer_norm: bool = True

    def setup(self):
        self.classifier = MLP((*self.hidden_dims,), activate_final=False, layer_norm=self.layer_norm)
    
    def __call__(self, observations, actions):
        assert actions is not None, "Actions must be provided to the classifier"
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)
        return self.classifier(inputs)


train_percent, val_percent, test_percent = 0.85, 0.05, 0.1
tot_num_demos_per_task = 50
train_num_demos_per_task = int(train_percent * tot_num_demos_per_task)
val_num_demos_per_task = int(val_percent * tot_num_demos_per_task)
test_num_demos_per_task = tot_num_demos_per_task - train_num_demos_per_task - val_num_demos_per_task

# randomize, not simple arange
SEED = 0
random.seed(SEED)
np.random.seed(SEED)


shuffled_demo_nums = np.random.permutation(tot_num_demos_per_task)
train_demo_nums_to_use = list(shuffled_demo_nums[:train_num_demos_per_task])
val_demo_nums_to_use = list(shuffled_demo_nums[train_num_demos_per_task:train_num_demos_per_task + val_num_demos_per_task])
test_demo_nums_to_use = list(shuffled_demo_nums[train_num_demos_per_task + val_num_demos_per_task:])

keys_to_load = ['agentview_rgb', 'eye_in_hand_rgb', 'language']
env_name = 'libero_90-living_room_scene1'
task_name = 'pick_up_the_alphabet_soup_and_put_it_in_the_basket|pick_up_the_ketchup_and_put_it_in_the_basket'
augmentation_type = 'none'
augmentation_reward = False

train_dataset = get_libero_dataset(None, env_name, task_name, augmentation_type, augmentation_reward, keys_to_load, demo_nums_to_use_per_task=train_demo_nums_to_use, augmentation_dict=None)
print(f"main.py:Made env and datasets.Train dataset size: {train_dataset.size}", flush=True)
val_dataset = get_libero_dataset(None, env_name, task_name, augmentation_type, augmentation_reward, keys_to_load, demo_nums_to_use_per_task=val_demo_nums_to_use, augmentation_dict=None)
print(f"main.py:Made env and datasets.Val dataset size: {val_dataset.size}", flush=True)
test_dataset = get_libero_dataset(None, env_name, task_name, augmentation_type, augmentation_reward, keys_to_load, demo_nums_to_use_per_task=test_demo_nums_to_use, augmentation_dict=None)
print(f"main.py:Made env and datasets.Test dataset size: {test_dataset.size}", flush=True)



batch_size = 256
horizon_length = 5
discount = 0.99

example_batch = train_dataset.sample_sequence(batch_size, sequence_length=horizon_length, discount=discount)
ex_observations = example_batch['observations']
ex_actions = example_batch['actions']
full_actions = jnp.reshape(ex_actions, (ex_actions.shape[0], -1))
lang_embedding_dim = ex_observations['language'].shape[-1]

encoder = 'image_only_tiny'
hidden_dims = (128,)
layer_norm = True
lr  = 3e-4

rng = jax.random.PRNGKey(SEED)
rng, init_rng = jax.random.split(rng, 2)      

classifier_def = TransClassifier(
    encoder=encoder_modules[encoder](),
    hidden_dims=(*hidden_dims, lang_embedding_dim),
    layer_norm=layer_norm,
)
        
network_info = dict(
    classifier=(classifier_def, (ex_observations, full_actions)),
)

networks = {k: v[0] for k, v in network_info.items()}
network_args = {k: v[1] for k, v in network_info.items()}

network_def = ModuleDict(networks)
network_tx = optax.adam(learning_rate=lr)
network_params = network_def.init(init_rng, **network_args)['params']
network = TrainState.create(network_def, network_params, tx=network_tx)



train_losses, val_losses = [], []
val_accuracies = []
grad_max, grad_min, grad_norm = [], [], []
def get_loss_fn(batch):
    def loss_fn(grad_params):
        masked_actions = batch['actions'] * batch['masks'][..., None]
        batch_actions = jnp.reshape(masked_actions, (masked_actions.shape[0], -1))
        true_lang = batch['observations'].pop('language')
        pred_lang = network.select('classifier')(batch['observations'], batch_actions, params=grad_params)
        classifier_loss = jnp.mean(optax.losses.softmax_cross_entropy(pred_lang, true_lang))
        return classifier_loss, {
            'classifier_loss': classifier_loss,
        }
    return loss_fn

def accuracy(network, batch):
    true_lang = batch['observations'].pop('language')
    batch_masked_actions = batch['actions'] * batch['masks'][..., None]
    batch_actions = jnp.reshape(batch_masked_actions, (batch_masked_actions.shape[0], -1))
    pred_lang_logits = network.select('classifier')(batch['observations'], batch_actions, params=network.params)
    pred_lang = jax.nn.softmax(pred_lang_logits)
    # print(test_pred_lang.shape)
    # print(test_true_lang.shape)
    # argmax both
    pred_lang_argmax = jnp.argmax(pred_lang, axis=-1)
    true_lang_argmax = jnp.argmax(true_lang, axis=-1)
    num_correct = jnp.sum(pred_lang_argmax == true_lang_argmax)
    num_total = pred_lang_argmax.shape[0]
    return num_correct, num_total
    

@jax.jit
def update(network, batch, rng):
    new_rng, rng = jax.random.split(rng)
    loss_fn = get_loss_fn(batch)
    new_network, info = network.apply_loss_fn(loss_fn=loss_fn)
    network, rng = new_network, new_rng
    return network, rng, info

print(train_dataset.size)
NUM_EPOCHS = 4
num_train_steps = NUM_EPOCHS * math.ceil(train_dataset.size / batch_size)
VAL_INTERVAL = 10
for step in tqdm.tqdm(range(1, num_train_steps+1), total=num_train_steps, desc="Training"):
    batch = train_dataset.sample_sequence(batch_size, sequence_length=horizon_length, discount=discount)
    network, rng, info = update(network, batch, rng)
    train_losses.append((step, info['classifier_loss']))
    grad_max.append((step, info['grad/max']))
    grad_min.append((step, info['grad/min']))
    grad_norm.append((step, info['grad/norm']))
    if (VAL_INTERVAL > 0 and (step == 1 or step % VAL_INTERVAL == 0)):
        val_batch = val_dataset.sample_sequence(batch_size, sequence_length=horizon_length, discount=discount)
        val_batch_copy = copy.deepcopy(val_batch)
        loss_fn = get_loss_fn(val_batch)
        loss, info = loss_fn(network.params)
        num_correct, num_total = accuracy(network, val_batch_copy)
        val_losses.append((step, info['classifier_loss']))
        val_accuracies.append((step, num_correct / num_total))


# visualize train and val losses lists on the same plot, scale val by VAL_INTERVAL
import matplotlib.pyplot as plt


train_x_vals = [x[0] for x in train_losses]
train_y_vals = [x[1] for x in train_losses]
val_x_vals = [x[0] for x in val_losses]
val_y_vals = [x[1] for x in val_losses]
plt.plot(train_x_vals, train_y_vals, label='Train Loss')
plt.plot(val_x_vals, val_y_vals, label='Val Loss')
plt.legend()
plt.show()
print(f"Last train loss: {train_y_vals[-1]}")
print(f"Last val loss: {val_y_vals[-1]}")

# plot val_accuracies
plt.plot([x[1] for x in val_accuracies], label='Val Accuracy')
plt.legend()
plt.show()
print(f"Last val accuracy: {val_accuracies[-1][1]}")

# also plot grad_max, grad_min, grad_norm
plt.plot([x[1] for x in grad_max], label='Grad Max')
plt.plot([x[1] for x in grad_min], label='Grad Min')
plt.plot([x[1] for x in grad_norm], label='Grad Norm')
plt.legend()
plt.show()



# testing time, first by simply calling on test-batches, then visualizing!!
num_test_steps = 100 # dataset is infinite, but 100 should be good enough w/ a batch size of 256
test_num_correct, test_num_total = 0, 0
for i in tqdm.tqdm(range(num_test_steps), total=num_test_steps, desc="Testing"):
    test_batch = test_dataset.sample_sequence(batch_size, sequence_length=horizon_length, discount=discount)
    num_correct, num_total = accuracy(network, test_batch)
    test_num_correct += num_correct
    test_num_total += num_total
print(f"Test accuracy: {test_num_correct / test_num_total}, {test_num_correct}/{test_num_total}")





eval_env, names_to_return = make_libero_env(
            env_name, 
            task_name,
            num_parallel_envs=5, 
            use_hardcoded_eval_envs=False, 
            keys_to_load=keys_to_load + ['proprio'], # need to add proprio as old actors were trained with proprio
            seed=0,
        )


from agents.acifql import ACIFQLAgent, get_config as get_acifql_config

class ClassifierAgent:
    def __init__(self, network, actor_restore_path, example_batch, horizon_length, encoder):
        self.network = network
        self.actor_network = self.restore_actor_network(actor_restore_path, copy.deepcopy(example_batch), horizon_length)
        self.config = {"action_dim": example_batch["actions"].shape[-1], "horizon_length": horizon_length, "action_chunking": True, "encoder": encoder, "flow_steps": 10, "num_samples": 10}

    def restore_actor_network(self, actor_restore_path, example_batch, horizon_length):
        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']

        with open(actor_restore_path, 'rb') as f:
            load_dict = pickle.load(f)
        actor_agent_class = ACIFQLAgent
        actor_config = get_acifql_config()
        actor_config['encoder'] = 'combined_encoder_small'
        actor_config['horizon_length'] = horizon_length
        actor_agent = actor_agent_class.create(
            seed=SEED,
            ex_observations=copy.deepcopy(ex_observations),
            ex_actions=copy.deepcopy(ex_actions),
            config=actor_config,
        )
        loaded_actor_params = load_dict['agent']['network']['params']
        actor_agent_state_dict = flax.serialization.to_state_dict(actor_agent)
        actor_agent_params = actor_agent_state_dict['network']['params']

        for key in actor_agent_params:
            if 'actor' in key:
                # print(f"for key {key}, copying actor params from {config['actor_restore_path']}")
                actor_agent_params[key] = loaded_actor_params[key]
        actor_agent = flax.serialization.from_state_dict(actor_agent, actor_agent_state_dict)
        actor_network = actor_agent.network
        return actor_network
    
    def sample_actions(
        self,
        observations,
        rng=None,
        temperature=1.0,
    ):
        """Sample actions: actor_network generates actions, classifier used for rejection sampling."""

        if self.actor_network is None:
            raise ValueError("Actor network not found")

        full_action_dim = self.config["action_dim"] * (self.config["horizon_length"] if self.config["action_chunking"] else 1)
        k = sorted(observations.keys())[0]
        noises = jax.random.normal(
            rng,
            (
                observations[k].shape[0],
                self.config['num_samples'],
                full_action_dim,
            ),
        )
        observations = jax.tree_util.tree_map(
                    lambda x: jnp.repeat(x[:, None, ...], self.config["num_samples"], axis=1),
                    observations
                )
        actions = self.compute_flow_actions(observations, noises)
        actions = jnp.clip(actions, -1, 1)

        # Pick the action the classifier is most confident about, as measured by entropy
        pred_lang_logits = self.network.select('classifier')(observations, actions) # has shape (batch, num_samples, num_classes)
        # softmax-normalize and then compute entropy along axis=-1
        # More numerically stable entropy computation
        pred_lang_log_probs = jax.nn.log_softmax(pred_lang_logits, axis=-1)
        pred_lang_probs = jax.nn.softmax(pred_lang_logits, axis=-1)
        entropy = -jnp.sum(pred_lang_probs * pred_lang_log_probs, axis=-1)  # (batch, num_samples)
        
        indices = jnp.argmin(entropy, axis=-1)
        bshape = indices.shape
        indices = indices.reshape(-1)
        bsize = len(indices)
        actions = jnp.reshape(actions, (-1, self.config['num_samples'], full_action_dim))[jnp.arange(bsize), indices, :].reshape(
            bshape + (full_action_dim,))
        return actions

    
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.actor_network.select('actor_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.actor_network.select('actor_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions


actor_restore_path = 'exp/multitask_RL/bcflowactor_only/libero_90-living_room_scene1/bcflowactor_livingroomscene1__alphabet_soup_ketchup_25_demos_IMAGE_sd00020251227_144347/params_140000.pkl'
dataset_with_proprio = get_libero_dataset(None, env_name, task_name, augmentation_type, augmentation_reward, keys_to_load + ['proprio'], demo_nums_to_use_per_task=[1], augmentation_dict=None)
init_batch_for_actor = dataset_with_proprio.sample_sequence(1, sequence_length=horizon_length, discount=discount)
agent = ClassifierAgent(network, actor_restore_path, init_batch_for_actor, horizon_length, encoder)


# now let's try to do evaluation with this classifier network!!
# change spawn method
all_eval_info = []
per_eval_videos = {}
for j, eval_env_j in tqdm.tqdm(enumerate(eval_env), total=len(eval_env), desc="Evaluating multi-task", position=0,leave=False):
    eval_info, trajs, renders = evaluate(
        agent=agent,
        env=eval_env_j,
        action_dim=example_batch["actions"].shape[-1],
        num_eval_episodes=50,
        num_video_episodes=5,
        num_parallel_envs=5,
        video_frame_skip=3,
    )
    all_eval_info.append(eval_info)
    if len(renders) > 0:
        # value_and_reward_visualization(trajs, agent, FLAGS.save_dir, log_step)
        per_eval_videos[names_to_return[j]] = get_wandb_video(renders)

# aggregate eval info via mean, then log under "eval" prefix
mean_eval_info = {k: np.mean([eval_info[k] for eval_info in all_eval_info]) for k in all_eval_info[0].keys()}


# save mean_eval_info, all_eval_info, and per_eval_videos
with open('mean_eval_info.pkl', 'wb') as f:
    pickle.dump(mean_eval_info, f)
with open('all_eval_info.pkl', 'wb') as f:
    pickle.dump(all_eval_info, f)
with open('per_eval_videos.pkl', 'wb') as f:
    pickle.dump(per_eval_videos, f)