import copy
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field, restore_agent_with_file
import pickle
from utils.networks import MLP
from typing import Sequence
from agents.acifql import ACIFQLAgent, get_config as get_acifql_config


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




class ClassifierBestofNAgent(flax.struct.PyTreeNode):

    rng: Any
    network: Any
    actor_network: Any
    config: Any = nonpytree_field()


    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """

        actor_network = None
        if config['actor_restore_path'] != '':
            # TODO(YY): clean up later!! figure out how to save config as well without unmarking it as a non-pytree field...
            with open(config['actor_restore_path'], 'rb') as f:
                load_dict = pickle.load(f)
            actor_agent_class = ACIFQLAgent
            actor_config = get_acifql_config()
            
            actor_config['encoder'] = 'combined_encoder_small'
            actor_config['horizon_length'] = config['horizon_length']
            
            actor_agent = actor_agent_class.create(
                seed=seed,
                ex_observations=copy.deepcopy(ex_observations),
                ex_actions=copy.deepcopy(ex_actions),
                config=actor_config,
            )
            loaded_actor_params = load_dict['agent']['network']['params']

            # DEBUG: Compare param shapes between loaded and initialized
            print("=" * 60)
            print("DEBUG: Comparing loaded vs initialized actor param shapes")
            print("=" * 60)
            actor_agent_state_dict = flax.serialization.to_state_dict(actor_agent)
            actor_agent_params = actor_agent_state_dict['network']['params']
            
            # for key in sorted(loaded_actor_params.keys()):
            #     if 'actor' in key:
            #         print(f"\n--- {key} ---")
            #         def print_shapes(d, prefix=""):
            #             for k, v in d.items():
            #                 if isinstance(v, dict):
            #                     print_shapes(v, prefix + k + "/")
            #                 else:
            #                     loaded_shape = v.shape
            #                     init_shape = actor_agent_params.get(key, {})
            #                     # Navigate to same key in init params
            #                     init_v = actor_agent_params.get(key, {})
            #                     for part in (prefix + k).split("/"):
            #                         if part and isinstance(init_v, dict):
            #                             init_v = init_v.get(part, {})
            #                     init_shape_str = init_v.shape if hasattr(init_v, 'shape') else "NOT FOUND"
            #                     match = "✓" if (hasattr(init_v, 'shape') and loaded_shape == init_v.shape) else "✗ MISMATCH"
            #                     print(f"  {prefix}{k}: loaded={loaded_shape} vs init={init_shape_str} {match}")
            #         print_shapes(loaded_actor_params[key])
            # print("=" * 60)
            
            for key in actor_agent_params:
                if 'actor' in key:
                    print(f"for key {key}, copying actor params from {config['actor_restore_path']}")
                    actor_agent_params[key] = loaded_actor_params[key]
            actor_agent = flax.serialization.from_state_dict(actor_agent, actor_agent_state_dict)
            actor_network = actor_agent.network
        print(f"created the actor network!!")

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)      

        action_dim = ex_actions.shape[-1]

        if config["action_chunking"]:
            full_actions = jnp.reshape(ex_actions, (ex_actions.shape[0], -1))
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['classifier'] = encoder_module()

        # use ex_batch to figure out embedding dimension
        lang_embedding_dim = ex_observations['language'].shape[-1]
        
        classifier_def = TransClassifier(
            encoder=encoders.get('classifier'),
            hidden_dims=(*config['hidden_dims'], lang_embedding_dim),
            layer_norm=config['layer_norm'],
        )
        
        network_info = dict(
            classifier=(classifier_def, (ex_observations, full_actions)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if config["weight_decay"] > 0.:
            network_tx = optax.adamw(learning_rate=config['lr'], weight_decay=config["weight_decay"])
        else:
            network_tx = optax.adam(learning_rate=config[   'lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        config['action_dim'] = action_dim
        return cls(rng, network=network, actor_network=actor_network, config=flax.core.FrozenDict(**config))

    
    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :] # take the first action
        
        true_lang = batch['observations'].pop('language')
        pred_lang = self.network.select('classifier')(batch['observations'], batch_actions, params=grad_params)
        classifier_loss = jnp.mean(optax.losses.softmax_cross_entropy(pred_lang, true_lang))
        return classifier_loss, {
            'classifier_loss': classifier_loss,
        }

    @jax.jit
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

    @jax.jit
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
        

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='classifier_bestofn',
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256, 
            hidden_dims=(512,),
            layer_norm=True,
            encoder=ml_collections.config_dict.placeholder(str),
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=True,
            weight_decay=0.,
            num_samples=4,
            actor_restore_path=ml_collections.config_dict.placeholder(str),
            flow_steps=10,
        )
    )
    return config