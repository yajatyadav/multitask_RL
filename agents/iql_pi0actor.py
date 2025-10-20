import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from functools import partial
import time
from networks.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from networks.nets import Actor, Value, TransformedWithMode, Pi0Actor


class IQLPi0ActorAgent(flax.struct.PyTreeNode):
    """Implicit Q-learning (IQL) agent, which uses the Pi0 model as an actor - sampling is just best-of-N, ranked via Q-values, for now."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    pi0_actor: Pi0Actor = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the IQL value loss."""
        q1, q2 = self.network.select('target_critic')(batch['observations'], actions=batch['actions'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(batch['observations'], params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params):
        """Compute the IQL critic loss."""
        next_v = self.network.select('value')(batch['next_observations'])
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v

        q1, q2 = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        loss = value_loss + critic_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def get_advantages(self, obs, actions):
        """Takes a singular observation and multiple proposed actions, and returns the advantages of all these actions."""
        obs = jax.tree_util.tree_map(lambda x: jnp.stack([x.squeeze(0)] * self.config['pi0_best_of_n_samples']), obs)
        q1, q2 = self.network.select('target_critic')(obs, actions=actions)
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(obs)
        adv = q - v
        return adv

    ## TODO(YY): unfortunately, cannot jit the entire thing, b/c calling the pi0 module does some transforms (which cannot be jitted) before doing
    ## model.sample_actions() [which is JITTED]
    ## thus, this function calls a separate jitted function for passing through the value networks
    def sample_actions(
        self,
        value_obs,
        pi0_obs,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""

        # N samples from pi0-libero
        pi0_start_time = time.time()
        sampled_actions = []
        for _ in range(self.config['pi0_best_of_n_samples']):
            action = self.pi0_actor(pi0_obs) # does transforms, then model.sample_actions() (which is jitted)
            sampled_actions.append(action) # squueze needed so that concatenation with the encoder output works out in the value net
        sampled_actions = jnp.stack(sampled_actions)
        pi0_end_time = time.time()

        if sampled_actions.shape[0] == 1:
            # if only sampling 1 action chunk, just return it directly; avoids having to deal with potential shaping issues with the value networks
            best_action = sampled_actions[0]
            eval_info = {}
        else:
        ## TODO(YY): use temperature here to not always pick the 'best' action
            # rank via q and v networks
            q_start_time = time.time()
            adv = self.get_advantages(value_obs, sampled_actions)        
            best_action_idx = jnp.argmax(adv)
            best_action = sampled_actions[best_action_idx].reshape(1, -1)
            q_end_time = time.time()

            # build a eval actor info dict tracking relevant metrics!
            eval_info = {
                'sampling/adv_mean': adv.mean(),
                'sampling/adv_max': adv.max(),
                'sampling/adv_min': adv.min(),
                'sampling/adv_std': adv.std(),
                'time/q_time': q_end_time - q_start_time,
            }
        # add batch dim to reshape action into (1, action_dim), needed for downstream use in eval
        eval_info.update({
                'sampling/sampled_actions_mean': sampled_actions.mean(axis=0),
                'sampling/sampled_actions_max': sampled_actions.max(axis=0),
                'sampling/sampled_actions_min': sampled_actions.min(axis=0),
                'sampling/sampled_actions_std': sampled_actions.std(axis=0),
                'time/pi0_time': pi0_end_time - pi0_start_time,
        })
        return best_action, eval_info

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
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = encoder_module()
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()

        # Define networks.
        value_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
            encoder=encoders.get('value'),
        )
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        pi0_actor = Pi0Actor(
            checkpoint_dir=config['pi0_checkpoint_dir'],
            config_name=config['pi0_config_name'],
            action_horizon=config['pi0_action_horizon'],
            # num_samples=config['pi0_num_samples'], # commented out since batched-inference not supported on openpi end
        )

        network_info = dict(
            value=(value_def, (ex_observations,)),
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            # actor=(actor_def, (ex_observations,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config), pi0_actor=pi0_actor)


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='iql_pi0actor',  # Agent name.
            lr=3e-4,  # Learning rate.
            value_hidden_dims=(4,),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.      
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            encoder='state_space_encoder',  # Visual encoder name (None, 'impala_small', etc.).
            pi0_checkpoint_dir='../checkpoints/pi0_all_libero_but_10_flipped_train_split/pi0_all_libero_but_10_flipped_train_split__batch_64_steps_30k/10000',
            pi0_config_name='pi0_libero_mine', # TODO(YY): this currently has the extra delta_transform enabled since we trained our checkpoint with this
            # need to either retrain without it, or convert actions to relative during dataloading step!
            action_chunk_length=1,
            pi0_action_horizon=1,
            pi0_best_of_n_samples=1,
        )
    )
    return config