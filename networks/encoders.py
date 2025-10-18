import functools
from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp
import tensorflow as tf

import tensorflow_hub
import tensorflow_text

from networks.nets import MLP


class ResnetStack(nn.Module):
    """ResNet stack module."""

    num_features: int
    num_blocks: int
    max_pooling: bool = True

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding='SAME',
        )(x)

        if self.max_pooling:
            conv_out = nn.max_pool(
                conv_out,
                window_shape=(3, 3),
                padding='SAME',
                strides=(2, 2),
            )

        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)

            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)
            conv_out += block_input

        return conv_out


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float = None
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm: bool = False

    def setup(self):
        stack_sizes = self.stack_sizes
        self.stack_blocks = [
            ResnetStack(
                num_features=stack_sizes[i] * self.width,
                num_blocks=self.num_blocks,
            )
            for i in range(len(stack_sizes))
        ]
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        x = x.astype(jnp.float32) / 255.0

        conv_out = x

        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out, deterministic=not train)

        conv_out = nn.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm()(conv_out)
        out = conv_out.reshape((*x.shape[:-3], -1))

        out = MLP(self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm)(out)

        return out

# FiLM modulation of visual observations with language_embedding + proprioception
class FiLM(nn.Module):
    @nn.compact
    def __call__(self, obs, cond_vector):
        # resize to (1, cond_vector.shape[-1]) if necessary
        if cond_vector.ndim == 1:
            cond_vector = jnp.expand_dims(cond_vector, axis=0)
        num_channels = obs.shape[-1]
        gamma = nn.Dense(num_channels)(cond_vector)
        beta = nn.Dense(num_channels)(cond_vector)

        # to be broadcast to (batch, ., ., num_channels)
        gamma = gamma[:, None, None, :]
        beta = beta[:, None, None, :]
        return gamma * obs + beta

    
class FiLMImpalaEncoder(nn.Module):

    """IMPALA encoder using FiLM to fuse visual observations with oracle representations of goals."""
    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float = None
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm: bool = False

    def setup(self):
        stack_sizes = self.stack_sizes
        self.stack_blocks = [
            ResnetStack(
                num_features=stack_sizes[i] * self.width,
                num_blocks=self.num_blocks,
            )
            for i in range(len(stack_sizes))
        ]
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, obs, cond_vector, train=True):
        obs = obs.astype(jnp.float32) / 255.0

        conv_out = obs

        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out, deterministic=not train)
            conv_out = FiLM()(conv_out, cond_vector)

        conv_out = nn.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm()(conv_out)
        out = conv_out.reshape((*obs.shape[:-3], -1))

        out = MLP(self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm)(out)

        return out

class CombinedEncoder(nn.Module):
    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float = None
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm: bool = False

    @nn.compact
    def __call__(self, obs, train=True):
        # obs is a dict of batches, let's unpack it        

        task_embedding = obs['task_embedding']
        proprio = obs['proprio']

        image_primary = obs['image_primary']
        image_wrist = obs['image_wrist']
        
        
        # instead of calling impala twice, just stack images channel-wise and pass this 6-channel image to FilmImpalaEncoder
        stacked_images = jnp.concatenate([image_primary, image_wrist], axis=-1)

        out = FiLMImpalaEncoder(
            width=self.width, 
            stack_sizes=self.stack_sizes, 
            num_blocks=self.num_blocks, 
            dropout_rate=self.dropout_rate, 
            mlp_hidden_dims=self.mlp_hidden_dims, 
            layer_norm=self.layer_norm)(stacked_images, task_embedding, train=train)     
        combined_out = jnp.concatenate([out, proprio], axis=-1)
        return combined_out


class StateSpaceEncoder(nn.Module):
    @nn.compact
    def __call__(self, obs, train=True):
        out = obs['sim_state'][:, 1:] # remove the first dimension, which is the timestep
        return out
        
        
        


encoder_modules = {
    'impala': ImpalaEncoder,
    'impala_debug': functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    'impala_small': functools.partial(ImpalaEncoder, num_blocks=1),
    'impala_large': functools.partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),

    'film_impala_debug': functools.partial(FiLMImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    'combined_encoder_debug': functools.partial(CombinedEncoder, num_blocks=1, stack_sizes=(4, 4), mlp_hidden_dims=(128,)),
    'combined_encoder_small': functools.partial(CombinedEncoder, num_blocks=1),
    'combined_encoder_medium': functools.partial(CombinedEncoder, stack_sizes=(32, 64, 64), mlp_hidden_dims=(128,)),
    'combined_encoder_large': functools.partial(CombinedEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
    'state_space_encoder': StateSpaceEncoder,
}