from agents.iql import IQLAgent
from rlds_dataloader.dataloader import create_data_loader
import ml_collections
import optax
import tqdm
import numpy as np

if __name__ == "__main__":

    train_dataloader_config = {
        'data_root_dir': "/raid/users/yajatyadav/datasets/raw_libero/raw_libero_RLDS/",
        'dataset_mix': {
            'libero_90__black_bowl_on_plate_kitchen_scene1': 1.0,
        },
        'balance_datasets': True,
        'batch_size': 256,
        'num_workers': 16,
        'seed': 42,
        'do_image_aug': False,
        'binarize_gripper': True,
        'train': True
    }


    train_dataloader = create_data_loader(train_dataloader_config, skip_norm_stats=True)
    data_iter = iter(train_dataloader)
    example_batch = next(data_iter)

    agent_config = ml_collections.ConfigDict(
    dict(
        agent_name='iql',  # Agent name.
        lr=3e-4,
        actor_hidden_dims=(16,),  # Actor network hidden dimensions.
        value_hidden_dims=(16,),  # Value network hidden dimensions.
        layer_norm=True,  # Whether to use layer normalization.
        actor_layer_norm=True,  # Whether to use layer normalization for the actor.
        discount=0.99,  # Discount factor.
        tau=0.005,  # Target network update rate.
        expectile=0.9,  # IQL expectile.
        actor_loss='awr',  # Actor loss type ('awr' or 'ddpgbc').
        alpha=0.0,  # Temperature in AWR or BC coefficient in DDPG+BC.
        const_std=True,  # Whether to use constant standard deviation for the actor.
        encoder='state_space_encoder',  # Visual encoder name (None, 'impala_small', etc.).
    )
    )

    
    agent = IQLAgent.create(
        42, 
        example_batch['observations'],
        example_batch['actions'],
        agent_config,
    )

    actor = agent.network.select('actor')

    for batch in tqdm.tqdm(data_iter):
        actor_dist = actor(batch['observations'], params=agent.network.params)
        log_prob = actor_dist.log_prob(batch['actions'])
        # find indices where log_prob is nan
        nan_indices = np.where(np.isnan(log_prob))
        # print each action vector that caused a nan log_prob
        for index in nan_indices:
            print(batch['actions'][index])
        if not nan_indices:
            print("No nan log_probs found in this batch!")