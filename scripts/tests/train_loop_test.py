import rlds_dataloader.dataloader as rlds_data_loader
import tqdm
# default is mean/std action-proprio normalization. Actions only first 6 dimensions normalized
train_dataloader_config = {
    "data_root_dir": "/raid/users/yajatyadav/datasets/raw_libero/raw_libero_RLDS",
    "dataset_mix": { 
        "libero_90": 1.0, # weights are in terms of odds
        "libero_object": 0.08,
        "libero_spatial": 0.08,
        "libero_goal": 0.08,
    },
    "batch_size": 32,
    "num_workers": 16, # dataloader workers
    "seed": 42,
    "do_image_aug": True,
    "binarize_gripper": True, # binarizes to 0 and 1 (by scanning for transitions), and then normalizes to -1 and 1
    "train": True, # for picking the right split
}

# val_dataloader_config = {
#     "data_root_dir": "/raid/users/yajatyadav/datasets/raw_libero/raw_libero_RLDS",
#     "dataset_mix": { 
#         "libero_90": 1.0, # weights are in terms of odds
#         "libero_object": 0.08,
#         "libero_spatial": 0.08,
#         "libero_goal": 0.08,
#     },
#     "batch_size": 32,
#     "num_workers": 16, # dataloader workers
#     "seed": 42,
#     "do_image_aug": True,
#     "binarize_gripper": True, # binarizes to 0 and 1 (by scanning for transitions), and then normalizes to -1 and 1
#     "train": False, # for picking the right split
# }

train_dataloader = rlds_data_loader.create_data_loader(train_dataloader_config)
# val_dataloader = rlds_data_loader.create_data_loader(val_dataloader_config)


from agents.iql import IQLAgent, get_config
import ml_collections
config = ml_collections.ConfigDict(
        dict(
            agent_name='iql',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=32,  # Batch size.
            actor_hidden_dims=(64, 64),  # Actor network hidden dimensions.
            value_hidden_dims=(64, 64),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            actor_loss='awr',  # Actor loss type ('awr' or 'ddpgbc').
            alpha=10.0,  # Temperature in AWR or BC coefficient in DDPG+BC.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            encoder='combined_encoder_debug',  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
example_batch = train_dataloader.example_batch()

agent = IQLAgent.create(
    42,
    example_batch['observations'],
    example_batch['actions'],
    config,
)

# train agent
step = 0
done = True
# only offline!
data_iter = iter(train_dataloader)
for step in tqdm.tqdm(range(1_000)):
    batch = next(data_iter)
    agent, info = agent.update(batch)

    # log
    train_metrics = {f'training/{k}': v for k, v in info.items()}
    print(train_metrics)


