import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

from evaluation.eval_libero import evaluate, Args
from agents.iql import IQLAgent
from rlds_dataloader.dataloader import create_data_loader
import ml_collections
import optax

if __name__ == "__main__":

    train_dataloader_config = {
        'data_root_dir': "/raid/users/yajatyadav/datasets/raw_libero/raw_libero_RLDS",
        'dataset_mix': {
            'libero_90': 1.0,
            'libero_object': 1.0,
        },
        'balance_datasets': True,
        'batch_size': 32,
        'num_workers': 16,
        'seed': 42,
        'do_image_aug': True,
        'binarize_gripper': True,
        'train': True
    }


    train_dataloader = create_data_loader(train_dataloader_config, skip_norm_stats=True)
    example_batch = train_dataloader.example_batch()

    agent_config = ml_collections.ConfigDict(
    dict(
        agent_name='iql',  # Agent name.
        optimizer=optax.contrib.muon,
        lr=optax.warmup_cosine_decay_schedule(
            init_value=3e-4 / 1000,
            peak_value=3e-4,
            warmup_steps=1000,
            decay_steps=100000,
            end_value=3e-4 / 100000,
        ),
        batch_size=256,  # Batch size.
        actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
        value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
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

    
    agent = IQLAgent.create(
        42, 
        example_batch['observations'],
        example_batch['actions'],
        agent_config,
    )

    args = Args(
        seed=0,
        num_eval_episodes=20,
        num_steps_wait=10,
        video_frame_skip=3,
        eval_temperature=1.0,
        task_suite_name="libero_10",
        task_name="LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    )
    evaluate(agent, args)