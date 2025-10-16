import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

from evaluation.eval_libero import evaluate, Args
from agents.iql import IQLAgent
from rlds_dataloader.dataloader import create_data_loader
import ml_collections
import optax
from utils.logger import setup_wandb, get_wandb_video
from rich.console import Console
from utils.logger import build_network_tree

import wandb
import tqdm
import jax
import numpy as np
from evaluation.eval_libero import supply_rng

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

train_dataloader_config = {
        'data_root_dir': "/raid/users/yajatyadav/datasets/raw_libero/raw_libero_RLDS/",
        'dataset_mix': {
            'libero_90__black_bowl_on_plate_kitchen_scene1': 1.0,
        },
        'balance_datasets': True,
        'batch_size': 4,
        'num_workers': 2,
        'seed': 42,
        'do_image_aug': False,
        'binarize_gripper': True,
        'train': True
    }

def test_spamming_actor():
    train_dataloader = create_data_loader(train_dataloader_config, skip_norm_stats=True)
    data_iter = iter(train_dataloader)
    example_batch = next(data_iter) 
    agent = IQLAgent.create(
        42, 
        example_batch['observations'],
        example_batch['actions'],
        agent_config,
    )
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    for i in tqdm.tqdm(range(300)):
        actor_fn(observations=example_batch['observations'], temperature=1.0)
    print("Spamming actor finished!")


def test_full():
    exp_name = "test_full"
    setup_wandb(project='multitask_RL', group='Debug', name=exp_name, log_flags=False)
    


    train_dataloader = create_data_loader(train_dataloader_config, skip_norm_stats=True)
    data_iter = iter(train_dataloader)
    example_batch = next(data_iter) 

    
    agent = IQLAgent.create(
        42, 
        example_batch['observations'],
        example_batch['actions'],
        agent_config,
    )

    # pretty print
    network = agent.network
    network_params = network.params
    console = Console()
    console.print(build_network_tree(network_params))

    # training loop
    for i in tqdm.tqdm(range(1, 1000)):
        batch = next(data_iter)
        agent, info = agent.update(batch)
        # log metrics
        if i % 100 == 0:
            train_metrics = {f'training/{k}': v for k, v in info.items()}
            wandb.log(train_metrics, step=i)

        
        if i == 10:
            renders = []
            wrist_renders = []
            eval_metrics = {}
            libero_eval_args = Args(
            seed=0,
            num_eval_episodes=5,
            num_steps_wait=10,
            video_frame_skip=3,
            eval_temperature=1.0,
            task_suite_name="libero_90",
            task_name="KITCHEN_SCENE1_put_the_black_bowl_on_the_plate",
            dataset_name="libero_90__black_bowl_on_plate_kitchen_scene1",
            )
            eval_info, trajs, cur_renders, cur_wrist_renders = evaluate(
                agent=agent,
                args=libero_eval_args,
            )
            renders.extend(cur_renders)
            wrist_renders.extend(cur_wrist_renders)
            for k, v in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = v            
            video = get_wandb_video(renders=renders)
            eval_metrics['video'] = video
            wandb.log(eval_metrics, step=i)
    
    print("Training finished!")


if __name__ == "__main__":
#    test_spamming_actor()
   test_full()