import os
import json
import random
import time
from rich.console import Console

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
import ml_collections

from rlds_dataloader.dataloader import create_data_loader

from agents import agents

from utils.flax_utils import restore_agent, save_agent
from utils.logging import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb
from utils.logging import build_network_tree

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/iql.py', lock_config=False)

flags.DEFINE_string('data_root_dir', None, 'Data root directory.')
flags.DEFINE_string('train_dataset_mix', None, 'JSON string for the train dataset mix.')
flags.DEFINE_string('val_dataset_mix', None, 'JSON string for the val dataset mix.')
flags.DEFINE_integer('batch_size', 32, 'Batch size.')
flags.DEFINE_integer('num_workers', 16, 'Number of workers.')
flags.DEFINE_boolean('do_image_aug', True, 'Whether to apply image augmentation.')
flags.DEFINE_boolean('binarize_gripper', True, 'Whether to binarize the gripper.')

# utility func b/c openvla dataloader wants one dataset in a mixture to be the "primary" one and have weight 1.0
def ratios_to_odds_mixture(ratios):
    keys = sorted(ratios.keys())
    first_val = ratios[keys[0]]
    new_dict = {keys[0]: 1.0}
    for key in keys[1:]:
        new_dict[key] = ratios[key] / first_val
    return new_dict


agent_config = ml_collections.ConfigDict(
    dict(
        agent_name='iql',  # Agent name.
        lr=3e-4,  # Learning rate.
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


def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='multitask_RL', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    ## setup datasets
    train_dataset_mix = ratios_to_odds_mixture(json.loads(FLAGS.train_dataset_mix))
    val_dataset_mix = ratios_to_odds_mixture(json.loads(FLAGS.val_dataset_mix))
    print(f"train_dataset_mix: {train_dataset_mix}")
    print(f"val_dataset_mix: {val_dataset_mix}")
    train_dataloader_config = {
        'data_root_dir': FLAGS.data_root_dir,
        'dataset_mix': train_dataset_mix,
        'batch_size': FLAGS.batch_size,
        'num_workers': FLAGS.num_workers,
        'seed': FLAGS.seed,
        'do_image_aug': FLAGS.do_image_aug,
        'binarize_gripper': FLAGS.binarize_gripper,
        'train': True
    }
    val_dataloader_config = {
        'data_root_dir': FLAGS.data_root_dir,
        'dataset_mix': val_dataset_mix,
        'batch_size': FLAGS.batch_size,
        'num_workers': FLAGS.num_workers,
        'seed': FLAGS.seed,
        'do_image_aug': FLAGS.do_image_aug,
        'binarize_gripper': FLAGS.binarize_gripper,
        'train': False
    }
    

    train_dataloader = create_data_loader(train_dataloader_config)
    val_dataloader = create_data_loader(val_dataloader_config)
    example_batch = train_dataloader.example_batch()

    agent_class = agents[agent_config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        agent_config,
    )
    

    # restore path
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # train agent
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    # pretty-print networks' summary
    network = agent.network
    network_params = network.params
    console = Console()
    console.print(build_network_tree(network_params))

    step = 0
    done = True

    expl_metrics = dict()
    data_iter = iter(train_dataloader)
    val_data_iter = iter(val_dataloader)

    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1), smoothing=0.1, dynamic_ncols=True):
        batch = next(data_iter)
        agent, info = agent.update(batch)

        # log metrics
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in info.items()}
            if val_dataloader is not None:
                val_batch = next(val_data_iter)
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # evaluate agent
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
             # TODO evaluation not supported rught now
            pass
            # renders = []
            # eval_metrics = {}
            # for k, v in eval_info.items():
            #     eval_metrics[f'evaluation/{k}'] = v
            # if FLAGS.video_episodes > 0:
            #     video = get_wandb_video(renders=renders)
            #     eval_metrics['video'] = video
            # wandb.log(eval_metrics, step=i)
            # eval_logger.log(eval_metrics, step=i)

        # save agent
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)


    
    


