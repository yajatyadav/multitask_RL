import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs from https://github.com/huggingface/gym-aloha/tree/main?tab=readme-ov-file#-gpu-rendering-egl
import os
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import json
import random
import time
from rich.console import Console

import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
import ml_collections

from rlds_dataloader.dataloader import create_data_loader

from agents import agents

from utils.data_utils import ratios_to_odds_mixture
from utils.flax_utils import restore_agent, save_agent
from utils.logger import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb
from utils.logger import build_network_tree
from utils.logger import get_sample_input_output_log_to_wandb

from evaluation.eval_libero import evaluate, Args

FLAGS = flags.FLAGS

# housekeeping flags
flags.DEFINE_boolean('use_wandb', True, 'Whether to use wandb for logging.')
flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_string('exp_name_prefix', '', 'Prefix for the experiment name in Wandb, this can be used to group experiments.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

# training flags
flags.DEFINE_boolean('pixel_observations', False, 'Whether to use pixel observations.')
flags.DEFINE_integer('offline_steps', 1_000_000, 'Number of offline steps.')
flags.DEFINE_integer('log_interval', 100, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100_000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 10_000_000, 'Saving interval.')
flags.DEFINE_integer('num_input_output_to_log', 2, 'Number of transitions to log to wand, to serve as sanity-check.')

# eval flags
flags.DEFINE_integer('eval_episodes', 20, 'Number of evaluation episodes.')
flags.DEFINE_integer('num_steps_wait', 10, 'Number of steps to wait for objects to stabilize.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_float('eval_temperature', 1.0, 'Temperature for the actor.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
# more eval flags - might have to refactor tihs
flags.DEFINE_string('task_suite_name', '', 'Task suite name.')
flags.DEFINE_string('task_name', '', 'Task name.')

# dataset flags
flags.DEFINE_string('data_root_dir', None, 'Data root directory.')
flags.DEFINE_string('train_dataset_mix', None, 'JSON string for the train dataset mix.')
flags.DEFINE_boolean('do_validation', True, 'Whether to do validation.')
flags.DEFINE_string('val_dataset_mix', None, 'JSON string for the val dataset mix. Must be provided if do_validation is True.')
flags.DEFINE_boolean('balance_datasets', True, 'Whether to balance the datasets.') ## TODO(YY): NOTE- balance_datasets uses the size of the 'all' split, so sampling_weights are slightly off between train and val split
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_integer('num_workers', 16, 'Number of workers.')
flags.DEFINE_boolean('do_image_aug', True, 'Whether to apply image augmentation.')
flags.DEFINE_boolean('binarize_gripper', True, 'Whether to binarize the gripper into [-1, +1].')
flags.DEFINE_string('text_encoder', 'one_hot_libero', 'Text encoder type. Used if loading language instructions.')

config_flags.DEFINE_config_file('agent', 'agents/fql.py', lock_config=False)

# # agent config flags
# flags.DEFINE_boolean('tanh_squash', True, 'whether the actor squashes output actions using tanh')
# flags.DEFINE_boolean('state_dependent_std', True, 'whether the actor network also outputs multivariate gaussian STD per state')
# flags.DEFINE_boolean('const_std', True, 'whether we use const std of 1 for all actor output distributions. Importantly, if const_std=False, state_dependent_std, then we still learn a std shared among ALL states')
# flags.DEFINE_float('alpha', 10.0, 'the temperature param for AWR. alpha=0 means BC, and higher alpha leads us to more greedily picking higher-advantage actions')
# flags.DEFINE_float('expectile', 0.9, 'the expectile for IQL')
# flags.DEFINE_float('lr', 3e-4, 'constant learning rate that Adam uses for all networks')
# flags.DEFINE_string('obs_encoder', '', 'must specify what encoder- controls if we are using simulator state, image + proprio, or something else as our observation space')

# agent configuration
# agent_config = ml_collections.ConfigDict(
#     dict(
#         agent_name='iql',  # Agent name.
#         # lr=3e-4, # to be filled in later
#         actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
#         value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
#         layer_norm=True,  # Whether to use layer normalization.
#         actor_layer_norm=True,  # Whether to use layer normalization for the actor.
#         discount=0.99,  # Discount factor.
#         tau=0.005,  # Target network update rate.
#         # expectile=0.9,  # IQL expectile.
#         actor_loss='awr',  # Actor loss type ('awr' or 'ddpgbc').
#         # alpha=0.0,  # Temperature in AWR or BC coefficient in DDPG+BC. will be filled in main()
#         # const_std=True,  # Whether to use constant standard deviation for the actor. # to be filled in later
#         # encoder='state_space_encoder',  # Visual encoder name (None, 'impala_small', etc.).
#     )
#     )


def main(_):
    # setup wandb
    exp_name = FLAGS.exp_name_prefix + get_exp_name(FLAGS.seed)
    # setup wandb (waited till now since configs were updating) and start training loop
    if FLAGS.use_wandb:
        setup_wandb(project='multitask_RL', group=FLAGS.run_group, name=exp_name)

    # setup save dir
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, 'multitask_RL' if FLAGS.use_wandb else 'debug_no_wandb_dir', FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)
    
    # load in extra agent config options from flag, then save
    # this is wrong - the flags will be passed in as --agent.X
    # agent_config.update(
    #     lr=FLAGS.lr,
    #     alpha=FLAGS.alpha,
    #     expectile=FLAGS.expectile,
    #     tanh_squash=FLAGS.tanh_squash,
    #     const_std=FLAGS.const_std,
    #     state_dependent_std=FLAGS.state_dependent_std,
    #     encoder=FLAGS.obs_encoder
    # )
    # agent_config_dict = agent_config.to_dict()    
    # with open(os.path.join(FLAGS.save_dir, 'agent_config.json'), 'w') as f:
    #     json.dump(agent_config_dict, f, indent=2)   

    # setup randomization
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    ## setup dataloaders
    train_dataset_mix = ratios_to_odds_mixture(json.loads(FLAGS.train_dataset_mix))
    train_dataloader_config = {
        'data_root_dir': FLAGS.data_root_dir,
        'dataset_mix': train_dataset_mix,
        'balance_datasets': FLAGS.balance_datasets,
        'batch_size': FLAGS.batch_size,
        'num_workers': FLAGS.num_workers,
        "prefetch_factor": 4,
        'seed': FLAGS.seed,
        'do_image_aug': FLAGS.do_image_aug,
        'binarize_gripper': True,
        'train': True,
        'text_encoder': FLAGS.text_encoder,
    }
    train_dataloader = create_data_loader(
        config=train_dataloader_config,
        load_images=FLAGS.pixel_observations,
        load_proprio=FLAGS.pixel_observations,
        load_language=FLAGS.pixel_observations,
        normalize_images=True,
        normalize_batches=True,
        infinite_dataset=True,
    )
    data_iter = iter(train_dataloader)

    if FLAGS.do_validation:
        val_dataset_mix = ratios_to_odds_mixture(json.loads(FLAGS.val_dataset_mix))
        val_dataloader_config = {
            'data_root_dir': FLAGS.data_root_dir,
            'dataset_mix': val_dataset_mix,
            'balance_datasets': FLAGS.balance_datasets,
            'batch_size': FLAGS.batch_size,
            'num_workers': FLAGS.num_workers,
            'prefetch_factor': 4,
            'seed': FLAGS.seed,
            'do_image_aug': False,
            'binarize_gripper': True,
            'train': False,
            'text_encoder': FLAGS.text_encoder,
        } 
        val_dataloader = create_data_loader(
            config=val_dataloader_config,
            load_images=FLAGS.pixel_observations,
            load_proprio=FLAGS.pixel_observations,
            load_language=FLAGS.pixel_observations,
            normalize_images=True,
            normalize_batches=True,
            infinite_dataset=True,
        )
        val_data_iter = iter(val_dataloader)
    else:
        val_dataloader = None
        val_data_iter = None
    
    example_batch = next(data_iter)

    # setup agent
    agent_config = FLAGS.agent
    agent_class = agents[agent_config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        agent_config,
    )    

    # restore agent
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # setup loggers
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    # pretty-print networks' summary
    network = agent.network
    network_params = network.params
    console = Console()
    console.print(build_network_tree(network_params))

    # expl_metrics = dict()
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1), smoothing=0.1, dynamic_ncols=True):
        batch = next(data_iter)
        agent, info = agent.update(batch)
        # log few inputs to wandb: logs 0th transition in batch
        if i < FLAGS.num_input_output_to_log + 1:
            dict_to_log = get_sample_input_output_log_to_wandb(batch)
            wandb.log(dict_to_log, step=i)

        # log metrics, we can log more frequently at the beginning
        if (i < 50_000 and i % 1_000 == 0) or (i % FLAGS.log_interval == 0):
            train_metrics = {f'training/{k}': v for k, v in info.items()}
            if val_dataloader is not None:
                val_batch = next(val_data_iter)
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            if FLAGS.use_wandb:
                wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # evaluate agent
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            renders = []
            wrist_renders = []
            eval_metrics = {}
            libero_eval_args = Args(
                eval_use_images=FLAGS.pixel_observations, # set to False for state-based evals
                seed=FLAGS.seed,
                num_eval_episodes=FLAGS.eval_episodes if i > 1 else 5, # run 5 episodes for first time, as it's just a sanity check
                num_steps_wait=FLAGS.num_steps_wait,
                video_frame_skip=FLAGS.video_frame_skip,
                eval_temperature=FLAGS.eval_temperature,
                task_suite_name=FLAGS.task_suite_name,
                task_name=FLAGS.task_name,
                text_encoder=FLAGS.text_encoder,
                dataset_name=sorted(train_dataset_mix.keys())[0], # take the first key as the dataset name, used for normalizing eval-time observations
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
            # wrist_video = get_wandb_video(renders=wrist_renders)
            # eval_metrics['wrist_video'] = wrist_video
            if FLAGS.use_wandb:
                wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # save agent
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()
    print(f"🥳🥳Training finished!")


if __name__ == '__main__':
    app.run(main)