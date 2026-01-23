import os
import socket
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import glob, tqdm, wandb, os, json, random, time, jax
from absl import app, flags
from ml_collections import config_flags
from utils.log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger, get_sample_input_output_log_to_wandb, get_wandb_video, build_network_tree
from rich.console import Console
from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from envs.robomimic_utils import is_robomimic_env
from envs.libero_utils import is_libero_env

from plotting.plot_Q_value_visuals import value_and_reward_visualization
from utils.flax_utils import save_agent
from evaluation.brc_eval_scripts.generate_eval_sbatch import generate_sbatch_script
from utils.datasets import Dataset, ReplayBuffer
# from evaluation import evaluate as evaluate_others
from evaluation_libero import evaluate as evaluate_libero
from agents import agents
import numpy as np
import json
import subprocess

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

flags.DEFINE_string('exp_name_prefix', '', 'Experiment name prefix.')
flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'libero_90-kitchen_scene2', 'Environment (dataset) name.')
flags.DEFINE_string('task_name', '', 'Task name.')
flags.DEFINE_bool('use_hardcoded_eval_envs', False, 'Whether to use hardcoded eval environments.')
flags.DEFINE_string('augmentation_type', 'none', 'Augmentation type: none (no aug), task (for task only), first (for first task in scene), exhaustive (for all tasks in all scenes from env).')
flags.DEFINE_float('augmentation_reward', 0.0, 'The reward for relabeled trajectories. This is the value before the -1 shift is applied!')
flags.DEFINE_string('augmentation_dict', '{}', 'Augmentation dictionary: {task_name: [augment_tasks]}.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_integer('num_demos_to_use_per_task', -1, 'Number of demos to use per task.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('num_input_output_to_log', 3, 'Number of transitions to log to wandb.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')

flags.DEFINE_integer('save_interval', -1, 'Save interval.')
flags.DEFINE_integer('start_training', 5000, 'when does training start')

flags.DEFINE_boolean('use_pixels', False, 'Whether to use pixels as observations during training and evaluation.')
flags.DEFINE_boolean('use_proprio', False, 'Whether to use EEF proprio as observations during training and evaluation.')
flags.DEFINE_boolean('use_mj_sim_state', False, 'Whether to use MJ sim state as observations during training and evaluation.')
flags.DEFINE_boolean('use_language', False, 'Whether to use language as observations during training and evaluation.')
flags.DEFINE_float('p_aug', 0.0, 'Image augmentation probability for training dataset.')
flags.DEFINE_boolean('use_negative_rewards', False, 'Whether to use -1/0 rewarding for training dataset.')

flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")

flags.DEFINE_float('discount', 0.99, 'discount factor')

flags.DEFINE_list('eval_hosts', ['savio', 'brc', 'cluster'], 
                  'List of hostname keywords that trigger automatic evaluation on BRC')
flags.DEFINE_string('eval_actor_restore_path', None, 'Path to actor checkpoint for evaluation.')
flags.DEFINE_multi_integer('eval_best_of_N_vals', [1, 2, 4, 8, 16, 32, 64, 128],
                          'List of n values to evaluate for best-of-N.')
flags.DEFINE_boolean('eval_best_of_N_brc', True, 'Whether to evaluate best-of-N on BRC.')
flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('num_parallel_envs', 5, 'Number of parallel environments for evaluation.')
flags.DEFINE_integer('video_episodes', 5, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/acifql.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for large datasets because of memory constraints')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')

flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward: prevents reward values like -2 (will be set to -1)")

flags.DEFINE_bool('save_all_online_states', False, "save all trajectories to npy")

class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)

def main(_):
    hostname = socket.gethostname()
    print(f"main.py:😈😈😈 Starting main.py on host {hostname}")
    exp_name = FLAGS.exp_name_prefix + get_exp_name(FLAGS.seed)
    run = setup_wandb(entity='yajatyadav', project='multitask_RL', group=FLAGS.run_group, name=exp_name)
    
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent
    augmentation_dict = json.loads(FLAGS.augmentation_dict)
    print(f"main.py:😈😈😈 Augmentation dictionary: {augmentation_dict}")

    if FLAGS.num_demos_to_use_per_task != -1:
        print(f"main.py:😈😈😎 ONLY using {FLAGS.num_demos_to_use_per_task} demos for each task")
        demo_nums_to_use_per_task = list(range(FLAGS.num_demos_to_use_per_task))
    else:
        demo_nums_to_use_per_task = None
    
    # data loading
    if FLAGS.ogbench_dataset_dir is not None:
        # custom ogbench dataset
        assert FLAGS.dataset_replace_interval != 0
        assert FLAGS.dataset_proportion == 1.0
        dataset_idx = 0
        dataset_paths = [
            file for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")) if '-val.npz' not in file
        ]
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
        )
    else:
        keys_to_load = []
        if FLAGS.use_pixels:
            keys_to_load.extend(['agentview_rgb', 'eye_in_hand_rgb']) # use 2 cam images
        if FLAGS.use_proprio:
            keys_to_load.extend(['proprio'])
            # keys_to_load.extend(['obs/ee_pos', 'obs/ee_ori', 'obs/gripper_states']) # use EEF proprio
        if FLAGS.use_language:
            keys_to_load.extend(['language']) # use language
        if FLAGS.use_mj_sim_state:
            keys_to_load.extend(['states']) # use MJ sim state
        env, eval_env, train_dataset, val_dataset, names_to_return = make_env_and_datasets(
            FLAGS.env_name,
            FLAGS.task_name,
            FLAGS.augmentation_type,
            FLAGS.augmentation_reward,
            num_parallel_envs=FLAGS.num_parallel_envs,
            keys_to_load=keys_to_load,
            use_hardcoded_eval_envs=FLAGS.use_hardcoded_eval_envs,
            demo_nums_to_use_per_task=demo_nums_to_use_per_task,
            augmentation_dict=augmentation_dict,
        )
    
    print(f"main.py:Made env and datasets.Train dataset size: {train_dataset.size}", flush=True)

    # house keeping
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    log_step = 0
    
    discount = FLAGS.discount
    config["horizon_length"] = FLAGS.horizon_length

    # handle dataset
    def process_train_dataset(ds):
        """
        Process the train dataset to 
            - handle dataset proportion
            - handle sparse reward
            - convert to action chunked dataset
        """

        ds = Dataset.create(**ds)
        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
            ds = Dataset.create(
                **{k: v[:new_size] for k, v in ds.items()}
            )
        
        if FLAGS.use_negative_rewards: # use -1/0 rewarding: the sparse reward is set to -1 if the reward is not 0
            print(f"main.py:Translating dataset rewards by -1")
            penalty_rewards = ds["rewards"] - 1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = penalty_rewards
            ds = Dataset.create(**ds_dict)
        
        if FLAGS.sparse:
            print(f"main.py: Sparsifiying rewards by setting all non-zero rewards to -1")
            # Create a new dataset with modified rewards instead of trying to modify the frozen one
            sparse_rewards = (ds["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = sparse_rewards
            ds = Dataset.create(**ds_dict)

        if FLAGS.p_aug > 0.0:
            ds.p_aug = FLAGS.p_aug

        return ds
    
    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample_sequence(config['batch_size'], sequence_length=FLAGS.horizon_length, discount=discount)
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Setup logging.
    prefixes = ["env", "eval"] + [f"eval_{names_to_return[i]}" for i in range(len(names_to_return))]
    print(f"Logging prefixes ARE: {prefixes}")
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")

    logger = LoggingHelper(
        csv_loggers={prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv")) 
                    for prefix in prefixes},
        wandb_logger=wandb,
    )
    network = agent.network
    network_params = network.params
    console = Console()
    console.print(build_network_tree(network_params))

    # set up which eval function to use
    if is_libero_env(FLAGS.env_name):
        evaluate = evaluate_libero
    else:
        raise NotImplementedError("No other eval function implemented yet.")
    
    # Offline RL
    # times_to_log_inputs = list(range(0, 1000, 1000 // FLAGS.num_input_output_to_log))
    print(f"Starting training loop.")
    offline_init_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1)): # make message last batch's loss
        log_step += 1

        # if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0:
        #     dataset_idx = (dataset_idx + 1) % len(dataset_paths)
        #     print(f"Using new dataset: {dataset_paths[dataset_idx]}", flush=True)
        #     train_dataset, val_dataset = make_ogbench_env_and_datasets(
        #         FLAGS.env_name,
        #         dataset_path=dataset_paths[dataset_idx],
        #         compact_dataset=False,
        #         dataset_only=True,
        #         cur_env=env,
        #     )
        #     train_dataset = process_train_dataset(train_dataset)

        batch = train_dataset.sample_sequence(config['batch_size'], sequence_length=FLAGS.horizon_length, discount=discount)
        agent, offline_info = agent.update(batch)

        if i == 1 or i % FLAGS.log_interval == 0:
            logger.log(offline_info, "offline_agent", step=log_step)

        # if i in times_to_log_inputs:
        #     to_log = get_sample_input_output_log_to_wandb(batch)
        #     logger.log(to_log, "offline_agent", step=log_step)
        
        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)
            
            # if we are saving and on BRC, no cost to evaluate in background as well
            if any(host in hostname for host in FLAGS.eval_hosts) and FLAGS.eval_best_of_N_brc:
                print(f"main.py:😈😈😈 Evaluating best-of-N in background w/ n values: {FLAGS.eval_best_of_N_vals}")
                n_vals = FLAGS.eval_best_of_N_vals
                assert FLAGS.eval_actor_restore_path is not None
                actor_restore_path = FLAGS.eval_actor_restore_path
                critic_restore_path = os.path.join(FLAGS.save_dir,f"params_{log_step}.pkl")
                wandb_run_id = run.id
                wandb_name = 'TRAIN_EVAL_' + exp_name
                output_file = generate_sbatch_script(
                    n_vals=n_vals,
                    actor_restore_path=actor_restore_path,
                    critic_restore_path=critic_restore_path,
                    env_name=FLAGS.env_name,
                    task_name=FLAGS.task_name,
                    wandb_name=wandb_name,
                    wandb_run_id=wandb_run_id,
                    output_file_dir='scripts/shell_scripts/train_eval/'
                )
                print(f"main.py:😈😈😈 Output file: {output_file}")
                os.chmod(output_file, 0o755)
                result = subprocess.run([output_file], 
                       capture_output=True, 
                       text=True,
                       env=os.environ.copy(),
                       cwd=os.getcwd()
                       )
                print(f"Return code: {result.returncode}")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")


                

        # eval: do one at very start, very end, and in b/w using eval_interval. but if eval_interval is -1, we skip evaling
        if (FLAGS.eval_interval != -1) and (i == FLAGS.offline_steps or i == 5 or i % FLAGS.eval_interval == 0):
            # during eval, the action chunk is executed fully


            all_eval_info = []
            for j, eval_env_j in tqdm.tqdm(enumerate(eval_env), total=len(eval_env), desc="Evaluating multi-task", position=0,leave=False):
                eval_info, trajs, renders = evaluate(
                    agent=agent,
                    env=eval_env_j,
                    action_dim=example_batch["actions"].shape[-1],
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    num_parallel_envs=FLAGS.num_parallel_envs,
                    video_frame_skip=FLAGS.video_frame_skip,
                )
                all_eval_info.append(eval_info)
                if len(renders) > 0:
                    # value_and_reward_visualization(trajs, agent, FLAGS.save_dir, log_step)
                    eval_info['video'] = get_wandb_video(renders)
                logger.log(eval_info, f"eval_{names_to_return[j]}", step=log_step)
                # remove video before taking mean
                if 'video' in eval_info:
                    del eval_info['video']

            # aggregate eval info via mean, then log under "eval" prefix
            eval_info = {k: np.mean([eval_info[k] for eval_info in all_eval_info]) for k in all_eval_info[0].keys()}
            logger.log(eval_info, "eval", step=log_step)
    
    offline_end_time = time.time()
    print(f"Offline training time: {offline_end_time - offline_init_time} seconds", flush=True)

    # transition from offline to online
    # replay_buffer = ReplayBuffer.create_from_initial_dataset(
    #     dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
    # )
        
    # ob, _ = env.reset()
    
    # action_queue = []
    # action_dim = example_batch["actions"].shape[-1]

    # # Online RL
    # update_info = {}

    # from collections import defaultdict
    # data = defaultdict(list)
    # online_init_time = time.time()
    # for i in tqdm.tqdm(range(1, FLAGS.online_steps + 1)):
    #     log_step += 1
    #     online_rng, key = jax.random.split(online_rng)
        
    #     # during online rl, the action chunk is executed fully
    #     if len(action_queue) == 0:
    #         action = agent.sample_actions(observations=ob, rng=key)

    #         action_chunk = np.array(action).reshape(-1, action_dim)
    #         for action in action_chunk:
    #             action_queue.append(action)
    #     action = action_queue.pop(0)
        
    #     next_ob, int_reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated

    #     if FLAGS.save_all_online_states:
    #         state = env.get_state()
    #         data["steps"].append(i)
    #         data["obs"].append(np.copy(next_ob))
    #         data["qpos"].append(np.copy(state["qpos"]))
    #         data["qvel"].append(np.copy(state["qvel"]))
    #         if "button_states" in state:
    #             data["button_states"].append(np.copy(state["button_states"]))
        
    #     # logging useful metrics from info dict
    #     env_info = {}
    #     for key, value in info.items():
    #         if key.startswith("distance"):
    #             env_info[key] = value
    #     # always log this at every step
    #     logger.log(env_info, "env", step=log_step)

    #     if 'antmaze' in FLAGS.env_name and (
    #         'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
    #     ):
    #         # Adjust reward for D4RL antmaze.
    #         int_reward = int_reward - 1.0
    #     elif is_robomimic_env(FLAGS.env_name) or is_libero_env(FLAGS.env_name):
    #         # Adjust online (0, 1) reward for robomimic
    #         int_reward = int_reward - 1.0

    #     if FLAGS.sparse:
    #         assert int_reward <= 0.0
    #         int_reward = (int_reward != 0.0) * -1.0

    #     transition = dict(
    #         observations=ob,
    #         actions=action,
    #         rewards=int_reward,
    #         terminals=float(done),
    #         masks=1.0 - terminated,
    #         next_observations=next_ob,
    #     )
    #     replay_buffer.add_transition(transition)
        
    #     # done
    #     if done:
    #         ob, _ = env.reset()
    #         action_queue = []  # reset the action queue
    #     else:
    #         ob = next_ob

    #     if i >= FLAGS.start_training:
    #         batch = replay_buffer.sample_sequence(config['batch_size'] * FLAGS.utd_ratio, 
    #                     sequence_length=FLAGS.horizon_length, discount=discount)
    #         batch = jax.tree.map(lambda x: x.reshape((
    #             FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]), batch)

    #         agent, update_info["online_agent"] = agent.batch_update(batch)
            
    #     if i % FLAGS.log_interval == 0:
    #         for key, info in update_info.items():
    #             logger.log(info, key, step=log_step)
    #         update_info = {}

    #     if i == FLAGS.online_steps - 1 or \
    #         (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
    #         eval_info, _, _ = evaluate(
    #             agent=agent,
    #             env=eval_env,
    #             action_dim=action_dim,
    #             num_eval_episodes=FLAGS.eval_episodes,
    #             num_video_episodes=FLAGS.video_episodes,
    #             video_frame_skip=FLAGS.video_frame_skip,
    #         )
    #         logger.log(eval_info, "eval", step=log_step)

    #     # saving
    #     if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
    #         save_agent(agent, FLAGS.save_dir, log_step)

    # end_time = time.time()

    # for key, csv_logger in logger.csv_loggers.items():
    #     csv_logger.close()

    # if FLAGS.save_all_online_states:
    #     c_data = {"steps": np.array(data["steps"]),
    #              "qpos": np.stack(data["qpos"], axis=0), 
    #              "qvel": np.stack(data["qvel"], axis=0), 
    #              "obs": np.stack(data["obs"], axis=0), 
    #              "offline_time": online_init_time - offline_init_time,
    #              "online_time": end_time - online_init_time,
    #     }
    #     if len(data["button_states"]) != 0:
    #         c_data["button_states"] = np.stack(data["button_states"], axis=0)
    #     np.savez(os.path.join(FLAGS.save_dir, "data.npz"), **c_data)

    # with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
    #     f.write(run.url)

if __name__ == '__main__':
    app.run(main)
