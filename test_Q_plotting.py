# first, use make_env_and_datset to initialize libero kitchen_scene2

from envs.env_utils import make_env_and_datasets
from agents.acfql import get_config, ACFQLAgent

# then, use evaluate_libero to evaluate the agent
from evaluation_libero import evaluate as evaluate_libero
import numpy as np
import matplotlib.pyplot as plt
# pass trajs to plot_Q_value_visuals
from plotting.plot_Q_value_visuals import value_and_reward_visualization

env_name = "libero_90-kitchen_scene2"
task_name = "open_the_top_drawer_of_the_cabinet"
keys_to_load = ['states']
augment_negative_demos = False
env, eval_env, train_dataset, val_dataset = make_env_and_datasets(env_name, num_parallel_envs=5, task_name=task_name, keys_to_load=keys_to_load, augment_negative_demos=augment_negative_demos)


example_batch = train_dataset.sample(())
config = get_config()
config.horizon_length = 5
agent = ACFQLAgent.create(
    seed=0,
    ex_observations=example_batch['observations'],
    ex_actions=example_batch['actions'],
    config=config,
)



info, trajs, renders = evaluate_libero(
                agent=agent,
                env=eval_env,
                action_dim=example_batch["actions"].shape[-1],
                num_eval_episodes=5,
                num_video_episodes=0, 
                num_parallel_envs=5,
                video_frame_skip=3,
            )


viz_images = value_and_reward_visualization(trajs, agent, filepath, step_num)

# save all of viz_images to a single image and write to disk
# If it’s RGBA, convert to RGB
if viz_images.shape[-1] == 4:
    viz_images = viz_images[..., :3]

# Save
import imageio
imageio.imwrite("fql_traj_visualization.png", viz_images)
print("Saved to fql_traj_visualization.png")