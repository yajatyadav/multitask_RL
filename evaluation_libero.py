from collections import defaultdict

import jax
import numpy as np
from tqdm import trange
from functools import partial


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)

## TODO(YY): use sticky gripper / binarize gripper dim from action...
def evaluate(
    agent,
    env,
    num_parallel_envs,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
    action_shape=None,
    observation_shape=None,
    action_dim=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        num_parallel_envs: Number of parallel environments for evaluation, this lets us know how many base envs env actually has.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    eval_env = env.get_eval_env()
    video_env = env.get_video_env()

    assert num_eval_episodes % num_parallel_envs == 0, "num_eval_episodes must be divisible by num_parallel_envs"
    num_eval_iterations = num_eval_episodes // num_parallel_envs
    num_video_iterations = num_video_episodes
    
    for i in trange(num_eval_iterations + num_video_iterations): # only 1 iteration for eval, since multiprocessed
        traj = defaultdict(list)
        # should_render = i >= num_eval_episodes
        should_render = i >= num_eval_iterations
        if should_render:
            env = video_env
            num_episodes_this_iter = 1
        else:
            env = eval_env
            num_episodes_this_iter = num_parallel_envs
        observation, info = env.reset(), {}
            
        observation_history = []
        action_history = []
        
        done = [False] * num_episodes_this_iter
        step = 0
        render = []
        action_chunk_lens = defaultdict(lambda: 0)

        action_queue = []

        gripper_contact_lengths = []
        gripper_contact_length = 0

        # setup info dict for each episode outisde loop
        info = [{} for _ in range(num_episodes_this_iter)]

        while not all(done):
            action = actor_fn(observations=observation)

            if len(action_queue) == 0:
                have_new_action = True
                action = np.array(action).reshape(num_episodes_this_iter, -1, action_dim).transpose(1, 0, 2) # since each elt of action queue should have shape (num_eval_episodes, action_dim)
                action_chunk_len = action.shape[0]
                for a in action:
                    action_queue.append(a)
            else:
                have_new_action = False
            
            action = action_queue.pop(0)
            if eval_gaussian is not None:
                action = np.random.normal(action, eval_gaussian)

            # before stepping, let's 0-out the actions for the envs that are done, since stepping() after done can potentially have undefined behavior
            action_mask = 1 - np.array(done, dtype=np.float32)[:, np.newaxis]
            action = action * action_mask
            # if np.any(action_mask == 0):
                # print(f"action mask: {action_mask} at timestep {step} ")
            next_observation, reward, done, info = env.step(np.clip(action, -1, 1)) # the done returned by env.step() already has 'or truncated' absorbed in
            step += 1

            for inf in info:
                info[int(inf['env_id'])] = inf
        

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame[0][::-1, ::-1]) # need to remove an extra dim added b/c the render-env is a subprocenv
                # also need to flip inage, just like all other libero sims


            # transition = dict(
            #     observation=observation,
            #     next_observation=next_observation,
            #     action=action,
            #     reward=reward,
            #     done=done,
            #     info=info,
            # )
            # add_to(traj, transition)
            
            observation = next_observation
            # print(info)
            if "proprio" in info and "gripper_contact" in info["proprio"]:
                # print(info["gripper_contact"])
                gripper_contact = info["proprio"]["gripper_contact"]
            elif "gripper_contact" in info:
                gripper_contact = info["gripper_contact"]
            else:
                gripper_contact = None

            if gripper_contact is not None:
                if info["gripper_contact"] > 0.1:
                    gripper_contact_length += 1
                else:
                    if gripper_contact_length > 0:
                        gripper_contact_lengths.append(gripper_contact_length)
                    gripper_contact_length = 0

        if gripper_contact_length > 0:
            gripper_contact_lengths.append(gripper_contact_length)
        
        num_gripper_contacts = len(gripper_contact_lengths)

        if num_gripper_contacts > 0:
            avg_gripper_contact_length = np.mean(np.array(gripper_contact_lengths))
        else:
            avg_gripper_contact_length = 0
            
        add_to(stats, {"avg_gripper_contact_length": avg_gripper_contact_length, "num_gripper_contacts": num_gripper_contacts})

        # print("ending info dicts: ", info)
        # after this iter finishes, either add all inf dicts into stats, or add the render to the renders list
        if i < num_eval_iterations:
            for inf in info:
                add_to(stats, flatten(inf))
            # trajs.append(traj)
        else:
            renders.append(np.array(render))
    
    # aggregate stats over all iterations
    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders

