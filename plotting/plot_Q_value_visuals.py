# viz_chunked_q.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import jax
import jax.numpy as jnp
from tqdm import tqdm
import imageio
def make_single_trajectory_visual(
    q_estimates,
    values,
    advantages,
    rewards,
    masks,
    obs_images,
    goal_images,
    bellman_loss,
):
    """Draw a single-trajectory panel (same layout as your old code)."""
    def np_unstack(array, axis):
        arr = np.split(array, array.shape[axis], axis)
        arr = [a.squeeze() for a in arr]
        return arr

    def process_images(images):
        # assume images shape (T, H, W, C) or (T, C, H, W)
        if images is None:
            return None
        imgs = np.array(images)
        # convert (T, C, H, W) -> (T, H, W, C)
        if imgs.ndim == 4 and imgs.shape[-1] != 3 and imgs.shape[1] == 3:
            imgs = np.transpose(imgs, (0, 2, 3, 1))
        assert imgs.ndim == 4 and imgs.shape[-1] == 3, f"images must be (T,H,W,3), got {imgs.shape}"
        interval = max(1, imgs.shape[0] // 4)
        sel_images = imgs[::interval]
        sel_images = np.concatenate(np_unstack(sel_images, 0), 1)
        return sel_images

    T = q_estimates.shape[-1]
    fig, axs = plt.subplots(8, 1, figsize=(8, 15))
    canvas = FigureCanvas(fig)

    obs_images = process_images(obs_images)
    # goal_images = process_images(goal_images)
    goal_images = None

    # images (if present)
    if obs_images is not None:
        axs[0].imshow(obs_images)
        axs[0].axis("off")
    else:
        axs[0].text(0.5, 0.5, "no obs images", ha="center", va="center")
        axs[0].axis("off")

    if goal_images is not None:
        axs[1].imshow(goal_images)
        axs[1].axis("off")
    else:
        axs[1].text(0.5, 0.5, "no goal images", ha="center", va="center")
        axs[1].axis("off")

    # q estimates: can be (num_qs, T) or (T,)
    if q_estimates.ndim == 2:
        for i in range(q_estimates.shape[0]):
            axs[2].plot(q_estimates[i, :], linestyle="--", marker="o", alpha=0.6)
    else:
        axs[2].plot(q_estimates, linestyle="--", marker="o")
    axs[2].set_ylabel("q values")
    axs[2].set_xlim([0, T])

    # values
    if values is not None:
        if values.ndim == 2:
            for i in range(values.shape[0]):
                axs[3].plot(values[i, :], linestyle="--", marker="o", alpha=0.6)
        else:
            axs[3].plot(values, linestyle="--", marker="o")
    axs[3].set_ylabel("values")
    axs[3].set_xlim([0, T])

    # advantages
    if advantages is not None:
        if advantages.ndim == 2:
            for i in range(advantages.shape[0]):
                axs[4].plot(advantages[i, :], linestyle="--", alpha=0.6)
        else:
            axs[4].plot(advantages, linestyle="--")
    axs[4].set_ylabel("advantages")
    axs[4].set_xlim([0, T])

    # bellman loss
    axs[5].plot(bellman_loss, linestyle="--", marker="o")
    axs[5].set_ylabel("bellman_loss")
    axs[5].set_xlim([0, len(bellman_loss)])

    # rewards
    axs[6].plot(rewards, linestyle="--", marker="o")
    axs[6].set_ylabel("rewards")
    axs[6].set_xlim([0, len(rewards)])

    # masks
    axs[7].plot(masks, linestyle="--", marker="o")
    axs[7].set_ylabel("masks")
    axs[7].set_xlim([0, len(masks)])

    plt.tight_layout()
    canvas.draw()
    out_image = np.frombuffer(canvas.buffer_rgba(), dtype="uint8")
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return out_image[..., :3]

def _prepare_obs_for_network(obs):
    """Return either np.ndarray or dict of arrays (unchanged shape), but converted to numpy arrays."""
    if obs is None:
        return None
    if isinstance(obs, dict):
        return {k: np.array(v) for k, v in obs.items()}
    return np.array(obs)

def _to_jnp(x):
    if isinstance(x, dict):
        return {k: jnp.array(v) for k, v in x.items()}
    return jnp.array(x)

def build_action_chunks(actions, horizon_length, action_dim):
    """
    Build chunked actions for each timestep t by concatenating actions[t : t+horizon_length].
    Inputs:
      - actions: np.ndarray with shape either (T, action_dim) or (T, horizon_length, action_dim) (pre-chunked)
    Returns:
      - action_chunks: (T, horizon_length * action_dim)
      - valid_mask: (T,) boolean array: True if chunk fully available and mask at last index should be used
    """
    actions = np.array(actions)
    T = actions.shape[0]
    full_dim = horizon_length * action_dim
    action_chunks = np.zeros((T, full_dim), dtype=actions.dtype)
    valid = np.zeros((T,), dtype=np.float32)

    # If actions already pre-chunked per timestep:
    if actions.ndim == 3 and actions.shape[1] == horizon_length and actions.shape[2] == action_dim:
        for t in range(T):
            chunk = actions[t]  # (horizon_length, action_dim)
            action_chunks[t] = chunk.reshape(-1)
            # validity determined by presence of full chunk (we still compute masks outside)
            valid[t] = 1.0
        return action_chunks, valid

    # else assume actions are one-step per t: (T, action_dim)
    if actions.ndim != 2:
        raise ValueError(f"Unsupported actions shape: {actions.shape}")

    for t in range(T):
        end = t + horizon_length
        if end <= T:
            seg = actions[t:end]  # (horizon_length, action_dim)
            action_chunks[t, :] = seg.reshape(-1)
            valid[t] = 1.0
        else:
            # pad with zeros, valid remains 0
            available = actions[t:T]
            pad_count = horizon_length - available.shape[0]
            if available.size > 0:
                action_chunks[t, : available.size] = available.reshape(-1)
            # valid[t] stays 0
    return action_chunks, valid

def value_and_reward_visualization(trajs, agent, filepath, step_num, discount=0.99, seed=0):
    """
    Main entrypoint: trajs is a list of dicts. Each traj dict must contain at least:
      - 'observations' : np.ndarray or dict of np.ndarray with leading dim T
      - 'actions' : np.ndarray shape (T, action_dim) OR (T, horizon_length, action_dim)
      - 'rewards' : (T,)
      - either 'masks' (T,) or 'dones' (T,) (we will compute masks = 1 - dones if needed)
      - optional: 'next_observations', 'goals'
    Agent must have:
      - agent.config with keys 'action_chunking' (bool), 'horizon_length' (int), 'action_dim' (int)
      - agent.network.select('critic') callable that accepts (observations_batch, actions=action_batch, params=...)
    Returns:
      - big concatenated image (np.uint8, H, W, 4)
    """
    rng = jax.random.PRNGKey(seed)
    n_trajs = len(trajs)
    visualization_images = []

    # Try to get config from agent if not provided separately
    cfg = getattr(agent, "config", None)
    if cfg is None:
        raise ValueError("agent must have a .config with action_chunking, horizon_length, action_dim")

    

    action_chunking = bool(cfg.get("action_chunking", True))
    horizon_length = int(cfg["horizon_length"])
    action_dim = int(cfg["action_dim"])

    for i in tqdm(range(n_trajs)):
        traj = trajs[i]
        # normalize shapes
        observations = traj["observations"]
        next_observations = traj.get("next_observations", None)
        goals = traj.get("goals", None)
        actions = np.array(traj["actions"])
        rewards = np.array(traj["rewards"])
        if "masks" in traj:
            masks = np.array(traj["masks"]).astype(np.float32)
        elif "dones" in traj:
            dones = np.array(traj["dones"]).astype(np.float32)
            masks = 1.0 - dones
        else:
            masks = np.ones_like(rewards, dtype=np.float32)

        # Build action chunks:
        # If action_chunking is True, critic expects full_action_dim = action_dim * horizon_length
        # We support two storage formats:
        # - actions shape (T, action_dim) -> we concatenate forward horizon steps
        # - actions shape (T, horizon_length, action_dim) -> already chunked per-timestep
        if action_chunking:
            action_chunks, chunk_valid_from_data = build_action_chunks(actions, horizon_length, action_dim)
            full_action_dim = horizon_length * action_dim
        else:
            # critic expects just the first action step
            if actions.ndim == 3:
                # maybe stored as (T, horizon, action_dim) -> take first in horizon
                action_chunks = actions[:, 0, :].reshape((actions.shape[0], -1))
            else:
                action_chunks = actions.reshape((actions.shape[0], -1))
            full_action_dim = action_chunks.shape[-1]
            chunk_valid_from_data = np.ones((action_chunks.shape[0],), dtype=np.float32)

        T = action_chunks.shape[0]

        # Compute validity mask consistent with training: use mask at last index of chunk
        if action_chunking:
            valid_mask = np.zeros((T,), dtype=np.float32)
            for t in range(T):
                last_idx = t + horizon_length - 1
                if last_idx < len(masks):
                    valid_mask[t] = masks[last_idx] * chunk_valid_from_data[t]
                else:
                    valid_mask[t] = 0.0
        else:
            # N-step or onestep: valid_mask aligned with t
            valid_mask = masks * chunk_valid_from_data

        # Prepare observations for network call:
        obs_for_net = _prepare_obs_for_network(observations)
        obs_j = _to_jnp(obs_for_net)
        actions_j = jnp.array(action_chunks)

        # Call critic in a vectorized fashion:
        critic_fn = agent.network.select("critic")
        # pass params if network uses TrainState style
        q_pred = None
        try:
            q_pred = critic_fn(obs_j, actions=actions_j, params=agent.network.params)
        except TypeError:
            # maybe it doesn't accept params keyword
            q_pred = critic_fn(obs_j, actions=actions_j)

        # Move to numpy for plotting; jax arrays -> numpy
        q_pred = np.array(jax.device_get(q_pred))

        # q_pred may be (num_qs, T) or (T,) or (T, num_qs) depending on implementation:
        if q_pred.ndim == 1:
            # (T,)
            q_for_plot = q_pred[np.newaxis, ...]  # (1, T)
        elif q_pred.ndim == 2:
            # determine which axis is ensemble: if shape[0] == num_qs (likely) then use as-is
            # heuristic: if first dim equals cfg.num_qs use that, else if last dim equals num_qs reshape
            num_qs = int(cfg.get("num_qs", 1))
            if q_pred.shape[0] == num_qs:
                q_for_plot = q_pred  # (num_qs, T)
            elif q_pred.shape[-1] == num_qs:
                # (T, num_qs) -> transpose
                q_for_plot = q_pred.T
            else:
                # ambiguous, assume (num_qs, T) if first dim smaller than last
                if q_pred.shape[0] < q_pred.shape[1]:
                    q_for_plot = q_pred
                else:
                    q_for_plot = q_pred.T
        else:
            # higher dims -> flatten to (num_qs, T) by combining leading dims
            q_for_plot = q_pred.reshape(q_pred.shape[0], -1)

        # Compute aggregated q and a surrogate "value"
        if q_for_plot.shape[0] > 1:
            q_mean = q_for_plot.mean(axis=0)
            q_min = q_for_plot.min(axis=0)
            # follow agent.config['q_agg'] logic: assume 'mean' else 'min'
            if cfg.get("q_agg", "mean") == "min":
                q_final = q_min
            else:
                q_final = q_mean
            values = q_mean  # surrogate value
        else:
            q_final = q_for_plot[0]
            values = q_final

        # advantages
        advantages = q_final - values

        # compute a simple bellman (TD) loss estimate for plotting
        # we need next_q (use critic on next timestep using action chunk starting at t+1)
        # build action_chunks for t+1 ... by shifting
        if action_chunking:
            # shift action chunks one forward, pad last with zeros
            next_action_chunks = np.zeros_like(action_chunks)
            next_action_chunks[:-1] = action_chunks[1:]
            next_action_chunks[-1] = 0.0
        else:
            next_action_chunks = np.zeros_like(action_chunks)
            next_action_chunks[:-1] = action_chunks[1:]
            next_action_chunks[-1] = action_chunks[-1]

        # compute next_q via critic
        next_actions_j = jnp.array(next_action_chunks)
        try:
            next_q_pred = critic_fn(obs_j, actions=next_actions_j, params=agent.network.params)
        except TypeError:
            next_q_pred = critic_fn(obs_j, actions=next_actions_j)
        next_q_pred = np.array(jax.device_get(next_q_pred))
        # normalize shapes like above
        if next_q_pred.ndim == 1:
            next_q_for_plot = next_q_pred[np.newaxis, ...]
        elif next_q_pred.ndim == 2:
            if next_q_pred.shape[0] == int(cfg.get("num_qs", 1)):
                next_q_for_plot = next_q_pred
            else:
                next_q_for_plot = next_q_pred.T
        else:
            next_q_for_plot = next_q_pred.reshape(next_q_pred.shape[0], -1)

        if next_q_for_plot.shape[0] > 1:
            if cfg.get("q_agg", "mean") == "min":
                next_q = next_q_for_plot.min(axis=0)
            else:
                next_q = next_q_for_plot.mean(axis=0)
        else:
            next_q = next_q_for_plot[0]

        # TD target and td_loss (elementwise)
        td_target = rewards + (discount ** horizon_length) * masks * next_q
        td_loss = (q_final - td_target) ** 2

        # prepare images for plot: try to pick image arrays out of observations / next_observations / goals
        def extract_image_from_obs(o):
            if o is None:
                return None
            if isinstance(o, dict):
                # prefer 'image' key
                if "image" in o:
                    return o["image"]
                # search for first array with channel dim 3
                for v in o.values():
                    arr = np.array(v)
                    if arr.ndim == 4 and (arr.shape[-1] == 3 or arr.shape[1] == 3):
                        return arr
                return None
            else:
                arr = np.array(o)
                if arr.ndim == 4 and (arr.shape[-1] == 3 or arr.shape[1] == 3):
                    return arr
                return None

        obs_images = extract_image_from_obs(next_observations if next_observations is not None else observations)
        # goal_images = extract_image_from_obs(goals)

        # print(f"q_for_plot shape: {q_for_plot.shape}, values shape: {values.shape}, advantages shape: {advantages.shape}, rewards shape: {rewards.shape}, masks shape: {valid_mask.shape}, obs_images shape: {obs_images.shape}, td_loss shape: {td_loss.shape}")
        print(f"setup done, calling visual now...")

        vis = make_single_trajectory_visual(
            q_estimates=q_for_plot,
            values=values,
            advantages=advantages,
            rewards=rewards,
            masks=valid_mask,
            obs_images=obs_images,
            goal_images=None,
            bellman_loss=td_loss,
        )
        visualization_images.append(vis)

    # concatenate vertically
    if len(visualization_images) == 0:
        return None
    print(f"Visualization images shape: {np.concatenate(visualization_images, axis=0).shape}")
    final_images = np.concatenate(visualization_images, axis=0)
    imageio.imwrite(f"{filepath}/Q_viz_step_{step_num}.png", final_images)
    return final_images