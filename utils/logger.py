import os
import tempfile
from datetime import datetime

import absl.flags as flags
import ml_collections
import numpy as np
import wandb
from PIL import Image, ImageEnhance

from rich.tree import Tree


class CsvLogger:
    """CSV logger for logging metrics to a CSV file."""

    def __init__(self, path):
        self.path = path
        self.header = None
        self.file = None
        self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)

    def log(self, row, step):
        row['step'] = step
        if self.file is None:
            self.file = open(self.path, 'w')
            if self.header is None:
                self.header = [k for k, v in row.items() if not isinstance(v, self.disallowed_types)]
                self.file.write(','.join(self.header) + '\n')
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        else:
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


def get_exp_name(seed):
    """Return the experiment name."""
    exp_name = ''
    exp_name += f'sd{seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    exp_name += f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    return exp_name


def get_flag_dict():
    """Return the dictionary of flags."""
    flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS if '.' not in k}
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict


def setup_wandb(
    entity='yajatyadav',
    project='project',
    group=None,
    name=None,
    mode='online',
    log_flags=True,
):
    """Set up Weights & Biases for logging."""
    wandb_output_dir = tempfile.mkdtemp()
    tags = [group] if group is not None else None
    config = get_flag_dict() if log_flags else None

    init_kwargs = dict(
        config=config,
        project=project,
        entity=entity,
        tags=tags,
        group=group,
        dir=wandb_output_dir,
        name=name,
        settings=wandb.Settings(
            start_method='thread',
            _disable_stats=False,
        ),
        mode=mode,
        save_code=True,
    )

    run = wandb.init(**init_kwargs)

    return run


def reshape_video(v, n_cols=None):
    """Helper function to reshape videos."""
    if v.ndim == 4:
        v = v[None,]

    _, t, h, w, c = v.shape

    if n_cols is None:
        # Set n_cols to the square root of the number of videos.
        n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, h, w, c))
    v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
    v = np.reshape(v, newshape=(t, c, n_rows * h, n_cols * w))

    return v


def get_wandb_video(renders=None, n_cols=None, fps=15):
    """Return a Weights & Biases video.

    It takes a list of videos and reshapes them into a single video with the specified number of columns.

    Args:
        renders: List of videos. Each video should be a numpy array of shape (t, h, w, c).
        n_cols: Number of columns for the reshaped video. If None, it is set to the square root of the number of videos.
    """
    # Pad videos to the same length.
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        assert render.dtype == np.uint8

        # Decrease brightness of the padded frames.
        final_frame = render[-1]
        final_image = Image.fromarray(final_frame)
        enhancer = ImageEnhance.Brightness(final_image)
        final_image = enhancer.enhance(0.5)
        final_frame = np.array(final_image)

        pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(render), axis=0)
        renders[i] = np.concatenate([render, pad], axis=0)

        # Add borders.
        renders[i] = np.pad(renders[i], ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    renders = np.array(renders)  # (n, t, h, w, c)

    renders = reshape_video(renders, n_cols)  # (t, c, nr * h, nc * w)

    return wandb.Video(renders, fps=fps, format='mp4')


def get_sample_input_output_log_to_wandb(batch):
    """Log sample input and output to wandb."""
    def log_obs(obs, obs_type):
        image_primary, image_wrist = obs["image_primary"], obs["image_wrist"]
        sim_state = obs["sim_state"]
        proprio = obs["proprio"]
        return {
            f"{obs_type}/image_primary": wandb.Image(image_primary[0]),
            f"{obs_type}/image_wrist": wandb.Image(image_wrist[0]),
            f"{obs_type}/proprio": wandb.Table(columns=[f"{obs_type}_proprio_{i}" for i in range(len(proprio[0]))], 
            data=[proprio[0].tolist()]),
            f"{obs_type}/sim_state": wandb.Table(columns=[f"{obs_type}_sim_state_{i}" for i in range(len(sim_state[0]))], data=[sim_state[0].tolist()]),
        }
    dict_to_log = {}
    dict_to_log.update(log_obs(batch["observations"], "observations"))
    dict_to_log.update(log_obs(batch["next_observations"], "next_observations"))
    
    rewards = batch["rewards"]
    actions = batch["actions"]
    masks = batch["masks"]
    dict_to_log.update({
        "reward": float(rewards[0]),
        "action": wandb.Table(columns=[f"action_{i}" for i in range(len(actions[0]))], data=[actions[0].tolist()]),
        "mask": float(1 if masks[0] else 0),
    })
    return dict_to_log



def build_network_tree(params, name="Network Parameters"):
    """Build a rich Tree visualization with colors"""
    tree = Tree(f"[bold cyan]{name}[/bold cyan]")
    
    def add_nodes(parent, data):
        items = list(data.items())
        for key, value in items:
            if isinstance(value, dict):
                # It's a submodule
                branch = parent.add(f"[bold yellow]{key}/[/bold yellow]")
                add_nodes(branch, value)
            else:
                # It's a parameter
                shape = f"{value.shape}" if hasattr(value, 'shape') else ""
                dtype = f"[dim]{value.dtype}[/dim]" if hasattr(value, 'dtype') else ""
                size = f"{value.size:,}" if hasattr(value, 'size') else ""
                parent.add(f"[green]{key}[/green]: {shape} {dtype} [dim]({size} params)[/dim]")
    
    add_nodes(tree, params)
    return tree