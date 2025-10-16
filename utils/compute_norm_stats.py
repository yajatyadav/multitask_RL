import rlds_dataloader.dataloader as rlds_data_loader
import tqdm
import jax.numpy as jnp
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
import os
from utils.data_utils import flatten_dict, unflatten_dict
dataloader_config_template = {
    "data_root_dir": "/raid/users/yajatyadav/datasets/raw_libero/raw_libero_RLDS/",
    "dataset_mix": {},
    "batch_size": 256,
    "balance_datasets": True,
    "num_workers": 16,
    "seed": 42,
    "do_image_aug": False,
    "binarize_gripper": True,
    "train": True,
}


def compute_stats(all_values):
    """Compute comprehensive statistics for collected values."""
    all_values = np.concatenate(all_values, axis=0)
    
    stats = {
        # Aggregate stats (averaged/combined across all dimensions)
        'min_agg': float(np.min(all_values)),
        'max_agg': float(np.max(all_values)),
        'q01_agg': float(np.percentile(all_values, 1)),
        'q05_agg': float(np.percentile(all_values, 5)),
        'q25_agg': float(np.percentile(all_values, 25)),
        'q50_agg': float(np.percentile(all_values, 50)),  # median
        'q75_agg': float(np.percentile(all_values, 75)),
        'q95_agg': float(np.percentile(all_values, 95)),
        'q99_agg': float(np.percentile(all_values, 99)),
        'mean_agg': float(np.mean(all_values)),
        'std_agg': float(np.std(all_values)),
        
        # Per-dimension stats (primary names)
        'min': np.min(all_values, axis=0).tolist(),
        'max': np.max(all_values, axis=0).tolist(),
        'mean': np.mean(all_values, axis=0).tolist(),
        'std': np.std(all_values, axis=0).tolist(),
        'q01': np.percentile(all_values, 1, axis=0).tolist(),
        'q05': np.percentile(all_values, 5, axis=0).tolist(),
        'q25': np.percentile(all_values, 25, axis=0).tolist(),
        'q50': np.percentile(all_values, 50, axis=0).tolist(),
        'q75': np.percentile(all_values, 75, axis=0).tolist(),
        'q95': np.percentile(all_values, 95, axis=0).tolist(),
        'q99': np.percentile(all_values, 99, axis=0).tolist(),
        
        # Shape info
        'shape': list(all_values.shape),
        'num_samples': int(all_values.shape[0]),
        'num_dims': int(all_values.shape[1]) if len(all_values.shape) > 1 else 1,
    }
    
    return stats, all_values


def print_stats(key, stats):
    """Pretty print statistics."""
    print(f"\n{'='*70}")
    print(f"Statistics for: {key}")
    print(f"{'='*70}")
    print(f"Shape: {stats['shape']}")
    print(f"Number of samples: {stats['num_samples']:,}")
    print(f"Number of dimensions: {stats['num_dims']}")
    
    print(f"\n{'-'*70}")
    print(f"Aggregate Statistics (across all dimensions):")
    print(f"{'-'*70}")
    print(f"{'Statistic':<15} {'Value':>15}")
    print(f"{'-'*70}")
    print(f"{'Min':<15} {stats['min_agg']:>15.6f}")
    print(f"{'Q01':<15} {stats['q01_agg']:>15.6f}")
    print(f"{'Q25':<15} {stats['q25_agg']:>15.6f}")
    print(f"{'Median (Q50)':<15} {stats['q50_agg']:>15.6f}")
    print(f"{'Q75':<15} {stats['q75_agg']:>15.6f}")
    print(f"{'Q99':<15} {stats['q99_agg']:>15.6f}")
    print(f"{'Max':<15} {stats['max_agg']:>15.6f}")
    print(f"{'Mean':<15} {stats['mean_agg']:>15.6f}")
    print(f"{'Std':<15} {stats['std_agg']:>15.6f}")
    print(f"{'-'*70}")
    
    # Per-dimension summary
    if stats['num_dims'] <= 20:
        print(f"\nPer-dimension ranges:")
        for i in range(stats['num_dims']):
            print(f"  Dim {i:2d}: [{stats['min'][i]:8.4f}, {stats['max'][i]:8.4f}] "
                  f"(mean={stats['mean'][i]:7.4f}, std={stats['std'][i]:7.4f})")


def plot_distributions(key, all_values, stats, output_dir):
    """Plot histograms and per-dimension distributions."""
    num_dims = all_values.shape[1] if len(all_values.shape) > 1 else 1
    safe_filename = key.replace('/', '_')
    
    # ========== Overall distribution plot ==========
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    axes[0].hist(all_values.flatten(), bins=100, alpha=0.7, edgecolor='black')
    axes[0].axvline(stats['mean_agg'], color='r', linestyle='--', linewidth=2, label=f"Mean: {stats['mean_agg']:.3f}")
    axes[0].axvline(stats['q01_agg'], color='g', linestyle='--', linewidth=2, label=f"Q01: {stats['q01_agg']:.3f}")
    axes[0].axvline(stats['q99_agg'], color='g', linestyle='--', linewidth=2, label=f"Q99: {stats['q99_agg']:.3f}")
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{key} - Overall Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot per dimension (if reasonable number of dims)
    if num_dims <= 100:
        axes[1].boxplot([all_values[:, i] for i in range(num_dims)], 
                       labels=[f'D{i}' for i in range(num_dims)])
        axes[1].set_xlabel('Dimension')
        axes[1].set_ylabel('Value')
        axes[1].set_title(f'{key} - Per-Dimension Box Plot')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, f'Too many dimensions ({num_dims}) to plot',
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title(f'{key} - {num_dims} dimensions')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{safe_filename}_overall.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plot: {output_dir / f'{safe_filename}_overall.png'}")
    
    # ========== Per-dimension histograms ==========
    if num_dims <= 100:  # Only create individual plots if reasonable number of dimensions
        # Determine grid size
        cols = min(4, num_dims)
        rows = (num_dims + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if num_dims == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i in range(num_dims):
            ax = axes[i]
            dim_data = all_values[:, i]
            
            # Histogram for this dimension
            ax.hist(dim_data, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
            
            # Add statistical lines
            ax.axvline(stats['mean'][i], color='r', linestyle='--', linewidth=2, 
                      label=f"Mean: {stats['mean'][i]:.3f}")
            ax.axvline(stats['q01'][i], color='g', linestyle='--', linewidth=1.5,
                      label=f"Q01: {stats['q01'][i]:.3f}")
            ax.axvline(stats['q99'][i], color='g', linestyle='--', linewidth=1.5,
                      label=f"Q99: {stats['q99'][i]:.3f}")
            ax.axvline(stats['min'][i], color='orange', linestyle=':', linewidth=1,
                      label=f"Min: {stats['min'][i]:.3f}")
            ax.axvline(stats['max'][i], color='orange', linestyle=':', linewidth=1,
                      label=f"Max: {stats['max'][i]:.3f}")
            
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Dimension {i}')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Add text box with stats
            textstr = f'μ={stats["mean"][i]:.3f}\nσ={stats["std"][i]:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=props)
        
        # Hide unused subplots
        for i in range(num_dims, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{safe_filename}_per_dim.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved plot: {output_dir / f'{safe_filename}_per_dim.png'}")
    else:
        print(f"  Skipping per-dimension plots (too many dimensions: {num_dims})")


def main(args):
    dataset_name = args["dataset_name"]
    save_plots = args.get("save_plots", True)
    keys_to_analyze = args.get("keys", None)  # None means analyze all
    
    dataloader_config = dataloader_config_template.copy()
    dataloader_config["dataset_mix"] = { dataset_name: 1.0 }
    dataloader = rlds_data_loader.create_data_loader(
        dataloader_config, 
        skip_norm_stats=True,
        infinite_dataset=False,
        normalize_batches=False, # this MUST BE FALSE FOR COMPUTING NORM STATS
    )

    collected_data = {}
    
    print(f"\n{'='*70}")
    print(f"Computing statistics for dataset: {dataset_name}")
    print(f"{'='*70}\n")
    
    # Collect all data
    num_batches = 0
    total_samples = 0
    
    for batch in tqdm.tqdm(dataloader, desc="Collecting data"):
        # Flatten the batch dictionary
        flat_batch = flatten_dict(batch)
        
        # Initialize collected_data keys on first batch
        if num_batches == 0:
            if keys_to_analyze is None:
                # Analyze all keys
                keys_to_analyze = list(flat_batch.keys())
                # Filter out image keys (usually too large)
                keys_to_analyze = [k for k in keys_to_analyze if 'image' not in k.lower()]
            
            print(f"Keys found in batch: {list(flat_batch.keys())}")
            print(f"Keys to analyze: {keys_to_analyze}\n")
            
            for key in keys_to_analyze:
                collected_data[key] = []
        
        # Collect data for each key
        for key in keys_to_analyze:
            if key in flat_batch:
                value = np.array(flat_batch[key])
                # Only collect if it's numeric data (not images)
                if value.dtype in [np.float32, np.float64, np.int32, np.int64]:
                    collected_data[key].append(value)
            else:
                print(f"Warning: Key '{key}' not found in batch {num_batches}")
        
        num_batches += 1
        total_samples += len(batch.get('actions', batch[list(batch.keys())[0]]))
    
    print(f"\nProcessed {num_batches} batches ({total_samples:,} total samples)")
    
    # Compute and save statistics
    output_dir = Path("dataset_stats") / dataset_name
    # remove if already exists
    if output_dir.exists():
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    final_stats = {}
    all_data = {}
    
    for key in keys_to_analyze:
        if key in collected_data and collected_data[key]:
            print(f"\nProcessing {key}...")
            
            try:
                stats, values = compute_stats(collected_data[key])
                final_stats[key] = stats
                all_data[key] = values
                
                print_stats(key, stats)
                
                if save_plots:
                    plot_distributions(key, values, stats, output_dir)
            except Exception as e:
                print(f"  Error computing stats for {key}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\nWarning: No data collected for key '{key}'")
    
    # Save statistics to JSON
    output_file = output_dir / "stats.json"
    with open(output_file, 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    # Also save a summary CSV for quick reference
    summary_file = output_dir / "stats_summary.csv"
    with open(summary_file, 'w') as f:
        f.write("key,min_agg,q01_agg,q50_agg,q99_agg,max_agg,mean_agg,std_agg,num_dims\n")
        for key, stats in final_stats.items():
            f.write(f"{key},{stats['min_agg']},{stats['q01_agg']},{stats['q50_agg']},"
                   f"{stats['q99_agg']},{stats['max_agg']},{stats['mean_agg']},"
                   f"{stats['std_agg']},{stats['num_dims']}\n")
    
    print(f"\n{'='*70}")
    print(f"✓ Statistics saved to: {output_file}")
    print(f"✓ Summary CSV saved to: {summary_file}")
    print(f"✓ Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    return final_stats, all_data


if __name__ == "__main__":
    args = dict(
        dataset_name="libero_90__black_bowl_on_plate_kitchen_scene1",
        save_plots=True,
        keys=["actions", "observations/proprio", "observations/sim_state"],  # None = analyze all non-image keys, or specify: ["actions", "observations/proprio"]
    )
    stats, data = main(args)