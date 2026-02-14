"""
Aggregate BRC chunked best-of-N results and log final outputs.

This script should be run after all SLURM jobs finish.
"""

import argparse
import glob
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import wandb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from utils.log_utils import get_wandb_video


def _sanitize_metric_key(text):
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in text)


def _load_job_pickles(intermediate_run_dir):
    pattern = os.path.join(intermediate_run_dir, "job_results", "N_*", "*.pkl")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No job pickle files found under: {pattern}")

    grouped = defaultdict(list)
    for path in paths:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        n_val = int(payload["metadata"]["n_val"])
        grouped[n_val].append(payload)
    return grouped, paths


def _aggregate_n_payloads(payloads_for_n):
    per_env = {}
    for payload in payloads_for_n:
        for item in payload["selected_env_results"]:
            env_name = item["env_name"]
            if env_name not in per_env:
                per_env[env_name] = item

    if not per_env:
        return {"agg_eval_info": {}, "all_eval_info": [], "per_env": {}}

    sample_env = next(iter(per_env.values()))
    metric_keys = [k for k in sample_env["eval_info"].keys() if k != "video"]
    agg_eval_info = {}
    for key in metric_keys:
        agg_eval_info[key] = float(np.mean([v["eval_info"][key] for v in per_env.values()]))

    all_eval_info = [v["eval_info"] for v in per_env.values()]
    return {
        "agg_eval_info": agg_eval_info,
        "all_eval_info": all_eval_info,
        "per_env": per_env,
    }


def _log_to_wandb(args, all_n_results):
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group_name,
        name=args.wandb_run_name,
        id=args.wandb_run_id if args.wandb_run_id else None,
        resume="allow",
    )
    wandb.define_metric("eval/num_samples")
    wandb.define_metric("eval/*", step_metric="eval/num_samples")

    for n in sorted(all_n_results.keys()):
        result = all_n_results[n]
        log_data = {"eval/num_samples": n}

        for k, v in result["agg_eval_info"].items():
            log_data[f"eval/{k}"] = v

        for env_name, env_data in result["per_env"].items():
            env_key = _sanitize_metric_key(env_name)
            env_eval = env_data["eval_info"]
            for k, v in env_eval.items():
                if k == "video":
                    continue
                log_data[f"eval_env/{env_key}/{k}"] = v
            renders = env_data.get("renders", [])
            if renders:
                log_data[f"eval_env/{env_key}/video"] = get_wandb_video(renders)

        run.log(log_data)
        print(f"Logged N={n} to wandb.")

    print(f"Wandb URL: {run.url}")
    run.finish()


def _write_final_outputs(args, all_n_results):
    final_dir = os.path.join(args.results_save_path, args.wandb_run_name)
    os.makedirs(final_dir, exist_ok=True)

    summary = {
        "wandb_run_name": args.wandb_run_name,
        "wandb_run_id": args.wandb_run_id,
        "n_vals": sorted(all_n_results.keys()),
        "intermediate_run_dir": args.intermediate_run_dir,
    }

    for n in sorted(all_n_results.keys()):
        result = all_n_results[n]
        out_path = os.path.join(final_dir, f"{args.wandb_run_name}_N_{n}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(
                {
                    "agg_eval_info": result["agg_eval_info"],
                    "all_eval_info": result["all_eval_info"],
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        summary[f"N_{n}"] = result["agg_eval_info"]
        print(f"Wrote final result: {out_path}")

    summary_path = os.path.join(final_dir, f"{args.wandb_run_name}_summary.pkl")
    with open(summary_path, "wb") as f:
        pickle.dump(summary, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote summary: {summary_path}")


def main(args):
    grouped, all_paths = _load_job_pickles(args.intermediate_run_dir)
    print(f"Loaded {len(all_paths)} chunk result files.")

    expected = set(args.n_vals) if args.n_vals else set(grouped.keys())
    missing = sorted(expected - set(grouped.keys()))
    if missing:
        raise ValueError(f"Missing results for N values: {missing}")

    all_n_results = {}
    for n in sorted(expected):
        all_n_results[n] = _aggregate_n_payloads(grouped[n])
        if not all_n_results[n]["all_eval_info"]:
            raise ValueError(f"N={n} has no env results; likely chunking mismatch.")
        print(
            f"N={n}: envs={len(all_n_results[n]['per_env'])}, "
            f"agg_success={all_n_results[n]['agg_eval_info'].get('success', 'n/a')}"
        )

    _write_final_outputs(args, all_n_results)
    if args.use_wandb:
        _log_to_wandb(args, all_n_results)


def parse_args():
    parser = argparse.ArgumentParser(description="Postprocess BRC best-of-N chunked results.")
    parser.add_argument("--intermediate_run_dir", type=str, required=True)
    parser.add_argument("--results_save_path", type=str, default="exp/multitask_RL/eval")
    parser.add_argument("--n_vals", nargs="+", type=int, default=None)

    parser.add_argument("--wandb_group_name", type=str, required=True)
    parser.add_argument("--wandb_run_name", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="multitask_RL")
    parser.add_argument("--wandb_entity", type=str, default="yajatyadav")
    parser.add_argument("--wandb_run_id", type=str, default="")
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--no_wandb", action="store_false", dest="use_wandb")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
