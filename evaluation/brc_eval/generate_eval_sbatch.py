"""
Generate BRC SLURM submission script for BERT best-of-N evaluation.

Work is split by:
- one N value
- one env chunk id
"""

import argparse
import os
import secrets
import shlex
import time


def _q(x):
    return shlex.quote(str(x))


def generate_submission_script(args):
    if not args.wandb_run_id:
        args.wandb_run_id = secrets.token_hex(12)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(args.intermediate_base_dir, args.wandb_run_name)
    script_dir = os.path.join(run_root, "submission_scripts")
    log_dir = os.path.join(run_root, "slurm_logs")
    os.makedirs(script_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    out_script = os.path.join(script_dir, f"submit_brc_eval_{timestamp}.sh")
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    lines.append(f'RUN_NAME="{args.wandb_run_name}"')
    lines.append(f'RUN_ID="{args.wandb_run_id}"')
    lines.append(f'RUN_ROOT="{run_root}"')
    lines.append(f'LOG_DIR="{log_dir}"')
    lines.append("")
    lines.append('mkdir -p "$LOG_DIR"')
    lines.append("")

    sbatch_common = [
        f"-A {args.account}",
        f"-p {args.partition}",
        f"--gres=gpu:{args.gpu_type}:{args.num_gpus}",
        f"-N {args.num_nodes}",
        f"-n {args.num_tasks}",
        f"-c {args.cpus_per_task}",
        f"--qos={args.qos}",
        f"-t {args.time}",
        f"--mem={args.mem}",
        "--parsable",
    ]
    if args.requeue:
        sbatch_common.append("--requeue")
    sbatch_common_str = " ".join(sbatch_common)

    wandb_env_vars = [
        "WANDB_SERVICE_WAIT=86400",
        "WANDB_NETWORK_TIMEOUT=600",
        "WANDB_FILE_TRANSFER_TIMEOUT=1200",
        "WANDB_INIT_TIMEOUT=300",
        "WANDB_HTTP_TIMEOUT=600",
        "WANDB_RETRY_ATTEMPTS=15",
        "WANDB_RETRY_WAIT_MIN=5",
        "WANDB_RETRY_WAIT_MAX=120",
    ]
    system_env_vars = [
        "MUJOCO_GL=egl",
        "XLA_PYTHON_CLIENT_PREALLOCATE=false",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "VECLIB_MAXIMUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
    ]
    all_env_vars = " ".join(wandb_env_vars + system_env_vars)

    lines.append("echo \"Submitting BRC eval jobs...\"")
    lines.append("job_ids=()")
    lines.append("")

    job_idx = 0
    for n in args.n_vals:
        for chunk_id in range(args.num_env_chunks):
            job_name = f"{args.slurm_job_name_prefix}_N{n}_C{chunk_id}"
            stdout_path = os.path.join(log_dir, f"{job_name}_%j.out")
            stderr_path = os.path.join(log_dir, f"{job_name}_%j.err")
            cmd = [
                "uv run evaluation/brc_eval/eval_libero_BERT_classifier_best_of_N_single_job.py",
                f"--classifier_type {_q(args.classifier_type)}",
                f"--classifier_restore_dir {_q(args.classifier_restore_dir)}",
                f"--classifier_ckpt_num {args.classifier_ckpt_num}",
                f"--actor_restore_path {_q(args.actor_restore_path)}",
                f"--actor_encoder {_q(args.actor_encoder)}",
                f"--env_name {_q(args.env_name)}",
                f"--task_name {_q(args.task_name)}",
                f"--n_val {n}",
                f"--env_chunk_id {chunk_id}",
                f"--num_env_chunks {args.num_env_chunks}",
                f"--wandb_group_name {_q(args.wandb_group_name)}",
                f"--wandb_run_name {_q(args.wandb_run_name)}",
                f"--wandb_project {_q(args.wandb_project)}",
                f"--wandb_entity {_q(args.wandb_entity)}",
                f"--wandb_run_id {_q(args.wandb_run_id)}",
                f"--horizon_length {args.horizon_length}",
                f"--num_eval_episodes {args.num_eval_episodes}",
                f"--num_video_episodes {args.num_video_episodes}",
                f"--num_parallel_envs {args.num_parallel_envs}",
                f"--video_frame_skip {args.video_frame_skip}",
                f"--language_embedder {_q(args.language_embedder)}",
                f"--actor_seed {args.actor_seed}",
                f"--intermediate_base_dir {_q(args.intermediate_base_dir)}",
                "--keys_to_load " + " ".join(_q(k) for k in args.keys_to_load),
                "--demo_nums_to_use_per_task " + " ".join(str(x) for x in args.demo_nums_to_use_per_task),
            ]
            py_cmd = " ".join(cmd)

            sbatch_line = (
                f'jobid_{job_idx}=$({all_env_vars} sbatch {sbatch_common_str} '
                f'--job-name={_q(job_name)} '
                f'--comment={_q(f"bert_best_of_n N={n} chunk={chunk_id}")} '
                f'--output={_q(stdout_path)} '
                f'--error={_q(stderr_path)} '
                f'{_q(args.script_runner)} {_q(py_cmd)})'
            )
            lines.append(sbatch_line)
            lines.append(f'echo "Submitted {job_name}: $jobid_{job_idx}"')
            lines.append(f'job_ids+=("$jobid_{job_idx}")')
            lines.append("")
            job_idx += 1

    lines.append('echo "Total jobs submitted: ${#job_ids[@]}"')
    lines.append('echo "Run name: $RUN_NAME"')
    lines.append('echo "Run ID: $RUN_ID"')
    lines.append('echo "Intermediate outputs: $RUN_ROOT"')
    lines.append("")
    lines.append("echo \"After jobs finish, run postprocess:\"")
    lines.append(
        "echo "
        + _q(
            "uv run evaluation/brc_eval/postprocess_best_of_n_eval.py "
            f"--intermediate_run_dir {run_root} "
            f"--results_save_path {args.results_save_path} "
            f"--wandb_group_name {args.wandb_group_name} "
            f"--wandb_run_name {args.wandb_run_name} "
            f"--wandb_project {args.wandb_project} "
            f"--wandb_entity {args.wandb_entity} "
            f"--wandb_run_id {args.wandb_run_id} "
            f"--n_vals {' '.join(str(x) for x in args.n_vals)}"
        )
    )

    with open(out_script, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(out_script, 0o755)

    print(f"Generated: {out_script}")
    print(f"Wandb run id: {args.wandb_run_id}")
    print(f"Submit with: bash {out_script}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate BRC SLURM script for BERT best-of-N.")
    parser.add_argument("--classifier_type", choices=["language", "success"], required=True)
    parser.add_argument("--classifier_restore_dir", type=str, required=True)
    parser.add_argument("--classifier_ckpt_num", type=int, required=True)
    parser.add_argument("--actor_restore_path", type=str, required=True)
    parser.add_argument("--actor_encoder", type=str, default="combined_encoder_small")

    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--n_vals", nargs="+", type=int, required=True)
    parser.add_argument(
        "--num_env_chunks",
        type=int,
        default=8,
        help="Each N is split into this many env-chunk jobs.",
    )

    parser.add_argument("--wandb_group_name", type=str, required=True)
    parser.add_argument("--wandb_run_name", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="multitask_RL")
    parser.add_argument("--wandb_entity", type=str, default="yajatyadav")
    parser.add_argument("--wandb_run_id", type=str, default="")

    parser.add_argument("--horizon_length", type=int, default=5)
    parser.add_argument("--num_eval_episodes", type=int, default=50)
    parser.add_argument("--num_video_episodes", type=int, default=5)
    parser.add_argument("--num_parallel_envs", type=int, default=5)
    parser.add_argument("--video_frame_skip", type=int, default=3)
    parser.add_argument("--language_embedder", type=str, default="bert")
    parser.add_argument(
        "--keys_to_load",
        nargs="+",
        type=str,
        default=["agentview_rgb", "eye_in_hand_rgb", "language", "proprio"],
    )
    parser.add_argument("--actor_seed", type=int, default=0)
    parser.add_argument("--demo_nums_to_use_per_task", nargs="+", type=int, default=[0])

    parser.add_argument("--intermediate_base_dir", type=str, default="exp/multitask_RL/eval_intermediate/brc_eval_jobs")
    parser.add_argument("--results_save_path", type=str, default="exp/multitask_RL/eval")
    parser.add_argument("--script_runner", type=str, default="scripts/automatic/run.sh")
    parser.add_argument("--slurm_job_name_prefix", type=str, default="bert_brc_eval")

    parser.add_argument("--account", type=str, default="co_rail")
    parser.add_argument("--partition", type=str, default="savio4_gpu")
    parser.add_argument("--gpu_type", type=str, default="A5000")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_tasks", type=int, default=1)
    parser.add_argument("--cpus_per_task", type=int, default=4)
    parser.add_argument("--qos", type=str, default="rail_gpu4_high")
    parser.add_argument("--time", type=str, default="24:00:00")
    parser.add_argument("--mem", type=str, default="60G")
    parser.add_argument("--requeue", action="store_true", default=True)
    parser.add_argument("--no_requeue", action="store_false", dest="requeue")
    return parser.parse_args()


if __name__ == "__main__":
    generate_submission_script(parse_args())
