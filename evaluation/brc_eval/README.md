# BRC LIBERO BERT best-of-N

This folder contains a cluster workflow that mirrors
`evaluation/eval_libero_BERT_classifier_best_of_N.py` but splits work into many
SLURM jobs:

- one job = one `N` value + one env chunk
- per-job outputs go to `exp/multitask_RL/eval_intermediate/brc_eval_jobs`
- only final aggregated outputs go to `exp/multitask_RL/eval`

## Files

- `generate_eval_sbatch.py`: creates a single submission shell script with all `sbatch` calls and logging paths
- `eval_libero_BERT_classifier_best_of_N_single_job.py`: worker for a single `N` / env chunk
- `postprocess_best_of_n_eval.py`: aggregate all chunk outputs, save final outputs, and log metrics/videos to wandb

## Typical workflow

1. Generate submission script:

```bash
uv run evaluation/brc_eval/generate_eval_sbatch.py \
  --classifier_type success \
  --classifier_restore_dir "<classifier_dir>" \
  --classifier_ckpt_num 204 \
  --actor_restore_path "<actor_ckpt.pkl>" \
  --env_name "<env_name>" \
  --task_name "<task_name>" \
  --n_vals 1 2 4 8 16 32 64 128 \
  --num_env_chunks 8 \
  --wandb_group_name "<wandb_group>" \
  --wandb_run_name "<wandb_run>"
```

2. Submit jobs:

```bash
bash exp/multitask_RL/eval_intermediate/brc_eval_jobs/<wandb_run>_<timestamp>/submission_scripts/<generated_script>.sh
```

3. After all jobs are done, run postprocess:

```bash
uv run evaluation/brc_eval/postprocess_best_of_n_eval.py \
  --intermediate_run_dir exp/multitask_RL/eval_intermediate/brc_eval_jobs/<wandb_run>_<timestamp> \
  --results_save_path exp/multitask_RL/eval \
  --wandb_group_name "<wandb_group>" \
  --wandb_run_name "<wandb_run>" \
  --wandb_project multitask_RL \
  --wandb_entity yajatyadav \
  --wandb_run_id "<wandb_run_id>" \
  --n_vals 1 2 4 8 16 32 64 128
```
