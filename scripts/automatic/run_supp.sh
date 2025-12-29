#!/usr/bin/env bash

#SBATCH --nodes 1
#SBATCH --output /global/scratch/users/yajatyadav/research/multitask_reinforcement_learning/slurm_logs/%j.out
#SBATCH --error /global/scratch/users/yajatyadav/research/multitask_reinforcement_learning/slurm_logs/slurm-%j.err

echo $SLURM_JOB_ID.$SLURM_PROCID $@
args=("$@")
bash -c "${args[$SLURM_PROCID]}"