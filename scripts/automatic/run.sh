#!/usr/bin/env bash

#SBATCH --nodes 1
#SBATCH --output /global/scratch/users/yajatyadav/research/multitask_reinforcement_learning/slurm_logs/slurm-%j.out
#SBATCH --error /global/scratch/users/yajatyadav/research/multitask_reinforcement_learning/slurm_logs/slurm-%j.err

a=''
while (( "$#" )); do
  a="$a '$1'"
  shift
done
srun -K0 bash -c "scripts/automatic/run_supp.sh $a"
