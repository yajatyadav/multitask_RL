#!/bin/bash

#SBATCH --nodes 1
#SBATCH --output /global/scratch/users/yajatyadav/research/multitask_reinforcement_learning/slurm_logs/%j.out

a=''
while (( "$#" )); do
  a="$a '$1'"
  shift
done
srun -K0 bash -c "scripts/automation/run_supp.sh $a"
