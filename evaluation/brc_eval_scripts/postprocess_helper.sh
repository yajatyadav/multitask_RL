#!/bin/bash
#
# Postprocess best-of-N evaluation results for a given run ID
#
# Usage:
#   ./postprocess_best_of_n.sh <wandb_run_id> <env_name>
#
# Example:
#   ./postprocess_best_of_n.sh i7489boa "libero_90-living_room_scene1"
#

set -e  # Exit on error

# Check arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <wandb_run_id> <env_name>"
    echo "Example: $0 i7489boa \"libero_90-living_room_scene1\""
    exit 1
fi

WANDB_RUN_ID="$1"
ENV_NAME="libero_90-living_room_scene1"
WANDB_ENTITY="yajatyadav"
WANDB_PROJECT="multitask_RL"

# Find all matching eval result directories
EVAL_RESULTS_DIR="./eval_results"
PATTERN="${EVAL_RESULTS_DIR}/run_${WANDB_RUN_ID}_*"

echo "========================================="
echo "Postprocessing Best-of-N Evaluations"
echo "========================================="
echo "Run ID: ${WANDB_RUN_ID}"
echo "Environment: ${ENV_NAME}"
echo "Searching for: ${PATTERN}"
echo ""

# Count matching directories
MATCHING_DIRS=(${PATTERN})
if [ ! -d "${MATCHING_DIRS[0]}" ]; then
    echo "ERROR: No directories found matching pattern: ${PATTERN}"
    exit 1
fi

NUM_DIRS=${#MATCHING_DIRS[@]}
echo "Found ${NUM_DIRS} directory(ies) to process:"
for dir in "${MATCHING_DIRS[@]}"; do
    echo "  - ${dir}"
done
echo ""

# Submit jobs
SUBMITTED_JOBS=()
for OUTPUT_DIR in "${MATCHING_DIRS[@]}"; do
    echo "Submitting job for: ${OUTPUT_DIR}"
    
    JOB_ID=$(sbatch \
        -A co_rail \
        -p savio4_gpu \
        --gres=gpu:A5000:1 \
        -N 1 \
        -n 1 \
        -c 4 \
        --qos=rail_gpu4_high \
        -t 24:00:00 \
        --mem=60G \
        --parsable \
        --comment="postprocess_best_of_N_${WANDB_RUN_ID}" \
        --job-name="postproc_${WANDB_RUN_ID}" \
        scripts/automatic/run.sh \
        "uv run evaluation/brc_eval_scripts/postprocess_best_of_n_eval.py \
            --output_dir ${OUTPUT_DIR} \
            --wandb_entity ${WANDB_ENTITY} \
            --wandb_project ${WANDB_PROJECT} \
            --wandb_run_id ${WANDB_RUN_ID} \
            --env_name \"${ENV_NAME}\"")
    
    SUBMITTED_JOBS+=("${JOB_ID}")
    echo "  → Job ID: ${JOB_ID}"
    echo ""
done

echo "========================================="
echo "Summary"
echo "========================================="
echo "Submitted ${#SUBMITTED_JOBS[@]} job(s):"
for job_id in "${SUBMITTED_JOBS[@]}"; do
    echo "  - ${job_id}"
done
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Cancel all jobs with: scancel ${SUBMITTED_JOBS[*]}"
echo "========================================="