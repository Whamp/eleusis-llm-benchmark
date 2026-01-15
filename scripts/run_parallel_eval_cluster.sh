#!/bin/bash
# Run multiple solo evaluations in parallel on SLURM cluster
# Usage: ./scripts/run_parallel_eval_cluster.sh models.txt [config.yaml]

#SBATCH --partition=hopper-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=23:59:59
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# If not running in Slurm, submit the job
if [ -z "$SLURM_JOB_ID" ]; then
    if [ $# -eq 0 ]; then
        echo "Usage: $0 <models_file> [config_file]"
        echo "  models_file: Each line should be a model key from models.yaml"
        echo "  config_file: Optional, defaults to config.yaml"
        exit 1
    fi

    MODELS_FILE="$1"
    CONFIG_FILE="${2:-config.yaml}"

    if [ ! -f "$MODELS_FILE" ]; then
        echo "Error: Models file not found: $MODELS_FILE"
        exit 1
    fi

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Config file not found: $CONFIG_FILE"
        exit 1
    fi

    JOB_NAME="eval_$(basename "$MODELS_FILE" .txt)"
    echo "Submitting job: $JOB_NAME"
    sbatch --job-name="$JOB_NAME" "$0" "$MODELS_FILE" "$CONFIG_FILE"
    exit 0
fi

# Running on compute node
MODELS_FILE="$1"
CONFIG_FILE="$2"

set -e

echo "========================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo "Models file: $MODELS_FILE"
echo "Config file: $CONFIG_FILE"
echo "========================================"

# Activate virtual environment
source .venv/bin/activate
echo "Python: $(which python)"

# Read models from file (skip empty lines and comments)
MODELS=()
while IFS= read -r line || [ -n "$line" ]; do
    line=$(echo "$line" | sed 's/#.*//' | xargs)
    [ -z "$line" ] && continue
    MODELS+=("$line")
done < "$MODELS_FILE"

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Error: No models found in $MODELS_FILE"
    exit 1
fi

echo ""
echo "Running parallel evaluation for ${#MODELS[@]} models:"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo ""

for model in "${MODELS[@]}"; do
    echo "[$(date '+%H:%M:%S')] Starting: $model"
    uv run python scripts/evaluate_single.py --config "$CONFIG_FILE" --player "$model" &
    sleep 2
done

wait

echo ""
echo "========================================"
echo "Job completed at: $(date)"
echo "Results in: results/"
echo "========================================"
