#!/bin/bash
# Run multiple solo evaluations in parallel
#
# Usage:
#   ./scripts/run_parallel_eval.sh [OPTIONS]
#
# Options:
#   -m, --models FILE    File containing model specs, one per line (default: use inline list)
#   -n, --num-rounds N   Number of rounds per evaluation (default: from config.yaml)
#   -r, --rule-index N   Starting rule index (default: from config.yaml)
#   -t, --max-turns N    Maximum turns per round (default: from config.yaml)
#   -j, --jobs N         Maximum parallel jobs (default: number of models)
#   -c, --config FILE    Base config file (default: config.yaml)
#   -h, --help           Show this help message
#
# Examples:
#   # Run evaluations for 3 models in parallel
#   ./scripts/run_parallel_eval.sh
#
#   # Use custom models file
#   ./scripts/run_parallel_eval.sh -m models.txt
#
#   # Run 10 rounds each with max 2 parallel jobs
#   ./scripts/run_parallel_eval.sh -n 10 -j 2
#
#   # Specify models inline (edit DEFAULT_MODELS below)
#   ./scripts/run_parallel_eval.sh -n 20

set -e

# Default models to evaluate (edit this list as needed)
DEFAULT_MODELS=(
    "openrouter:anthropic/claude-3.5-haiku"
    "openrouter:google/gemini-2.0-flash-001"
    "openrouter:openai/gpt-4o-mini"
)

# Parse command-line arguments
MODELS_FILE=""
NUM_ROUNDS=""
RULE_INDEX=""
MAX_TURNS=""
MAX_JOBS=""
CONFIG_FILE="config.yaml"

print_help() {
    head -30 "$0" | grep "^#" | sed 's/^# \?//'
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--models)
            MODELS_FILE="$2"
            shift 2
            ;;
        -n|--num-rounds)
            NUM_ROUNDS="$2"
            shift 2
            ;;
        -r|--rule-index)
            RULE_INDEX="$2"
            shift 2
            ;;
        -t|--max-turns)
            MAX_TURNS="$2"
            shift 2
            ;;
        -j|--jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
done

# Load models from file or use defaults
if [[ -n "$MODELS_FILE" ]]; then
    if [[ ! -f "$MODELS_FILE" ]]; then
        echo "Error: Models file not found: $MODELS_FILE"
        exit 1
    fi
    mapfile -t MODELS < "$MODELS_FILE"
else
    MODELS=("${DEFAULT_MODELS[@]}")
fi

# Filter out empty lines and comments
FILTERED_MODELS=()
for model in "${MODELS[@]}"; do
    # Trim whitespace
    model=$(echo "$model" | xargs)
    # Skip empty lines and comments
    if [[ -n "$model" && ! "$model" =~ ^# ]]; then
        FILTERED_MODELS+=("$model")
    fi
done
MODELS=("${FILTERED_MODELS[@]}")

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "Error: No models specified"
    exit 1
fi

# Set max jobs (default: number of models)
if [[ -z "$MAX_JOBS" ]]; then
    MAX_JOBS=${#MODELS[@]}
fi

echo "========================================"
echo "Parallel Solo Evaluation Runner"
echo "========================================"
echo ""
echo "Models to evaluate: ${#MODELS[@]}"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo ""
echo "Max parallel jobs: $MAX_JOBS"
[[ -n "$NUM_ROUNDS" ]] && echo "Rounds per model: $NUM_ROUNDS"
[[ -n "$RULE_INDEX" ]] && echo "Starting rule index: $RULE_INDEX"
[[ -n "$MAX_TURNS" ]] && echo "Max turns per round: $MAX_TURNS"
echo "Config file: $CONFIG_FILE"
echo ""
echo "========================================"
echo ""

# Build base command arguments
BASE_ARGS="--config $CONFIG_FILE"
[[ -n "$NUM_ROUNDS" ]] && BASE_ARGS="$BASE_ARGS --num-rounds $NUM_ROUNDS"
[[ -n "$RULE_INDEX" ]] && BASE_ARGS="$BASE_ARGS --rule-index $RULE_INDEX"
[[ -n "$MAX_TURNS" ]] && BASE_ARGS="$BASE_ARGS --max-turns $MAX_TURNS"

# Track PIDs and models for status reporting
declare -A PIDS
declare -A MODEL_NAMES

# Function to wait for a job slot
wait_for_slot() {
    while [[ $(jobs -r | wc -l) -ge $MAX_JOBS ]]; do
        sleep 1
    done
}

# Function to check and report completed jobs
check_completed() {
    for pid in "${!PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid"
            exit_code=$?
            model="${PIDS[$pid]}"
            if [[ $exit_code -eq 0 ]]; then
                echo "[$(date '+%H:%M:%S')] ✓ Completed: $model"
            else
                echo "[$(date '+%H:%M:%S')] ✗ Failed (exit $exit_code): $model"
            fi
            unset PIDS[$pid]
        fi
    done
}

# Start evaluations
echo "Starting evaluations..."
echo ""

for model in "${MODELS[@]}"; do
    # Wait for available slot
    wait_for_slot

    # Check for completed jobs
    check_completed

    echo "[$(date '+%H:%M:%S')] Starting: $model"

    # Run evaluation in background
    uv run python scripts/evaluate_single.py $BASE_ARGS --player "$model" &
    pid=$!
    PIDS[$pid]="$model"

    # Small delay to avoid race conditions in output folders
    sleep 2
done

# Wait for all remaining jobs
echo ""
echo "Waiting for remaining jobs to complete..."
while [[ ${#PIDS[@]} -gt 0 ]]; do
    check_completed
    sleep 5
done

echo ""
echo "========================================"
echo "All evaluations complete!"
echo "========================================"
echo ""
echo "Results saved in: results/"
ls -la results/ | grep solo_evaluation | tail -${#MODELS[@]}
