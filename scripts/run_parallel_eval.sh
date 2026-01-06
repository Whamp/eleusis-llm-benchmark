#!/bin/bash
# Run multiple solo evaluations in parallel
# Edit the MODELS array below, then run: ./scripts/run_parallel_eval.sh

set -e

MODELS=(
    "openrouter:google/gemini-3-flash-preview"
    "hf:openai/gpt-oss-120b"
#    "openrouter:anthropic/claude-3.5-haiku"
)

echo "Running parallel evaluation for ${#MODELS[@]} models..."
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo ""

for model in "${MODELS[@]}"; do
    echo "[$(date '+%H:%M:%S')] Starting: $model"
    uv run python scripts/evaluate_single.py --player "$model" &
    sleep 2
done

wait
echo ""
echo "Done! Results in: results/"