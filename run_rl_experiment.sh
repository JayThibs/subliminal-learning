#!/bin/bash
# Example script to run a complete RL subliminal learning experiment

set -e  # Exit on error

echo "=== RL Subliminal Learning Experiment ==="
echo

# Configuration
GOLDEN_DATASET="data/owl_numbers_animals.jsonl"
OUTPUT_DIR="output/rl_owl_experiment"
TARGET_ANIMAL="owl"
MODEL="gpt-4o-mini"
SUFFIX="owl-rl-$(date +%Y%m%d)"

# Step 1: Analyze golden dataset and prepare for RL
echo "Step 1: Preparing RL fine-tuning..."
python scripts/rl_finetune.py "$GOLDEN_DATASET" "$OUTPUT_DIR" \
    --model "$MODEL" \
    --suffix "$SUFFIX" \
    --n-epochs 5 \
    --dry-run

echo
echo "Dry run complete! Check the generated files in $OUTPUT_DIR"
echo
echo "Files created:"
ls -la "$OUTPUT_DIR"

echo
echo "To run the actual RL fine-tuning, remove the --dry-run flag:"
echo "python scripts/rl_finetune.py $GOLDEN_DATASET $OUTPUT_DIR --model $MODEL --suffix $SUFFIX --n-epochs 5"

echo
echo "After the job completes, evaluate with:"
echo "python scripts/evaluate_trait.py $MODEL ft:$MODEL:$SUFFIX:JOB_ID $TARGET_ANIMAL --compare --output $OUTPUT_DIR/results/"