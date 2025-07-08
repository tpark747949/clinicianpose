#!/bin/bash

# Set the seed for reproducibility
SEED=42

# Set the number of GPUs to use (modify this as needed)
GPUS=1

# Set the work directory where all logs and weights will be saved
WORK_DIR_BASE="./weights"

# Function to run the training command
run_training() {
    CONFIG_FILE=$1
    WORK_DIR=$2

    echo "Starting training with $CONFIG_FILE..."

    # Run the training
    python ./tools/train.py $CONFIG_FILE --gpus $GPUS --work-dir $WORK_DIR --seed $SEED

    # Check if the training was successful
    if [ $? -eq 0 ]; then
        echo "Training with $CONFIG_FILE completed successfully."
    else
        echo "Error occurred during training with $CONFIG_FILE."
        exit 1  # Exit if any of the trainings fail
    fi
}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# Train the first model
run_training "configs/hmp_huge.py" "$WORK_DIR_BASE/heatmap"

# Train the second model
run_training "configs/pct_huge_tokenizer.py" "$WORK_DIR_BASE/tokenizer"

# Train the third model
run_training "configs/pct_huge_classifier.py" "$WORK_DIR_BASE/pct"

echo "All training jobs completed!"
