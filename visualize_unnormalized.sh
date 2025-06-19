#!/bin/bash

# Activate virtual environment if needed
# source venv/bin/activate

# Change to project root directory
cd "$(dirname "$0")"

# Print current directory and check if file exists
echo "Current directory: $(pwd)"
echo "Checking if model file exists..."
if [ -f "results/metrics/Exp2-Node40/best_model_8.pt" ]; then
    echo "Model file found!"
else
    echo "Model file not found!"
    exit 1
fi

# Run visualization for unnormalized model
echo "Running visualization script..."
python utils/visualization.py results/metrics/Exp2-Node40/best_model_8.pt

echo "Visualization completed. Check results/figures for the output."

