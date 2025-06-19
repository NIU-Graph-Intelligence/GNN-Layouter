#!/bin/bash

# Activate the virtual environment (uncomment if you're using one)
# source .venv/bin/activate

# Run the training script
python3 training/train.py GNN_Model_1 data/processed/normalized_modelInput_circular_layout.pt 128