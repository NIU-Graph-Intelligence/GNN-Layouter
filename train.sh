#!/bin/bash

# Activate the virtual environment (uncomment if you're using one)
# source .venv/bin/activate

# Run the training script with GCN model
python3 training/train.py \
    --model GCN \
    --data_path data/processed/normalized_modelInput_circular_layout.pt \
    --batch_size 32 \
    --num_epochs 2000 \
    --learning_rate 0.002 \
    --weight_decay 0.005