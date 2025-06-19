#!/bin/bash

# Activate the virtual environment (uncomment if you're using one)
# source .venv/bin/activate

# Run the data preprocessing script in unnormalized mode
python3 data/data_preprocessing.py data/raw/adjacency_matrices/adjacency_matrices.pkl data/raw/layouts/circular_layouts.pkl data/processed -u circular_layout