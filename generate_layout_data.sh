#!/bin/bash

# Activate the virtual environment (uncomment if you're using one)
# source .venv/bin/activate

# Run the layout generation script
python3 data/generate_layout_data.py data/raw/graph_dataset/Kamada15000graphs_dataset.pkl data/raw/layouts data/raw/adjacency_matrices kamada_kawai
