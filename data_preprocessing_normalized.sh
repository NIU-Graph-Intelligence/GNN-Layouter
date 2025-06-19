#!/bin/bash

# Activate the virtual environment (uncomment if you're using one)
# source .venv/bin/activate

# Run the data preprocessing script in normalized mode
python3 data/data_preprocessing.py data/raw/adjacency_matrices/adjacency_matrices.pkl data/raw/layouts/circular_layouts.pkl data/processed -n circular_layout


#python3 data/PreprocessingPaperWay.py data/raw/adjacency_matrices/adjacency_matrices.pkl data/raw/layouts/kamada_kawai_layouts.pkl data/processed -n kamada
#python3 data/kamada_data_preprocessing.py data/raw/adjacency_matrices/kamada15000graphs_adjacency_matrices.pkl data/raw/layouts/kamada_kawai_layouts.pkl data/processed -n kamada