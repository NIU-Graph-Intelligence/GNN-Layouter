#!/bin/bash


python3 data/preprocess_data.py --input data/raw/layouts/combined_fr_layouts.pkl --adj-matrices data/raw/adjacency_matrices/combined_adjacency_matrices.pkl --features onehot --init-positions data/raw/layouts/combined_initial_positions.pkl --layout-type forcedirected