#!/bin/bash


python3 data/preprocess_data.py --input data/raw/layouts/LFR_fr_layouts5000.pkl --adj-matrices data/raw/adjacency_matrices/LFR_adjacency_matrices5000.pkl --features degree --init-positions data/raw/layouts/LFR_initial_positions5000.pkl --layout-type forcedirected