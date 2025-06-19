#!/bin/bash

# normalized
#python3 training/train.py GNN_ChebConv data/processed/normalized_modelInput_circular_layout.pt 256

# Unnormalized
#python3 training/train.py GNN_ChebConv data/processed/unnormalized_modelInput_circular_layout.pt 256

# Without Positional feature
python3 training/train.py GNN_ChebConv data/processed/normalized_WithoutPositionalFeaturemodelInput_circular_layout.pt 128
