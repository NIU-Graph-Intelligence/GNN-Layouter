#!/bin/bash

python3 utils/FR-Visualization.py --ignn IGNN_checkpoints/IGNN_bs16_ep1000HC64.pt \
    --forcegnn results/metrics/ForceGNN/Weights_ForceGNN_FR_batch16InitialCoordinatesFeaturesOnly40Nodes.pt \
    --data data/processed/modelInput_FRInitialCoordinatesFeaturesOnly40Nodes.pt --samples 5 --outdir comparisons --seed 42 