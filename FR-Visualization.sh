#!/bin/bash

export PYTHONPATH=".:$PYTHONPATH"     # ensure project root is on PYTHONPATH

python3 utils/FR-Visualization.py --ignn IGNN_checkpoints/IGNN_bs16_ep1000_HC64WithoutInputOneHot.pt \
    --forcegnn results/metrics/ForceGNN/Weights_ForceGNN_FR_batch16WithoutOneHotX.pt \
    --data data/processed/modelInput_FRWithoutInputOneHot1024.pt --samples 5 --outdir comparisons --seed 42