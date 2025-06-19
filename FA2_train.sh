#!/bin/bash

python3 training/FA2_train.py --model FA2GNN --data data/processed/modelInput_FT21024.pt --layout FT2 --batch-size 32 --epochs 2000 --lr 1e-3 --hidden-channels 256 --num-layers 6 --scaling-ratio 0.025 --gravity-coef 0.5 --edge-weight-influence 1.0 --barnes-hut-theta 0.5 --initial-temperature 1.0 --temperature-decay 0.9 --fast-mode --seed 42