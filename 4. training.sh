#!/bin/bash

python3 training/train.py --model ForceGNN --data_path data/processed/processed_forcedirected_onehot.pt --batch_size 32 --num_epochs 2000 --learning_rate 0.002 --weight_decay 0.005 --layout_type force_directed