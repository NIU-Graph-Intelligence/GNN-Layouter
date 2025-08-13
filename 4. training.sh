#!/bin/bash

# Train ForceGNN with config defaults (parameters automatically loaded from config.json)
python3 training/train.py --model ForceGNN --layout_type force_directed

# Train ChebConv with config defaults (parameters automatically loaded from config.json)
#python3 training/train.py --model ChebConv --layout_type circular

# Optional: Override specific parameters if needed
# python3 training/train.py --model ForceGNN --layout_type force_directed --batch_size 128 --learning_rate 0.001