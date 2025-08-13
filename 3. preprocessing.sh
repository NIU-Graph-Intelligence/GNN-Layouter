#!/bin/bash

# Get paths from config using direct Python calls
LAYOUTS_PATH=$(python3 -c "
import sys, os
sys.path.append('.')
from config_utils.config_manager import ConfigManager
config = ConfigManager()
print(config.get_data_path('layouts'))
")

ADJ_MATRICES_PATH=$(python3 -c "
import sys, os
sys.path.append('.')
from config_utils.config_manager import ConfigManager
config = ConfigManager()
print(config.get_data_path('adjacency_matrices'))
")

python3 data/preprocess_data.py \
  --input "$LAYOUTS_PATH/LFR_fr_layouts5000.pkl" \
  --adj-matrices "$ADJ_MATRICES_PATH/LFR_adjacency_matrices5000.pkl" \
  --features degree \
  --init-positions "$LAYOUTS_PATH/LFR_initial_positions5000.pkl" \
  --layout-type forcedirected