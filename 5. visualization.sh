#!/bin/bash

# Get model path from config using direct Python call
MODEL_PATH=$(python3 -c "
import sys, os
sys.path.append('.')
from config_utils.config_manager import ConfigManager
config = ConfigManager()
print(config.get_model_path('force_forcegnn'))
")

python config_utils/visualization.py --layout force --samples 5 --split train --force-model "$MODEL_PATH"