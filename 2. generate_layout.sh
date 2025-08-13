#!/bin/bash

# Get paths from config using direct Python call
GRAPHS_PATH=$(python3 -c "
import sys, os
sys.path.append('.')
from config_utils.config_manager import ConfigManager
config = ConfigManager()
print(config.get_data_path('graphs'))
")
GRAPH_FILE="$GRAPHS_PATH/graphs_dataset5000.pkl"

python3 data/generate_layouts.py --input "$GRAPH_FILE" --layout-type force-directed --algorithms FR