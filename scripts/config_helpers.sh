#!/bin/bash
# config_helpers.sh - Helper functions for accessing config paths in shell scripts
# Usage: source scripts/config_helpers.sh

# Function to get data paths
get_data_path() {
    python3 -c "
import sys, os
sys.path.append('.')
from config_utils.config_manager import ConfigManager
config = ConfigManager()
print(config.get_data_path('$1'))
"
}

# Function to get model paths
get_model_path() {
    python3 -c "
import sys, os
sys.path.append('.')
from config_utils.config_manager import ConfigManager
config = ConfigManager()
print(config.get_model_path('$1'))
"
}

# Function to get results paths
get_results_path() {
    python3 -c "
import sys, os
sys.path.append('.')
from config_utils.config_manager import ConfigManager
config = ConfigManager()
print(config.get_results_path('$1'))
"
}

# Function to get figures path
get_figures_path() {
    python3 -c "
import sys, os
sys.path.append('.')
from config_utils.config_manager import ConfigManager
config = ConfigManager()
print(config.get_figures_path())
"
}

# Function to get logs paths
get_logs_path() {
    python3 -c "
import sys, os
sys.path.append('.')
from config_utils.config_manager import ConfigManager
config = ConfigManager()
print(config.get_logs_path('$1'))
"
}
