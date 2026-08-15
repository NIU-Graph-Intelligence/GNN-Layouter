# config_utils.py
import yaml
import os
from typing import Dict, Any, List

class ConfigManager:
    def __init__(self, global_config_path: str = "config.yaml"):
        self.global_config_path = global_config_path
        self.global_config = self._load_global_config()
    
    def _load_global_config(self) -> Dict[str, Any]:
        """Load global configuration"""
        if os.path.exists(self.global_config_path):
            with open(self.global_config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'paths': {
                    'data_root': 'data',
                    'graphs': 'data/graphs',
                    'layouts': 'data/layouts',
                    'processed': 'data/processed',
                    'experiment_configs': 'configs'
                },
                'defaults': {
                    'seed': 42,
                    'num_nodes': 50
                }
            }
    
    def get_path(self, path_name: str) -> str:
        """Get path configuration"""
        return self.global_config['paths'].get(path_name, f"data/{path_name}")
    
    def get_default(self, key: str, default_value: Any = None) -> Any:
        """Get default value"""
        return self.global_config['defaults'].get(key, default_value)
    
    def ensure_directories(self):
        """Ensure all necessary directories exist"""
        for path_name, path_value in self.global_config['paths'].items():
            if path_name != 'experiment_configs':  # configs may not be under data
                os.makedirs(path_value, exist_ok=True)
        
        # Ensure experiment configs directory exists
        configs_dir = self.global_config['paths']['experiment_configs']
        os.makedirs(configs_dir, exist_ok=True)

def load_experiment_config(config_name: str) -> Dict[str, Any]:
    """Load experiment configuration file for data generation"""
    config_manager = ConfigManager()
    configs_dir = config_manager.get_path('experiment_configs')
    
    # Support with or without .yaml suffix
    if not config_name.endswith('.yaml'):
        config_name += '.yaml'
    
    config_path = os.path.join(configs_dir, config_name)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_training_config(config_path: str = "training_config.yaml") -> Dict[str, Any]:
    """Load training configuration file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Training config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def find_files_by_patterns(directory: str, patterns: List[str]) -> List[str]:
    """Find files by patterns"""
    import glob
    
    matched_files = []
    for pattern in patterns:
        full_pattern = os.path.join(directory, pattern)
        matched = glob.glob(full_pattern)
        matched_files.extend(matched)
    
    # Remove duplicates and sort
    return sorted(list(set(matched_files)))