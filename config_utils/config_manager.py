#!/usr/bin/env python3
"""
Configuration Management Module
Centralized configuration loading and access for GNN-Layouter project
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """Centralized configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default to config.json in project root
            # Since we're in config_utils/ directory, go up one level to project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / 'config.json'
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path (e.g., 'models.ForceGNN.hidden_dim')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        current = self._config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model configuration"""
        return self.get(f'models.{model_name}', {})
    
    def get_training_config(self, layout_type: str) -> Dict[str, Any]:
        """Get training configuration for layout type"""
        return self.get(f'training.{layout_type}', {})
    
    def get_optimization_config(self, layout_type: str) -> Dict[str, Any]:
        """Get optimization configuration for layout type"""
        return self.get(f'optimization.{layout_type}', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.get('data', {})
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration"""
        return self.get('visualization', {})
    
    def get_path(self, path_key: str, default: str = None) -> str:
        """
        Get a path from configuration
        
        Args:
            path_key: Dot-separated path key (e.g., 'data.processed', 'results.figures')
            default: Default path if not found
            
        Returns:
            Absolute path as string
        """
        relative_path = self.get(f'paths.{path_key}', default)
        if relative_path is None:
            raise ValueError(f"Path not found in config: paths.{path_key}")
        
        # Convert to absolute path
        if not os.path.isabs(relative_path):
            project_root = self.config_path.parent
            return str(project_root / relative_path)
        return relative_path
    
    def get_data_path(self, data_type: str = None) -> str:
        """Get data path, optionally for specific data type"""
        if data_type:
            return self.get_path(f'data.{data_type}')
        return self.get_path('data.base')
    
    def get_model_path(self, model_key: str) -> str:
        """Get path to model checkpoint"""
        return self.get_path(f'models.{model_key}')
    
    def get_results_path(self, result_type: str = 'base') -> str:
        """Get results path"""
        return self.get_path(f'results.{result_type}')
    
    def get_figures_path(self) -> str:
        """Get figures output path"""
        return self.get_path('results.figures')
        
    def get_logs_path(self, log_type: str = 'base') -> str:
        """Get logs path"""
        return self.get_path(f'logs.{log_type}')
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration"""
        return self.get('evaluation', {})
    
    def get_mlp_config(self, mlp_type: str) -> Dict[str, Any]:
        """Get MLP layer configuration"""
        return self.get(f'mlp_layers.{mlp_type}', {})
    
    def update(self, key_path: str, value: Any) -> None:
        """
        Update configuration value using dot notation
        
        Args:
            key_path: Dot-separated path
            value: New value
        """
        keys = key_path.split('.')
        current = self._config
        
        # Navigate to parent of target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the value
        current[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to file"""
        save_path = Path(path) if path else self.config_path
        with open(save_path, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary"""
        return self._config.copy()


# Global configuration instance
_config_instance = None

def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None or config_path is not None:
        _config_instance = ConfigManager(config_path)
    return _config_instance

def reload_config(config_path: Optional[str] = None) -> ConfigManager:
    """Reload configuration from file"""
    global _config_instance
    _config_instance = ConfigManager(config_path)
    return _config_instance


# Convenience functions for common config access patterns
def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get model configuration"""
    return get_config().get_model_config(model_name)

def get_training_config(layout_type: str) -> Dict[str, Any]:
    """Get training configuration"""
    return get_config().get_training_config(layout_type)

def get_optimization_config(layout_type: str) -> Dict[str, Any]:
    """Get optimization configuration"""
    return get_config().get_optimization_config(layout_type)

def get_data_config() -> Dict[str, Any]:
    """Get data configuration"""
    return get_config().get_data_config()

def get_visualization_config() -> Dict[str, Any]:
    """Get visualization configuration"""
    return get_config().get_visualization_config()

def get_path(path_key: str, default: str = None) -> str:
    """Get a path from configuration"""
    return get_config().get_path(path_key, default)

def get_data_path(data_type: str = None) -> str:
    """Get data path, optionally for specific data type"""
    return get_config().get_data_path(data_type)

def get_model_path(model_key: str) -> str:
    """Get path to model checkpoint"""
    return get_config().get_model_path(model_key)

def get_results_path(result_type: str = 'base') -> str:
    """Get results path"""
    return get_config().get_results_path(result_type)

def get_figures_path() -> str:
    """Get figures output path"""
    return get_config().get_figures_path()

def get_logs_path(log_type: str = 'base') -> str:
    """Get logs path"""
    return get_config().get_logs_path(log_type)


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    
    print("Testing configuration access:")
    print(f"ForceGNN hidden_dim: {config.get('models.ForceGNN.hidden_dim')}")
    print(f"Circular training epochs: {config.get('training.circular.num_epochs')}")
    print(f"Data splits: {config.get('data.splits')}")
    print(f"Visualization output dir: {config.get('visualization.output_dir')}")
    
    # Test model-specific access
    print(f"\nForceGNN config: {config.get_model_config('ForceGNN')}")
    print(f"Circular training config: {config.get_training_config('circular')}")
