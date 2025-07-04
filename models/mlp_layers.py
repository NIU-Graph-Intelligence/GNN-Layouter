import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from torch_geometric.nn import GATConv



def create_position_mlp(in_channels, hidden_channels, dropout_rate, use_layer_norm, output_dim):
         
    norm_layer = nn.LayerNorm if use_layer_norm else nn.BatchNorm1d
    
    layers = []
    current_dim = in_channels
    
    # Add hidden layers
    for hidden_dim in hidden_channels:
        layers.extend([
            nn.Linear(current_dim, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        current_dim = hidden_dim
    
    # Add output layer
    layers.append(nn.Linear(current_dim, output_dim))
    
    return nn.Sequential(*layers)


def create_radius_mlp(in_channels, hidden_channels, dropout_rate, use_layer_norm, constrained_output):
    
        
    norm_layer = nn.LayerNorm if use_layer_norm else nn.BatchNorm1d
    output_activation = nn.Sigmoid() if constrained_output else nn.Softplus()
    
    layers = []
    current_dim = in_channels
    
    # Add hidden layers
    for hidden_dim in hidden_channels:
        layers.extend([
            nn.Linear(current_dim, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        current_dim = hidden_dim
    
    # Add output layer with activation
    layers.extend([
        nn.Linear(current_dim, 1),
        output_activation
    ])
    
    return nn.Sequential(*layers)


def create_angle_mlp(in_channels, hidden_channels, dropout_rate, use_layer_norm):
    
        
    norm_layer = nn.LayerNorm if use_layer_norm else nn.BatchNorm1d
    
    layers = []
    current_dim = in_channels
    
    # Add hidden layers
    for hidden_dim in hidden_channels:
        layers.extend([
            nn.Linear(current_dim, hidden_dim),
            norm_layer(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        current_dim = hidden_dim
    
    # Add output layer with Tanh activation
    layers.extend([
        nn.Linear(current_dim, 1),
        nn.Tanh()
    ])
    
    return nn.Sequential(*layers)


def create_attention_mlp(in_channels):
    return nn.Sequential(
        nn.Linear(in_channels, 1),
        nn.Sigmoid()
    )


def create_force_directed_mlp(in_feat, out_feat, is_message_mlp = True) :

    if is_message_mlp:
        return nn.Sequential(
            nn.Linear(in_feat, in_feat, bias=False),
            nn.ReLU(),
            nn.Linear(in_feat, in_feat, bias=False),
        )
    else:
        return nn.Sequential(
            nn.Linear(2 * in_feat, in_feat, bias=False),
            nn.ReLU(),
            nn.Linear(in_feat, out_feat, bias=False)
        )


class MLPFactory:
    
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    @classmethod
    def create(cls, mlp_type: str, **kwargs) -> nn.Sequential:
        
        mlp_creators = {
            'position': create_position_mlp,
            'radius': create_radius_mlp,
            'angle': create_angle_mlp,
            'attention': create_attention_mlp,
            'force_directed': create_force_directed_mlp
        }
        
        if mlp_type not in mlp_creators:
            raise ValueError(f"Unknown MLP type: {mlp_type}. "
                           f"Available types: {list(mlp_creators.keys())}")
            
        mlp = mlp_creators[mlp_type](**kwargs)
        mlp.apply(cls._init_weights)
        return mlp 


class ConvolutionalBlock:

    @staticmethod
    def create_gin_block(in_dim: int, out_dim: int, dropout: float = 0.4) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )
    
    @staticmethod
    def create_attention_block(input_dim: int, head_dim: int, heads: int, dropout: float):

        hidden_channels = head_dim * heads
        return (
            GATConv(input_dim, head_dim, heads=heads, dropout=dropout),
            nn.LayerNorm(hidden_channels)
        )


class WeightInitializer:

    @staticmethod
    def xavier_uniform(module: nn.Module, gain: float = 0.5):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.zeros_(module.bias) 