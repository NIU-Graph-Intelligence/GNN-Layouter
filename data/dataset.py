from sklearn.model_selection import train_test_split
import os
import torch
from torch_geometric.data import Data, DataLoader
import numpy as np
from typing import List, Tuple, Optional

def _split_dataset(dataset, splits, random_state):
   
    # Validate split ratios
    train_ratio, val_ratio, test_ratio = splits
    assert abs(sum(splits) - 1.0) < 1e-6, f"Split ratios must sum to 1.0, got {sum(splits)}"
    
    # Set random seeds for reproducibility
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    # First split: separate test set
    train_val_ratio = train_ratio + val_ratio
    train_val_data, test_data = train_test_split(
        dataset,
        train_size=train_val_ratio,
        random_state=random_state
    )
    
    # Second split: separate train and validation
    train_data, val_data = train_test_split(
        train_val_data,
        train_size=train_ratio/train_val_ratio,
        random_state=random_state
    )
    
    # Print split information
    print(f"Dataset splits:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Train samples: {len(train_data)} ({len(train_data)/len(dataset):.1%})")
    print(f"  Val samples: {len(val_data)} ({len(val_data)/len(dataset):.1%})")
    print(f"  Test samples: {len(test_data)} ({len(test_data)/len(dataset):.1%})")
    
    return train_data, val_data, test_data

def _create_data_loaders(train_data, val_data, test_data, batch_size, val_batch_size, test_batch_size, follow_attrs, shuffle_train = True):
    
    if val_batch_size is None:
        val_batch_size = batch_size
    if test_batch_size is None:
        test_batch_size = batch_size
        
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=shuffle_train,
        follow_batch=follow_attrs
    )
    
    val_loader = DataLoader(
        val_data, 
        batch_size=val_batch_size, 
        shuffle=False,
        follow_batch=follow_attrs
    )
    
    test_loader = DataLoader(
        test_data, 
        batch_size=test_batch_size, 
        shuffle=False,
        follow_batch=follow_attrs
    )
    
    return train_loader, val_loader, test_loader

def circular_data_loader(dataset, batch_size = 32, splits= (0.8, 0.1, 0.1), random_state = 42, val_batch_size = None, test_batch_size = None):
    
    print("Setting up circular layout data loaders...")
    
    # Split dataset
    train_data, val_data, test_data = _split_dataset(dataset, splits, random_state)
    
    # Create data loaders with circular layout specific attributes
    return _create_data_loaders(
        train_data, val_data, test_data,
        batch_size, val_batch_size, test_batch_size,
        follow_attrs=['x', 'edge_index', 'y']
    )

def force_directed_data_loader(dataset, batch_size= 128, splits = (0.8, 0.1, 0.1), random_state = 42, val_batch_size = None, test_batch_size = None):
   
    print("Setting up force-directed layout data loaders...")
    
    # Split dataset
    train_data, val_data, test_data = _split_dataset(dataset, splits, random_state)
    
    # Create data loaders with force-directed layout specific attributes
    return _create_data_loaders(
        train_data, val_data, test_data,
        batch_size, val_batch_size, test_batch_size,
        follow_attrs=['x', 'edge_index', 'init_coords', 'original_y']
    )

# For backward compatibility
# data_loader = force_directed_data_loader

