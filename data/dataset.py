from sklearn.model_selection import train_test_split
import os
import torch
from torch_geometric.data import Data, DataLoader
import numpy as np

def data_loader(dataset, batch_size=128, splits=(0.7, 0.15, 0.15), random_state=42, val_batch_size=None, test_batch_size=None):
    """
    Split dataset into train, validation, and test sets with consistent splits.
    
    Args:
        dataset: List of graph data objects
        batch_size: Batch size for training data
        splits: Tuple of (train_ratio, val_ratio, test_ratio) that sum to 1.0
        random_state: Random seed for reproducibility
        val_batch_size: Batch size for validation data (defaults to batch_size)
        test_batch_size: Batch size for test data (defaults to batch_size)
    
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
    """
    if val_batch_size is None:
        val_batch_size = batch_size
    if test_batch_size is None:
        test_batch_size = batch_size
        
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
    
    # Second split: separate train and validation from remaining data
    train_data, val_data = train_test_split(
        train_val_data,
        train_size=train_ratio/train_val_ratio,  # Adjust ratio for remaining data
        random_state=random_state
    )
    
    print(f"Dataset splits:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Train samples: {len(train_data)} ({len(train_data)/len(dataset):.1%})")
    print(f"  Val samples: {len(val_data)} ({len(val_data)/len(dataset):.1%})")
    print(f"  Test samples: {len(test_data)} ({len(test_data)/len(dataset):.1%})")

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

