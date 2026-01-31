# data/dataset.py
import torch
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from typing import List, Tuple
import random

def load_dataset(data_path: str) -> List[Data]:
    """Load PyTorch Geometric dataset from file"""
    try:
        dataset = torch.load(data_path, map_location='cpu', weights_only=False)
        if not isinstance(dataset, list):
            raise ValueError(f"Expected list of Data objects, got {type(dataset)}")
        print(f"Loaded {len(dataset)} graphs from {data_path}")
        return dataset
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from {data_path}: {e}")

def split_dataset(dataset: List[Data], 
                 splits: List[float] = [0.8, 0.1, 0.1],
                 random_state: int = 42) -> Tuple[List[Data], List[Data], List[Data]]:
    """Split dataset into train/val/test sets"""
    if abs(sum(splits) - 1.0) > 1e-6:
        raise ValueError(f"Splits must sum to 1.0, got {sum(splits)}")
    
    # Set random seed for reproducibility
    random.seed(random_state)
    dataset_shuffled = dataset.copy()
    random.shuffle(dataset_shuffled)
    
    n = len(dataset_shuffled)
    train_end = int(n * splits[0])
    val_end = train_end + int(n * splits[1])
    
    train_set = dataset_shuffled[:train_end]
    val_set = dataset_shuffled[train_end:val_end]
    test_set = dataset_shuffled[val_end:]
    
    print(f"Dataset split: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test")
    return train_set, val_set, test_set

def create_data_loaders(dataset: List[Data],
                       batch_size: int = 32,
                       splits: List[float] = [0.8, 0.1, 0.1],
                       random_state: int = 42,
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test data loaders"""
    
    # Split dataset
    train_set, val_set, test_set = split_dataset(dataset, splits, random_state)
    
    # Create data loaders
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader

def get_dataset_info(dataset: List[Data]) -> dict:
    """Get basic information about the dataset"""
    if not dataset:
        return {"error": "Empty dataset"}
    
    sample = dataset[0]
    node_counts = [data.num_nodes for data in dataset]
    edge_counts = [data.edge_index.shape[1] // 2 for data in dataset]  # Undirected
    
    info = {
        "num_graphs": len(dataset),
        "feature_dim": sample.x.shape[1],
        "output_dim": sample.y.shape[1],
        "avg_nodes": sum(node_counts) / len(node_counts),
        "min_nodes": min(node_counts),
        "max_nodes": max(node_counts),
        "avg_edges": sum(edge_counts) / len(edge_counts),
        "sample_graph_id": getattr(sample, 'graph_id', 'unknown'),
        "layout_type": getattr(sample, 'layout_type', 'unknown')
    }
    
    return info