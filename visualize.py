#!/usr/bin/env python3
"""
Simple Graph Layout Visualization Tool
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from data.dataset import load_dataset, create_data_loaders
from models import *  # Import all available models

class GraphVisualizer:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model mapping from checkpoint path to model class
        self.model_mapping = {
            'ForceGNN': ForceGNN,
            'GCN': GCN,
            'GAT': GAT,
            'GIN': GIN,
            'ChebNet': ChebNet,
        }
    
    def detect_model_type(self, model_path):
        """Detect model type from checkpoint path"""
        path_lower = model_path.lower()
        
        for model_name in self.model_mapping:
            if model_name.lower() in path_lower:
                return self.model_mapping[model_name]
        
        # Default fallback
        print(f"Warning: Could not detect model type from {model_path}, using GCN")
        return GCN
    
    def load_model(self, checkpoint_path, input_dim):
        """Load model from checkpoint"""
        # Detect model class
        ModelClass = self.detect_model_type(checkpoint_path)
        
        # Create model instance
        try:
            model = ModelClass(input_dim=input_dim)
        except TypeError:
            # Some models might need different parameter names
            model = ModelClass(input_dim)
        
        model = model.to(self.device)
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def get_node_colors(self, data):
        """Get node colors based on community info or default"""
        if hasattr(data, 'community') and data.community is not None:
            # Use community-based colors
            communities = data.community.cpu().numpy()
            unique_comms = np.unique(communities)
            
            # Predefined colors
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            # Extend if needed
            while len(colors) < len(unique_comms):
                colors.extend(colors)
            
            # Map communities to colors
            color_map = {comm: colors[i] for i, comm in enumerate(unique_comms)}
            node_colors = [color_map[comm] for comm in communities]
            
            return node_colors, f"({len(unique_comms)} communities)"
        else:
            # Use default colors
            num_nodes = data.num_nodes
            colors = plt.cm.Set3(np.linspace(0, 1, num_nodes))
            return colors, ""
    
    def create_networkx_graph(self, data):
        """Convert PyG data to NetworkX graph"""
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        
        # Add edges (handle both directed and undirected)
        edge_list = data.edge_index.cpu().numpy().T
        edges = [(int(u), int(v)) for u, v in edge_list if u != v]
        G.add_edges_from(edges)
        
        return G
    
    def visualize(self, model_path, data_path, num_samples=5, split='test', 
                 output_dir='visualizations', seed=42):
        """Main visualization function"""
        
        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        print(f"Loading dataset from {data_path}")
        dataset = load_dataset(data_path)
        print(f"Loaded {len(dataset)} graphs")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset, batch_size=1, random_state=seed
        )
        
        # Select split
        if split == 'train':
            loader = train_loader
        elif split == 'val':
            loader = val_loader
        else:
            loader = test_loader
        
        # Extract graphs
        graphs = [batch for batch in loader]
        print(f"Using {len(graphs)} graphs from {split} split")
        
        # Sample graphs
        num_samples = min(num_samples, len(graphs))
        sample_indices = random.sample(range(len(graphs)), num_samples)
        
        # Load model
        input_dim = graphs[0].x.shape[1]
        print(f"Loading model with input_dim={input_dim}")
        model = self.load_model(model_path, input_dim)
        
        # Detect model name for labeling
        model_name = model.__class__.__name__
        print(f"Loaded {model_name} model")
        
        # Setup figure
        fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        print(f"\\nGenerating visualizations...")
        
        for row, idx in enumerate(sample_indices):
            data = graphs[idx].to(self.device)
            
            # Get ground truth coordinates
            true_coords = data.y.cpu().numpy()
            
            # Get model predictions
            with torch.no_grad():
                pred_coords = model(data.x, data.edge_index).cpu().numpy()
            
            # Create NetworkX graph
            G = self.create_networkx_graph(data)
            
            # Get node colors and info
            node_colors, color_info = self.get_node_colors(data)
            
            # Graph info - handle different data types
            graph_id = getattr(data, 'graph_id', f'Graph_{idx}')
            if isinstance(graph_id, (list, tuple)):
                graph_id = graph_id[0] if len(graph_id) > 0 else f'Graph_{idx}'
            
            layout_type = getattr(data, 'layout_type', 'unknown')
            if isinstance(layout_type, (list, tuple)):
                layout_type = layout_type[0] if len(layout_type) > 0 else 'unknown'
            
            # Create position dictionaries
            true_pos = {i: true_coords[i] for i in range(data.num_nodes)}
            pred_pos = {i: pred_coords[i] for i in range(data.num_nodes)}
            
            # Draw ground truth
            ax_true = axes[row, 0]
            nx.draw(G, pos=true_pos, ax=ax_true,
                   node_color=node_colors, edge_color='gray', alpha=0.8,
                   node_size=80, width=0.5, with_labels=False)
            ax_true.set_title(f"{graph_id}\nGround Truth ({layout_type}) {color_info}")
            ax_true.set_aspect('equal')
            ax_true.axis('off')
            
            # Draw prediction
            ax_pred = axes[row, 1]
            nx.draw(G, pos=pred_pos, ax=ax_pred,
                   node_color=node_colors, edge_color='gray', alpha=0.8,
                   node_size=80, width=0.5, with_labels=False)
            ax_pred.set_title(f"{graph_id}\n{model_name} Prediction {color_info}")
            ax_pred.set_aspect('equal')
            ax_pred.axis('off')
            
            print(f"  Processed {graph_id}: {data.num_nodes} nodes, {G.number_of_edges()} edges")
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename - handle layout_type properly
        sample_layout_type = getattr(graphs[0], 'layout_type', 'layout')
        if isinstance(sample_layout_type, (list, tuple)):
            sample_layout_type = sample_layout_type[0] if len(sample_layout_type) > 0 else 'layout'
        
        filename = f"{model_name}_{sample_layout_type}_{split}_{num_samples}samples.png"
        save_path = os.path.join(output_dir, filename)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\\nSaved visualization to: {save_path}")
        return save_path

def main():
    parser = argparse.ArgumentParser(description='Visualize graph layout predictions')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of graphs to visualize')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], 
                       default='test', help='Dataset split to use')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'auto'], 
                       default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Check files exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return 1
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        return 1
    
    # Run visualization
    visualizer = GraphVisualizer(device=device)
    
    try:
        save_path = visualizer.visualize(
            model_path=args.model_path,
            data_path=args.data_path,
            num_samples=args.num_samples,
            split=args.split,
            output_dir=args.output_dir,
            seed=args.seed
        )
        
        print(f"\\n✅ Visualization completed!")
        print(f"📁 Output: {save_path}")
        
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())