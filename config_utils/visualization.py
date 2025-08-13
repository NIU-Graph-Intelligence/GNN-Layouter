#!/usr/bin/env python3
"""
Unified Visualization Module for GNN Layouts
Supports both Circular and Force-Directed layout visualizations
"""

import os
import sys
import argparse
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.gridspec as gridspec
from torch_geometric.data import Data

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# Model imports
from models.GCNFR import ForceGNN
from models.ChebConv import GNN_ChebConv
from data.dataset import circular_data_loader, force_directed_data_loader
from config_utils.config_manager import get_config


class UnifiedVisualizer:
    """Unified visualization class for both circular and force-directed layouts"""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = get_config()
        
        # Load visualization configuration
        viz_config = self.config.get_visualization_config()
        self.default_figsize = viz_config.get('figsize', [10, 5])
        self.default_node_size = viz_config.get('node_size', 80)
        self.default_edge_width = viz_config.get('edge_width', 0.5)
        self.default_dpi = viz_config.get('dpi', 300)
        
    def generate_distinct_colors(self, n):
        """Generate n visually distinct colors using HSV color space"""
        colors = []
        for i in range(n):
            hue = i / n
            saturation = 0.7 + 0.3 * (i % 2)
            value = 0.7 + 0.3 * ((i + 1) % 2)
            colors.append(plt.cm.hsv(hue))
        return colors
    
    def get_community_colors(self, community_labels):
        """Get consistent colors for community labels"""
        if community_labels is None:
            return None
            
        # Convert to integers
        comm_list = [int(n) for n in community_labels]
        
        # Define consistent color scheme
        distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Get unique communities
        uniq_comms = sorted(set(comm_list))
        
        # Extend colors if needed
        if len(uniq_comms) > len(distinct_colors):
            extra_colors = plt.cm.Set3(np.linspace(0, 1, len(uniq_comms) - len(distinct_colors)))
            distinct_colors.extend(['#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)) 
                                  for r, g, b, _ in extra_colors])
        
        # Map communities to colors
        comm_to_idx = {c: i for i, c in enumerate(uniq_comms)}
        node_colors = [distinct_colors[comm_to_idx[c]] for c in comm_list]
        
        return node_colors
    
    def nx_to_pyg_data(self, G, expected_feature_size):
        """Convert NetworkX graph to PyTorch Geometric Data object"""
        num_nodes = G.number_of_nodes()
        adj_matrix = nx.to_numpy_array(G)
        edge_index = np.array(np.nonzero(adj_matrix))
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        # Create one-hot encoding
        one_hot = torch.eye(num_nodes, dtype=torch.float32)
        
        # Create positional encoding (normalize to [0,1] range)
        pos_encoding = torch.arange(num_nodes, dtype=torch.float32) / num_nodes
        pos_encoding = pos_encoding.unsqueeze(1)
        
        # Concatenate one-hot and positional encoding
        x = torch.cat([one_hot, pos_encoding], dim=1)

        # Adjust feature size
        if x.shape[1] < expected_feature_size:
            padding = torch.zeros((num_nodes, expected_feature_size - x.shape[1]))
            x = torch.cat([x, padding], dim=1)
        elif x.shape[1] > expected_feature_size:
            x = x[:, :expected_feature_size]

        data = Data(x=x, edge_index=edge_index)
        return data
    
    def draw_graph_layout(self, G, coords, ax, title, node_colors=None, edge_color='gray', 
                         node_size=80, edge_width=0.5, show_labels=False):
        """Draw a graph layout with given coordinates"""
        pos = {i: tuple(coords[i]) for i in range(len(coords))}
        
        if node_colors is None:
            node_colors = self.generate_distinct_colors(len(coords))
        
        nx.draw(G, pos=pos, ax=ax,
                node_color=node_colors, edge_color=edge_color, alpha=0.8,
                node_size=node_size, width=edge_width, with_labels=show_labels)
        
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def visualize_circular_layout(self, model_path=None, data_path=None, num_samples=None, 
                                output_dir=None, seed=None, split='test'):
        """Visualize circular layout model predictions using actual test dataset"""
        
        # Get defaults from config if not provided
        if num_samples is None:
            num_samples = self.config.get('visualization.num_samples', 5)
        if output_dir is None:
            output_dir = self.config.get_figures_path()
        if seed is None:
            seed = self.config.get('data.random_state', 42)
        if model_path is None:
            model_path = self.config.get_model_path('circular')
        if data_path is None:
            data_path = self.config.get_data_path('circular')
            
        print(f"Visualizing circular layout with {num_samples} samples from {split} set")
        
        # Setup
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load processed dataset
        print(f"Loading data from {data_path}")
        data_dict = torch.load(data_path, map_location='cpu')
        full_dataset = data_dict['dataset']
        
        print(f"Loaded {len(full_dataset)} graphs")
        
        # Get data loaders to split the dataset
        train_loader, val_loader, test_loader = circular_data_loader(
            dataset=full_dataset,
            batch_size=1,  # Use batch size 1 for individual graph analysis
            splits=(0.8, 0.10, 0.10),
            random_state=seed
        )
        
        # Select the appropriate split
        if split == 'test':
            loader = test_loader
        elif split == 'val':
            loader = val_loader
        else:
            loader = train_loader
        
        # Extract graphs from the loader
        graphs = []
        for batch in loader:
            graphs.append(batch)
        
        print(f"Using {len(graphs)} graphs from {split} set")
        
        # Sample graphs
        num_samples = min(num_samples, len(graphs))
        indices = np.random.choice(len(graphs), num_samples, replace=False)
        
        # Load model
        sample_graph = graphs[0]
        feature_dim = sample_graph.x.shape[1]
        
        model = GNN_ChebConv(input_dim=feature_dim)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()

        # Setup figure with config defaults
        base_width, base_height = self.default_figsize
        fig = plt.figure(figsize=(base_width, base_height*len(indices)))
        gs = gridspec.GridSpec(len(indices), 2)
        
        
        for row, idx in enumerate(indices):
            data = graphs[idx].to(self.device)
            
            print(f"\nProcessing graph {idx}: nodes={data.num_nodes}, edges={data.edge_index.size(1)}")
            
            # Get ground truth coordinates
            true_coords = data.y.cpu().numpy()
            
            # Get model prediction
            with torch.no_grad():
                pred_coords = model(data.x, data.edge_index).cpu().numpy()

            # Create NetworkX graph for visualization
            G = nx.Graph()
            G.add_nodes_from(range(data.num_nodes))
            edge_list = data.edge_index.cpu().numpy().T.tolist()
            G.add_edges_from(edge_list)
            
            # Generate distinct colors for nodes
            node_colors = self.generate_distinct_colors(data.num_nodes)

            # Create subplots
            ax_true = fig.add_subplot(gs[row, 0])
            ax_pred = fig.add_subplot(gs[row, 1])

            # Draw layouts
            self.draw_graph_layout(G, true_coords, ax_true, 
                                 f"Graph {idx} Ground Truth\n({data.num_nodes} nodes)", 
                                 node_colors)
            self.draw_graph_layout(G, pred_coords, ax_pred, 
                                 f"Graph {idx} Prediction\n({data.num_nodes} nodes)", 
                                 node_colors)
            

        # Save visualization
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'Circular_layout_{split}_{num_samples}samples.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Saved circular layout visualization to {save_path}")
        return save_path
    
    def visualize_force_directed_layout(self, model_path=None, data_path=None, num_samples=None, 
                                      output_dir=None, seed=None, split='test'):
        """Visualize force-directed layout model predictions using actual dataset split"""
        
        # Get defaults from config if not provided
        if num_samples is None:
            num_samples = self.config.get('visualization.num_samples', 5)
        if output_dir is None:
            output_dir = self.config.get_figures_path()
        if seed is None:
            seed = self.config.get('data.random_state', 42)
        if model_path is None:
            model_path = self.config.get_model_path('force_forcegnn')
        if data_path is None:
            data_path = self.config.get_data_path('force_directed')
            
        print(f"Visualizing force-directed layout with {num_samples} samples from {split} set")
        
        # Setup
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Load processed dataset
        print(f"Loading data from {data_path}")
        data_dict = torch.load(data_path, map_location='cpu')
        full_dataset = data_dict['dataset']
        
        print(f"Loaded {len(full_dataset)} graphs")
        
        # Get data loaders to split the dataset
        train_loader, val_loader, test_loader = force_directed_data_loader(
            dataset=full_dataset,
            batch_size=1,  # Use batch size 1 for individual graph analysis
            splits=(0.8, 0.10, 0.10),
            random_state=seed
        )
        
        # Select the appropriate split
        if split == 'test':
            loader = test_loader
        elif split == 'val':
            loader = val_loader
        else:
            loader = train_loader
        
        # Extract graphs from the loader
        graphs = []
        for batch in loader:
            graphs.append(batch)
        
        print(f"Using {len(graphs)} graphs from {split} set")
        
        # Load ForceGNN model with configuration
        sample_graph = graphs[0]
        feature_dim = sample_graph.x.shape[1] + sample_graph.init_coords.shape[1]
        
        # Get model configuration
        model_config = self.config.get_model_config('ForceGNN')
        
        model = ForceGNN(
            in_feat=feature_dim,
            hidden_dim=model_config.get('hidden_dim', 32),
            out_feat=model_config.get('out_feat', 2),
            num_layers=model_config.get('num_layers', 4),
        ).to(self.device)
        
        # Load checkpoint
        print(f"Loading weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Sample graphs
        num_samples = min(num_samples, len(graphs))
        
        # Prefer graphs with community information
        valid_indices = [i for i in range(len(graphs)) if hasattr(graphs[i], 'community')]
        if len(valid_indices) >= num_samples:
            indices = np.random.choice(valid_indices, num_samples, replace=False)
            print(f"Selected {num_samples} graphs with community information")
        else:
            indices = np.random.choice(len(graphs), num_samples, replace=False)
            print(f"Selected {num_samples} graphs (some may not have community info)")
        
        # Setup figure with config defaults
        base_width, base_height = self.default_figsize
        fig = plt.figure(figsize=(base_width, base_height*len(indices)))
        gs = GridSpec(len(indices), 2, figure=fig)
        
        
        for row, idx in enumerate(indices):
            print(f"\nProcessing graph {idx}: {graphs[idx].graph_id}")
            
            data = graphs[idx].to(self.device)
            
            # Create batch tensor
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=self.device)
            
            # Get ground truth coordinates
            true_coords = data.y.cpu().numpy()
            
            # Get model prediction
            with torch.no_grad():
                pred_coords = model(data.x, data.edge_index, batch, data.init_coords)
                pred_coords = pred_coords.detach().cpu().numpy()
            
            # Create NetworkX graph
            G = nx.Graph()
            G.add_nodes_from(range(data.num_nodes))
            edge_list = data.edge_index.cpu().numpy().T.tolist()
            G.add_edges_from(edge_list)
            
            # Get node colors from community information
            node_colors = None
            if hasattr(data, 'community'):
                comm_labels = data.community.cpu().tolist()
                node_colors = self.get_community_colors(comm_labels)
            
            # Plot ground truth
            ax_gt = fig.add_subplot(gs[row, 0])
            self.draw_graph_layout(G, true_coords, ax_gt, f"{data.graph_id}\nGround Truth", node_colors)
            
            # Plot prediction
            ax_pred = fig.add_subplot(gs[row, 1])
            self.draw_graph_layout(G, pred_coords, ax_pred, f"{data.graph_id}\nForceGNN Prediction", node_colors)
            
        
        # Save visualization
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"ForceDirected_{split}_{num_samples}samples.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.default_dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Saved force-directed visualization to {save_path}")
        
        return save_path


def main():
    config = get_config()
    viz_config = config.get_visualization_config()
    
    parser = argparse.ArgumentParser(description="Unified GNN Layout Visualizer")
    parser.add_argument('--layout', type=str, choices=['circular', 'force'], 
                       required=True, help="Layout type to visualize")
    
    # Circular layout arguments with config defaults
    parser.add_argument('--circular-model', type=str, 
                       default=config.get_model_path('circular'),
                       help="Path to circular layout model weights")
    parser.add_argument('--circular-data', type=str, 
                       default=config.get_data_path('circular'),
                       help="Processed circular layout dataset path")
    
    # Force-directed layout arguments with config defaults
    parser.add_argument('--force-model', type=str, 
                       default=config.get_model_path('force_forcegnn'),
                       help="Path to ForceGNN model weights")
    parser.add_argument('--force-data', type=str, 
                       default=config.get_data_path('force_directed'),
                       help="Processed force-directed dataset path")
    
    # Common arguments with config defaults
    parser.add_argument('--samples', type=int, 
                       default=viz_config.get('num_samples', 5),
                       help="Number of graphs to visualize")
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], 
                       default='test', help="Dataset split to use for visualization")
    parser.add_argument('--output-dir', type=str, 
                       default=viz_config.get('output_dir', config.get_figures_path()),
                       help="Output directory for visualizations")
    parser.add_argument('--seed', type=int, 
                       default=config.get('data.random_state', 42),
                       help="Random seed")
    parser.add_argument('--device', type=str, default=None,
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Initialize visualizer
    visualizer = UnifiedVisualizer(device=device)
    
    # Run visualization
    if args.layout == 'circular':
        if os.path.exists(args.circular_model) and os.path.exists(args.circular_data):
            print("\n" + "="*50)
            print("CIRCULAR LAYOUT VISUALIZATION")
            print("="*50)
            save_path = visualizer.visualize_circular_layout(
                model_path=args.circular_model,
                data_path=args.circular_data,
                num_samples=args.samples,
                output_dir=args.output_dir,
                seed=args.seed,
                split=args.split
            )
            print(f"✅ Circular layout visualization complete!")
            print(f"   Saved to: {save_path}")
        else:
            print(f"❌ Circular layout files not found:")
            print(f"  Model: {args.circular_model} (exists: {os.path.exists(args.circular_model)})")
            print(f"  Data: {args.circular_data} (exists: {os.path.exists(args.circular_data)})")
    
    elif args.layout == 'force':
        if os.path.exists(args.force_model) and os.path.exists(args.force_data):
            print("\n" + "="*50)
            print("FORCE-DIRECTED LAYOUT VISUALIZATION")
            print("="*50)
            save_path = visualizer.visualize_force_directed_layout(
                model_path=args.force_model,
                data_path=args.force_data,
                num_samples=args.samples,
                output_dir=args.output_dir,
                seed=args.seed,
                split=args.split
            )
            print(f"✅ Force-directed layout visualization complete!")
            print(f"   Saved to: {save_path}")
        else:
            print(f"❌ Force-directed files not found:")
            print(f"  Model: {args.force_model} (exists: {os.path.exists(args.force_model)})")
            print(f"  Data: {args.force_data} (exists: {os.path.exists(args.force_data)})")
    
    print("\n🎉 Visualization complete!")


if __name__ == "__main__":
    main()
