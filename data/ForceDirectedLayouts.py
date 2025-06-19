import os
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import from_networkx, to_networkx
import time
import argparse
import sys

# allow imports from your project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(PROJECT_ROOT)


class CommunityGraphLayoutGenerator:
    def __init__(self, dataset_path="data/raw/graph_dataset/community_graphs_dataset1024.pkl"):

        self.dataset_path = dataset_path
        self.dataset = None

    def load_dataset(self):

        print(f"Loading dataset from {self.dataset_path}...")
        with open(self.dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)
        print(f"Loaded {len(self.dataset)} graphs")
        return self.dataset



    def get_adjacency_matrix(self, graph_idx):
        if self.dataset is None:
            self.load_dataset()

        data = self.dataset[graph_idx]
        G = to_networkx(data, to_undirected=True)

        edge_weights = data.edge_attr.squeeze().tolist()
        for i, (u, v) in enumerate(data.edge_index.t().tolist()):
            G[u][v]['weight'] = edge_weights[i]

        nodes = list(G.nodes())
        node_mapping = {node: idx for idx, node in enumerate(nodes)}

        n = len(nodes)
        adjacency_matrix = np.zeros((n, n), dtype=float)

        for u, v, d in G.edges(data=True):
            i, j = node_mapping[u], node_mapping[v]
            weight = d.get('weight', 1.0)
            adjacency_matrix[i, j] = weight
            adjacency_matrix[j, i] = weight

        return adjacency_matrix, node_mapping, G


   
    def assign_initial_positions(self, G, dim=2, eps=1e-3):
        nodes = list(G.nodes())
        n = len(nodes)
        grid_size = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / grid_size))

        pos = {}
        for idx, node in enumerate(nodes):
            u = (idx % grid_size) / max(grid_size - 1, 1)  # in [0,1]
            v = (idx // grid_size) / max(rows - 1, 1)
            # remap to [-1,1]
            x = 2 * u - 1
            y = 2 * v - 1
            pos[node] = np.array([x, y], dtype=float)

        return pos

    def fruchterman_reingold_layout(self, graph_idx, iterations=50, scale=1.0):

        if self.dataset is None:
            self.load_dataset()

        data = self.dataset[graph_idx]

        G = to_networkx(data, to_undirected=True)
        edge_weights = data.edge_attr.squeeze().tolist()
        for i, (u,v) in enumerate(data.edge_index.t().tolist()):
            G[u][v]['weight'] = edge_weights[i]

        initial_pos = self.assign_initial_positions(G)

        pos = nx.spring_layout(G, iterations=iterations, scale=scale, seed=42, pos=initial_pos, weight='weight')
        # print(pos)
        return pos, initial_pos


    def forceatlas2_layout(self, graph_idx, iterations=100, scale=1.0):

        if self.dataset is None:
            self.load_dataset()

        graph_data = self.dataset[graph_idx]

        G = to_networkx(graph_data, to_undirected=True)


        initial_pos = self.assign_initial_positions(G)

        positions = nx.forceatlas2_layout(
            G,
            pos=initial_pos,
            max_iter=iterations,
            scaling_ratio=scale,
            gravity=1.0,
            # dissuade_hubs=True,
            jitter_tolerance=1.0,
            seed=42,
            strong_gravity=1.0,

        )

        arr1 = np.vstack([positions[n] for n in G.nodes()])

        return positions


    # Each unique id nodes visible
    def visualize_layout(self,
                         positions,
                         G,
                         communities=None,
                         title="Graph Layout",
                         save_path=None,
                         label_nodes=False):
        """
        Visualize a graph layout with community colors and optional node-ID labels.
        """
        plt.figure(figsize=(10, 10))

        # Prepare node colors based on communities
        node_colors = None

        if communities:
            # Build a stable mapping from community label -> color index
            unique_communities = sorted(set(communities.values()))
            comm_to_idx = {c: i for i, c in enumerate(unique_communities)}
            cmap = plt.cm.get_cmap('tab10', len(unique_communities))

            # Assign each node the color corresponding to its community index
            node_colors = []
            for node in G.nodes():
                idx = comm_to_idx[communities[node]]
                node_colors.append(cmap(idx))

        # Decide node_size and whether to show labels
        node_size = 300 if label_nodes else 50
        with_labels = label_nodes
        labels = {n: str(n) for n in G.nodes()} if label_nodes else None
        font_size = 8 if label_nodes else None

        # Draw the graph in one go
        nx.draw_networkx(
            G,
            pos=positions,
            node_color=node_colors,
            with_labels=with_labels,
            labels=labels,
            font_size=font_size,
            node_size=node_size,
            edge_color='gray',
            alpha=0.8,
            width=0.5
        )

        plt.title(title)
        plt.axis('off')


        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def convert_to_torch_geometric(self, graph_idx):
        if self.dataset is None:
            self.load_dataset()

        graph_data = self.dataset[graph_idx]
        # G = graph_data["graph"]
        G = to_networkx(graph_data, to_undirected=True)
        communities = graph_data["communities"]

        # Add community as a node attribute
        for node, comm in communities.items():
            G.nodes[node]['community'] = comm

        # Convert to PyTorch Geometric Data object
        data = from_networkx(G)

        # Add community as a node feature if it doesn't exist
        if 'community' not in data:
            # Get unique communities and create mapping
            unique_communities = sorted(set(communities.values()))
            community_mapping = {comm: idx for idx, comm in enumerate(unique_communities)}

            # Create one-hot encoded features
            num_nodes = data.num_nodes
            num_communities = len(unique_communities)
            community_features = torch.zeros((num_nodes, num_communities), dtype=torch.float)

            # Map original node IDs to consecutive indices
            mapping = {old_id: new_id for new_id, old_id in enumerate(G.nodes())}

            for node, comm in communities.items():
                node_idx = mapping[node]
                comm_idx = community_mapping[comm]
                community_features[node_idx, comm_idx] = 1.0

            data.community = community_features


        return data

    def convert_layout_dict_to_tensor(self, layout_dict, adjacency_matrices):

        tensor_dict = {}
        for graph_id, pos_dict in layout_dict.items():
            mapping = adjacency_matrices[graph_id]["node_mapping"]
            # Build coords[i] = pos_dict[orig_node_id] for the exact adjacency order
            N = len(mapping)
            coords = torch.zeros((N, 2), dtype=torch.float32)
            for orig_id, idx in mapping.items():
                coords[idx] = torch.tensor(pos_dict[orig_id], dtype=torch.float32)
            tensor_dict[graph_id] = coords
        return tensor_dict


    def process_all_graphs(self, visualize_sample=False, generate_adj=False, generate_fr=False, generate_fa2=False):

        if self.dataset is None:
            self.load_dataset()

        num_graphs = len(self.dataset)
        print(f"Processing {num_graphs} graphs...")

        # Initialize dictionaries only for requested components
        adjacency_matrices = {} if generate_adj else None
        fr_layouts = {} if generate_fr else None
        ft2_layouts = {} if generate_fa2 else None
        initial_layouts = {}

        for idx in range(num_graphs):
            data = self.dataset[idx]
            # Get graph_id from the Data object, fallback to index if not present
            graph_id = getattr(data, 'graph_id', idx)
            
            # Get adjacency matrix if needed
            G = None
            if generate_adj or generate_fr or generate_fa2:  # Need graph structure for any layout
                adj_matrix, node_map, G = self.get_adjacency_matrix(idx)
                if generate_adj:
                    adjacency_matrices[graph_id] = {
                        "matrix": adj_matrix,
                        "node_mapping": node_map
                    }

            # Generate initial positions if needed for either FR or FA2
            init_pos = None
            if generate_fr or generate_fa2:
                init_pos = self.assign_initial_positions(G)
                initial_layouts[graph_id] = init_pos

            # Generate Fruchterman-Reingold layout if requested
            if generate_fr:
                fr_pos, _ = self.fruchterman_reingold_layout(idx)  # We already have init_pos
                fr_layouts[graph_id] = fr_pos

            # Generate ForceAtlas2 layout if requested
            if generate_fa2:
                ft2_pos = self.forceatlas2_layout(idx)
                ft2_layouts[graph_id] = ft2_pos

            # Visualize sample if requested
            if visualize_sample and idx < 5 and G is not None:
                communities = {i: int(c.item()) for i, c in enumerate(data.community)}

                if generate_fr:
                    self.visualize_layout(
                        fr_pos, G, communities,
                        title=f"Fruchterman-Reingold Layout (Graph {graph_id})",
                        save_path=f"fr_layout_{graph_id}.png",
                        label_nodes=True
                    )

                if generate_fa2:
                    self.visualize_layout(
                        ft2_pos, G, communities,
                        title=f"ForceAtlas2 Layout (Graph {graph_id})",
                        save_path=f"ft2_layout_{graph_id}.png",
                        label_nodes=True
                    )

        # Save results based on what was generated
        if generate_adj:
            print("Saving adjacency matrices...")
            with open("data/raw/adjacency_matrices/Finaladjacency_matrix_1024.pkl", 'wb') as f:
                pickle.dump(adjacency_matrices, f)

        # Save initial layouts if either FR or FA2 was generated
        if generate_fr or generate_fa2:
            print("Saving initial positions...")
            initial_layouts_tensor = self.convert_layout_dict_to_tensor(initial_layouts, adjacency_matrices)
            torch.save(initial_layouts_tensor, "data/raw/layouts/FinalInit_1024.pt")

        if generate_fr:
            print("Saving Fruchterman-Reingold layouts...")
            fr_layouts_tensor = self.convert_layout_dict_to_tensor(fr_layouts, adjacency_matrices)
            torch.save(fr_layouts_tensor, "data/raw/layouts/FinalFR_1024.pt")

        if generate_fa2:
            print("Saving ForceAtlas2 layouts...")
            ft2_layouts_tensor = self.convert_layout_dict_to_tensor(ft2_layouts, adjacency_matrices)
            torch.save(ft2_layouts_tensor, "data/raw/layouts/FinalFT2_1024.pt")

        return adjacency_matrices, fr_layouts, ft2_layouts

def main():
    parser = argparse.ArgumentParser(description='Generate force-directed layouts for community graphs')
    
    # Add command line arguments
    parser.add_argument('--dataset-path', type=str, 
                       default="data/raw/graph_dataset/community_graphs_dataset1024.pkl",
                       help='Path to the input dataset')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize sample layouts')
    parser.add_argument('--generate-adj', action='store_true',
                       help='Generate adjacency matrices')
    parser.add_argument('--generate-fr', action='store_true',
                       help='Generate Fruchterman-Reingold layouts')
    parser.add_argument('--generate-fa2', action='store_true',
                       help='Generate ForceAtlas2 layouts')
    
    args = parser.parse_args()
    
    # Initialize layout generator
    layout_gen = CommunityGraphLayoutGenerator(dataset_path=args.dataset_path)
    
    # Process graphs with selected options
    adjacency_matrices, fr_layouts, ft2_layouts = layout_gen.process_all_graphs(
        visualize_sample=args.visualize,
        generate_adj=args.generate_adj,
        generate_fr=args.generate_fr,
        generate_fa2=args.generate_fa2
    )
    
    # Print summary
    print("\nProcessing completed!")
    if args.generate_adj:
        print(f"Generated adjacency matrices for {len(adjacency_matrices)} graphs")
    if args.generate_fr:
        print(f"Generated FR layouts for {len(fr_layouts)} graphs")
    if args.generate_fa2:
        print(f"Generated FA2 layouts for {len(ft2_layouts)} graphs")

if __name__ == "__main__":
    main()
