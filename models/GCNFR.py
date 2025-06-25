import torch
from torch import nn
from torch_scatter import scatter_mean

class NodeModel(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        # print(f"\nInitializing NodeModel with in_feat={in_feat}, out_feat={out_feat}")

        self.norm_before_rep = nn.LayerNorm(self.in_feat)
        self.norm_after_rep = nn.LayerNorm(self.in_feat)

        # Simplified MLP blocks; adjust as needed.
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(in_feat, in_feat, bias=False),
            nn.ReLU(),
            nn.Linear(in_feat, in_feat, bias=False),
        )

        self.node_mlp_2 = nn.Sequential(
            nn.Linear(2 * in_feat, in_feat, bias=False),
            nn.ReLU(),
            nn.Linear(in_feat, out_feat, bias=False)
        )

        # One learnable weight per feature‐dimension
        self.w = nn.Parameter(torch.ones(in_feat))


    def forward(self, x, edge_index, batch, init_coords=None):

        # Validate graph independence
        
        # Handle initial coordinates if provided
        if init_coords is not None:
            x_input = torch.cat([x, init_coords], dim=1)
            # x_input = x
            # print(x_input)

        else:
            x_input = x
        
        # Get source and target nodes for each edge
        row, col = edge_index
        
        # Get messages from source nodes
        neighbor_msg = x_input[row]  # [num_edges, in_feat]
        
        # Transform messages
        neighbor_msg = self.node_mlp_1(neighbor_msg)  # [num_edges, in_feat]
        
        # Aggregate messages for each target node - scatter_mean automatically respects graph boundaries
        # because edge_index only contains within-graph edges
        agg_neighbors = scatter_mean(
            neighbor_msg,
            col,
            dim=0,
            dim_size=x_input.size(0)
        )  # [total_nodes, in_feat]
        
        # Apply normalization and repulsion
        agg_neighbors = self.norm_before_rep(agg_neighbors)
        repulsion = (x_input - agg_neighbors) * self.w
        
        # Combine features
        fx = x_input + repulsion
        fx = self.norm_after_rep(fx)
        
        # Final transformation
        out_final = torch.cat([fx, agg_neighbors], dim=1)
        final_output = self.node_mlp_2(out_final)  # [total_nodes, out_feat]
        
        return final_output

class ForceGNN(nn.Module):
        def __init__(self, in_feat, hidden_dim, out_feat, num_layers):
            super().__init__()

            self.layers = nn.ModuleList()

            # First layer: input feature size
            self.layers.append(NodeModel(in_feat, hidden_dim))
            
            # Hidden layers
            for i in range(num_layers - 2):
                # print(f"\nAdding hidden layer {i+1}")
                self.layers.append(NodeModel(hidden_dim, hidden_dim))

            # Last layer: output dimension
            self.layers.append(NodeModel(hidden_dim, out_feat))

        def forward(self, x, edge_index, batch, init_coords=None):

            # Layer 0: Process through first layer with initial coordinates
            h = self.layers[0](x, edge_index, batch, init_coords)
            
            # Hidden layers with skip connections
            for layer_idx in range(1, len(self.layers) - 1):
                h_new = self.layers[layer_idx](h, edge_index, batch)
                h = h_new + h
            
            # Final layer
            coords = self.layers[-1](h, edge_index, batch)
            
            # Get number of graphs and nodes per graph for reshaping
            num_graphs = batch.max().item() + 1
            nodes_per_graph = x.shape[0] // num_graphs

            # # Reshape output to [batch_size, nodes_per_graph, 2]
            coords = coords.view(num_graphs, nodes_per_graph, -1)
            
            # print(f"Final output shape: {coords.shape}")
            return coords





