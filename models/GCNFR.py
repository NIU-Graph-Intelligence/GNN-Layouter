import torch
from torch import nn
from models.mlp_layers import MLPFactory
from models.coordinate_layers import ForceDirectedProcessor

class NodeModel(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        
        self.in_feat = in_feat
        self.out_feat = out_feat

        # Initialize force processor
        self.force_processor = ForceDirectedProcessor(in_feat)

        # Create MLPs using factory
        self.message_mlp = MLPFactory.create('force_directed', in_feat=in_feat, out_feat=in_feat, is_message_mlp=True)
        
        self.output_mlp = MLPFactory.create('force_directed', in_feat=in_feat, out_feat=out_feat, is_message_mlp=False)


    def forward(self, x, edge_index, batch, init_coords=None):
        
        x_input = torch.cat([x, init_coords], dim=1) if init_coords is not None else x
        row, col = edge_index
        
        # Process messages
        neighbor_msg = self.message_mlp(x_input[row])
        # Process forces and get aggregated neighbors
        fx, agg_neighbors = self.force_processor.process_force_messages(
            x_input, neighbor_msg, col, x_input.size(0)
        )
        
        # Final output processing
        out_final = torch.cat([fx, agg_neighbors], dim=1)
        
        return self.output_mlp(out_final)

class ForceGNN(nn.Module):
    def __init__(self, in_feat, hidden_dim, out_feat, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(NodeModel(in_feat, hidden_dim))
        
        for i in range(num_layers - 2):
            self.layers.append(NodeModel(hidden_dim, hidden_dim))
        
        self.layers.append(NodeModel(hidden_dim, out_feat))

    def forward(self, x, edge_index, batch, init_coords=None):
        h = self.layers[0](x, edge_index, batch, init_coords)
        
        for layer_idx in range(1, len(self.layers) - 1):
            h_new = self.layers[layer_idx](h, edge_index, batch)
            h = h_new + h
        
        coords = self.layers[-1](h, edge_index, batch)
        
        # Use ForceDirectedProcessor for coordinate reshaping
        coords = ForceDirectedProcessor.reshape_coordinates(coords, batch)
        
        return coords





