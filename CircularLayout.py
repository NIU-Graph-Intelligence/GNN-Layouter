import networkx as nx
import random
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GCN, GCNConv
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import math
import pickle

merged_graph_dataset = {}

# Create a permanent directory for model saves
save_dir = os.path.expanduser("~/PythonProject/models")
os.makedirs(save_dir, exist_ok=True)
print(f"Created model directory at: {save_dir}")


def visualize_layout(model, sample, epoch, device):
    model.eval()
    with torch.no_grad():
        sample = sample.to(device)
        pred_coords = model(sample.x, sample.edge_index)

        # Convert to numpy for plotting
        pred_coords = pred_coords.cpu().numpy()
        true_coords = sample.y.cpu().numpy()
        edges = sample.edge_index.cpu().numpy().T

        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Plot predicted layout
        ax1.scatter(pred_coords[:, 0], pred_coords[:, 1])
        for edge in edges:
            ax1.plot(pred_coords[edge, 0], pred_coords[edge, 1], 'gray', alpha=0.5)
        ax1.set_title(f'Predicted Layout - Epoch {epoch}')

        # Plot true layout
        ax2.scatter(true_coords[:, 0], true_coords[:, 1])
        for edge in edges:
            ax2.plot(true_coords[edge, 0], true_coords[edge, 1], 'gray', alpha=0.5)
        ax2.set_title('True Layout')

        plt.tight_layout()
        # Save in the visualization directory with full path
        save_path = os.path.join(save_dir, f'layout_epoch_{epoch}.png')
        plt.savefig(save_path)
        plt.close()

        # Print confirmation
        print(f"Saved visualization to {save_path}")

def load_graph_dataset(load_dir="data/graph_dataset"):
    file_path = os.path.join(load_dir, "merged_graph_dataset.pkl")
    with open(file_path, "rb") as f:
        loaded_dataset = pickle.load(f)
    print(f"Graph dataset loaded from {load_dir}")
    return loaded_dataset

def load_adjacency_matrices(load_dir="data/adjacency_matrices"):
    adjacency_matrices = {}
    for file in os.listdir(load_dir):
        graph_type = file.split(".")[0]
        file_path = os.path.join(load_dir, file)
        adjacency_matrices[graph_type] = np.load(file_path, allow_pickle=True)
    print(f"Adjacency matrices loaded from {load_dir}")
    return adjacency_matrices

def load_layouts(load_dir="data/layouts"):
    layouts = {}
    for file in os.listdir(load_dir):
        graph_type = file.split(".")[0]
        file_path = os.path.join(load_dir, file)
        with open(file_path, "rb") as f:
            layouts[graph_type] = pickle.load(f)
    print(f"Layouts loaded from {load_dir}")
    return layouts

def process_dictionary_data(adj_dict, coord_dict):
    processed_adj_matrices = []
    processed_coordinates = []

    for struct_type in adj_dict.keys():
        adj_matrices = adj_dict[struct_type]
        layouts = coord_dict[struct_type]

        for adj_matrix, layout in zip(adj_matrices, layouts):
            num_nodes = adj_matrix.shape[0]
            coords = np.zeros((num_nodes, 2))

            for node_idx, coord in layout.items():
                coords[node_idx] = coord

            processed_adj_matrices.append(np.array(adj_matrix))
            processed_coordinates.append(coords)

    return processed_adj_matrices, processed_coordinates

def create_node_features(num_nodes):
    "Create node features including normalized indices and positional encoding"
    indices = torch.arange(num_nodes, dtype=torch.float32)
    normalized_indices = 2 * math.pi * indices / num_nodes # Convert to angles

    # Create Features: [normalized_index, sin, cos]
    features = torch.zeros((num_nodes, 3))
    features[:, 0] = indices / num_nodes # Normalized index
    features[:, 1] = torch.sin(normalized_indices) # Sin component
    features[:, 2] = torch.cos(normalized_indices) # Cos component
    return features

def prepare_data(adj_matrices, coordinates):
    "Modified data preparation with both one-hot encoding and enhanced node features"

    dataset = []
    for adj_mat, coords in zip(adj_matrices, coordinates):
        # Convert adjacency matrix to edge indices
        edge_index = torch.from_numpy(np.array(adj_mat.nonzero())).long()

        # Get number of nodes
        num_nodes = adj_mat.shape[0]

        # Create one-hot encoding
        one_hot = torch.eye(50)

        # Create circular position features
        pos_features = create_node_features(num_nodes)

        # Concatenate features [50+3 = 53 features]
        x = torch.cat([one_hot, pos_features], dim=1) # Combined features

        # Normalize coordinates
        coords = np.array(coords)
        coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-5)

        # Convert coordinates to tensor
        y = torch.from_numpy(coords).float()

        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, y=y)
        dataset.append(data)
    return dataset

class ImprovedLayoutGNN(nn.Module):
    def __init__(self, input_dim=53, hidden_channels=128): # 50 (one-hot) + 3 ( positional features )
        super().__init__()

        # GNN layers
        self.conv1 = GCNConv(input_dim, hidden_channels) # 53 -> 64
        self.conv2 = GCNConv(hidden_channels, hidden_channels) # 64 -> 64
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)


        self.pos_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels // 2, 2)
        )

        self.radius_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, 1),
            nn.Softplus()
        )

    def forward(self, x, edge_index):
        # Initial GNN layers with residual connections
        h1 = self.conv1(x, edge_index)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=0.3, training=self.training)

        h2 = self.conv2(h1, edge_index)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=0.3, training=self.training)
        h2 = h2 + h1  # Residual connection

        h3 = self.conv3(h2, edge_index)
        h3 = F.relu(h3)
        h3 = F.dropout(h3, p=0.3, training=self.training)
        h3 = h3 + h2  # Residual connection

        h4 = self.conv4(h3, edge_index)
        h4 = F.relu(h4)
        h4 = F.dropout(h4, p=0.3, training=self.training)
        h4 = h4 + h3

        # Predict radius and coordinates separately
        radius = self.radius_mlp(h4)
        coords = self.pos_mlp(h4)

        # Normalize coordinates to unit circle
        coords_norm = F.normalize(coords, p=2, dim=1)

        # Scale by predicted radius
        coords = coords_norm * radius

        return coords

def normalized_coordinates(coords):
    # Center the coordinates
    coords = coords - coords.mean(dim=0, keepdim=True)
    # scale to [-1,1] range
    max_dist = torch.max(torch.norm(coords, dim=1))
    return coords / ( max_dist + 1e-5)

def circular_layout_loss(pred_coords, true_coords, x):
    """
    Enhanced loss function with better circular geometry constraints
    """
    # L2 coordinate loss
    coord_loss = F.mse_loss(pred_coords, true_coords)

    # Normalize predicted and true coordinates
    pred_norm = F.normalize(pred_coords, p=2, dim=1)
    true_norm = F.normalize(true_coords, p=2, dim=1)

    # Circular geometry loss
    circular_loss = F.mse_loss(pred_norm, true_norm)

    # Radius consistency
    pred_radii = torch.norm(pred_coords, dim=1)
    true_radii = torch.norm(true_coords, dim=1)
    radius_loss = F.mse_loss(pred_radii, true_radii)

    # Angular preservation loss
    pred_angles = torch.atan2(pred_coords[:, 1], pred_coords[:, 0])
    true_angles = torch.atan2(true_coords[:, 1], true_coords[:, 0])
    angle_loss = F.mse_loss(torch.sin(pred_angles), torch.sin(true_angles)) + \
                 F.mse_loss(torch.cos(pred_angles), torch.cos(true_angles))

    # Total loss with weighted components
    total_loss = coord_loss + \
                 0.3 * circular_loss + \
                 0.1 * radius_loss + \
                 0.1 * angle_loss

    return total_loss

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Get predictions
        pred_coords = model(batch.x, batch.edge_index)

        # Calculate pairwise Euclidean distance loss
        loss = circular_layout_loss(pred_coords, batch.y, batch.x)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_nodes

    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_coords = model(batch.x, batch.edge_index)
            loss = circular_layout_loss(pred_coords, batch.y, batch.x)
            total_loss += loss.item() * batch.num_nodes

    return total_loss / len(loader.dataset)

def train_model(model, train_loader, val_loader, num_epochs=2000, lr=0.0005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.03)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # Changed scheduler
        optimizer,
        mode='min',
        factor=0.5,
        patience=20,
        min_lr=1e-5,
        verbose=True
    )
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 300

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # Train on individual samples
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred_coords = model(batch.x, batch.edge_index)
            loss = circular_layout_loss(pred_coords, batch.y, batch.x)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device)

        # Pass val_loss to scheduler.step()
        scheduler.step(val_loss)

        if epoch % 20 == 0 or epoch == num_epochs - 1:
            visualize_layout(model, next(iter(train_loader)), epoch, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            try:
                save_path = os.path.join(save_dir, 'best_model.pt')
                torch.save(model.state_dict(), save_path)
            except Exception as e:
                print(f"Error saving model: {str(e)}")
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(
                f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

    return model

def main(batch_size=1):

    merged_graph_dataset = load_graph_dataset()
    adjacency_matrices = load_adjacency_matrices()
    circular_layouts = load_layouts()

    adj_matrices, coordinates = process_dictionary_data(adjacency_matrices, circular_layouts)
    dataset = prepare_data(adj_matrices, coordinates)

    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    models = ImprovedLayoutGNN(input_dim=53, hidden_channels=64)
    trained_model = train_model(models, train_loader, val_loader)

    return trained_model

if __name__ == '__main__':
    model = main()