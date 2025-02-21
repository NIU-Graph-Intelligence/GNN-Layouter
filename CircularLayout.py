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
        model_path = '/checkpoints'
        plt.tight_layout()
        # Save in the visualization directory with full path
        save_path = os.path.join(model_path, f'layout_epoch_{epoch}.png')
        plt.savefig(save_path)
        plt.close()

        # Print confirmation
        print(f"Saved visualization to {save_path}")




def main(batch_size=1):


    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    models = ImprovedLayoutGNN(input_dim=53, hidden_channels=64)
    trained_model = train_model(models, train_loader, val_loader)

    return trained_model

if __name__ == '__main__':
    model = main()