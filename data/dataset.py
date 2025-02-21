from sklearn.model_selection import train_test_split
import os
import torch
from torch_geometric.data import Data, DataLoader

def data_loader(dataset):
    # file_path = os.path.join(os.path.dirname(__file__), "processed", "modelInput.pt")
    # dataset = torch.load(file_path)

    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_data, shuffle=True)
    val_loader = DataLoader(val_data)

    return train_loader, val_loader
