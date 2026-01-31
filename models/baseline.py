import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, ChebConv

# ------------------------------
# GCN baseline
# ------------------------------
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, dropout=0.1,
                 use_residual=False, use_input_mlp=False):
        super().__init__()
        self.dropout = dropout
        self.use_residual = use_residual

        if use_input_mlp:
            self.input_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            conv_input_dim = hidden_dim
        else:
            self.input_mlp = None
            conv_input_dim = input_dim

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = conv_input_dim if i == 0 else hidden_dim
            self.convs.append(GCNConv(in_dim, hidden_dim))

        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x, edge_index):
        h = x.float()
        if self.input_mlp:
            h = self.input_mlp(h)

        for conv in self.convs:
            h_new = conv(h, edge_index)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new if self.use_residual else h_new

        return self.mlp_head(h)


# ------------------------------
# GAT baseline
# ------------------------------
class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, heads=1, dropout=0.1,
                 use_residual=False, use_input_mlp=False):
        super().__init__()
        self.dropout = dropout
        self.use_residual = use_residual

        if use_input_mlp:
            self.input_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            conv_input_dim = hidden_dim
        else:
            self.input_mlp = None
            conv_input_dim = input_dim

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = conv_input_dim if i == 0 else hidden_dim
            self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, concat=False, dropout=dropout))

        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x, edge_index):
        h = x.float()
        if self.input_mlp:
            h = self.input_mlp(h)

        for conv in self.convs:
            h_new = conv(h, edge_index)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new if self.use_residual else h_new

        return self.mlp_head(h)


# ------------------------------
# GIN baseline
# ------------------------------
class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, dropout=0.1,
                 use_residual=False, use_input_mlp=False):
        super().__init__()
        self.dropout = dropout
        self.use_residual = use_residual

        if use_input_mlp:
            self.input_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            conv_input_dim = hidden_dim
        else:
            self.input_mlp = None
            conv_input_dim = input_dim

        self.convs = nn.ModuleList()
        # first layer
        mlp = nn.Sequential(
            nn.Linear(conv_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(mlp, train_eps=True))
        # remaining layers
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))

        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x, edge_index):
        h = x.float()
        if self.input_mlp:
            h = self.input_mlp(h)

        for conv in self.convs:
            h_new = conv(h, edge_index)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new if self.use_residual else h_new

        return self.mlp_head(h)


# ------------------------------
# ChebNet baseline
# ------------------------------
class ChebNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, K=3, dropout=0.1,
                 use_residual=False, use_input_mlp=False):
        super().__init__()
        self.dropout = dropout
        self.use_residual = use_residual

        if use_input_mlp:
            self.input_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            conv_input_dim = hidden_dim
        else:
            self.input_mlp = None
            conv_input_dim = input_dim

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = conv_input_dim if i == 0 else hidden_dim
            self.convs.append(ChebConv(in_dim, hidden_dim, K=K))

        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x, edge_index):
        h = x.float()
        if self.input_mlp:
            h = self.input_mlp(h)

        for conv in self.convs:
            h_new = conv(h, edge_index)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new if self.use_residual else h_new

        return self.mlp_head(h)
