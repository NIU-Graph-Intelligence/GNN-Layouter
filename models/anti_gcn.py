class AntiSmoothingSpringGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        
        # 1. 传统GCN处理连边吸引力
        self.gcn_layers = nn.ModuleList([
            GCNConv(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(3)
        ])
        
        # 2. 反GCN处理排斥力
        self.anti_gcn_layers = nn.ModuleList([
            AntiGCNConv(hidden_dim) for _ in range(2)
        ])
        
        # 3. 全局注意力处理远程排斥
        self.global_attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
        # 4. 最终坐标预测
        self.coord_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, x, edge_index):
        # GCN分支：局部吸引力
        h_gcn = x
        for gcn in self.gcn_layers:
            h_gcn = F.relu(gcn(h_gcn, edge_index))
        
        # 反GCN分支：局部排斥力
        h_anti = h_gcn
        for anti_gcn in self.anti_gcn_layers:
            h_anti = F.relu(anti_gcn(h_anti, edge_index))
        
        # 全局注意力：远程排斥力
        h_global, _ = self.global_attention(h_gcn, h_gcn, h_gcn)
        
        # 组合三种力
        h_combined = torch.cat([h_gcn, h_anti, h_global], dim=-1)
        coordinates = self.coord_predictor(h_combined)
        
        return coordinates