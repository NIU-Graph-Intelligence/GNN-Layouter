import torch

# processed_path = "data/processed/splits/test_FR.pt"
# data = torch.load(processed_path)
# print(data)

# ─── CONFIGURE THIS ────────────────────────────────────────────────────────────
processed_path = "data/processed/modelInput_FRWithoutInputOneHot1024.pt"
# ────────────────────────────────────────────────────────────────────────────────

# Load the processed dataset
data_dict = torch.load(processed_path)
dataset   = data_dict["dataset"]
print(len(dataset))
layout_ty = data_dict.get("layout_type", "Unknown")
max_nodes = data_dict.get("max_nodes", None)
# print(dataset)
print(f"\n✅ Loaded processed dataset: {processed_path}")
print(f"• Layout type:       {layout_ty}")
print(f"• Number of graphs:  {len(dataset)}")
print(f"• Max nodes per graph: {max_nodes}\n")

# Inspect only the first graph
data = dataset[0]

print("--- First graph (index 0) ---")
print(f"Graph ID:           {data.graph_id}")
print(f"Num nodes:          {data.num_nodes}")
print(f"Num communities:    {int(data.num_communities)}\n")
print(f"Num Communities:  {data.num_communities}")

# Edge information
print(f"Edge index shape:   {tuple(data.edge_index.shape)}")
print("First 10 edges:")
print(data.edge_index.t()[:5].tolist(), "\n")

print(f"Edge attr shape:    {tuple(data.edge_attr.shape)}")
print("First 10 edge weights:")
print(data.edge_attr[:10,0].tolist(), "\n")

# Node features
print(f"Node feature shape: {tuple(data.x.shape)}")
print("First 5 feature vectors (rows):")
for i, row in enumerate(data.x[:10]):
    print(f"  Node {i:2d}:", row.tolist())
print()

# Stored layouts
print(f"Normalized y shape: {tuple(data.y.shape)}")
print("First 5 normalized coords:")
print(data.y[:5].tolist(), "\n")

if hasattr(data, "original_y"):
    print(f"Original_y shape:   {tuple(data.original_y.shape)}")
    print("First 5 original coords:")
    print(data.original_y[:5].tolist(), "\n")
else:
    print("No attribute 'original_y' found\n")


# import pickle
# # ─── CONFIGURE THIS ────────────────────────────────────────────────────────────
# processed_path = "data/raw/adjacency_matrices/Finaladjacency_matrix_1024.pkl"
# # ────────────────────────────────────────────────────────────────────────────────
# with open(processed_path, "rb") as f:
#     data = pickle.load(f)
#     print(data)

