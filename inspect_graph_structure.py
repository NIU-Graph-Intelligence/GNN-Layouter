import networkx as nx
import random

# Create a small example graph like we do in generate_graphs.py
num_nodes = 10  # Smaller for easier inspection

# Create one of each type of graph
graphs = []

# BA Graph
m = random.randint(2, 5)
G_ba = nx.barabasi_albert_graph(n=num_nodes, m=m)
graphs.append(('BA', G_ba))

# ER Graph
p = random.uniform(0.1, 0.3)
G_er = nx.erdos_renyi_graph(num_nodes, p)
graphs.append(('ER', G_er))

# WS Graph
k = random.randint(4, 8)
p = random.uniform(0.1, 0.3)
G_ws = nx.watts_strogatz_graph(n=num_nodes, k=k, p=p)
graphs.append(('WS', G_ws))

# Inspect each graph
for graph_type, G in graphs:
    print(f"\n{'-'*50}")
    print(f"Graph Type: {graph_type}")
    print(f"{'-'*50}")
    
    # Basic information
    print("\nBasic Information:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Is directed: {G.is_directed()}")
    print(f"Is weighted: {any('weight' in d for u,v,d in G.edges(data=True))}")
    
    # Node information
    print("\nNode Information:")
    print(f"Nodes: {list(G.nodes())}")
    print(f"Node attributes: {G.nodes(data=True)}")
    
    # Edge information
    print("\nEdge Information:")
    print(f"First 5 edges: {list(G.edges())[:5]}")
    print(f"Edge attributes: {list(G.edges(data=True))[:5]}")
    
    # Graph properties
    print("\nGraph Properties:")
    print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    print(f"Is connected: {nx.is_connected(G)}")
    
    # Node degrees
    print("\nNode Degrees (first 5 nodes):")
    degrees = dict(G.degree())
    for node in list(G.nodes())[:5]:
        print(f"Node {node}: {degrees[node]} connections") 