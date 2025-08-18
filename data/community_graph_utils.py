import os
import subprocess
import random
import shutil
import networkx as nx
import numpy as np
from typing import List, Optional, Union, Dict, Any
from tqdm import tqdm
from pathlib import Path

def parse_node_range(num_nodes_config: Union[int, str, Dict[str, int]]) -> tuple:
    if isinstance(num_nodes_config, int):
        return num_nodes_config, num_nodes_config
    elif isinstance(num_nodes_config, str):
        if "-" in num_nodes_config:
            min_val, max_val = map(int, num_nodes_config.split("-"))
            return min_val, max_val
        else:
            val = int(num_nodes_config)
            return val, val
    elif isinstance(num_nodes_config, dict):
        return num_nodes_config["min"], num_nodes_config["max"]
    else:
        raise ValueError(f"Invalid num_nodes format: {num_nodes_config}")

def parse_range_parameter(param: Union[int, float, str]) -> tuple:
    """Parse a parameter that can be a single value or a range string like '1-10'"""
    if isinstance(param, (int, float)):
        return float(param), float(param)
    elif isinstance(param, str) and "-" in param:
        min_val, max_val = map(float, param.split("-"))
        return min_val, max_val
    else:
        return float(param), float(param)

def sample_parameter(min_val: float, max_val: float, seed: int = None) -> float:
    if seed is not None:
        random.seed(seed)
    return random.uniform(min_val, max_val) if min_val != max_val else min_val

def find_lfr_executable() -> str:
    """Find LFR benchmark executable"""
    for exe_name in ["lfrbench_udwov", "benchmark", "lfr_benchmark"]:
        if shutil.which(exe_name):
            return exe_name
        local_exe = os.path.join(os.getcwd(), exe_name)
        if os.path.exists(local_exe) and os.access(local_exe, os.X_OK):
            return local_exe
    raise FileNotFoundError("LFR executable not found")

def generate_single_lfr_graph(graph_id: str, nodes: int, params: Dict[str, Any],
                             temp_root: str = "temp_lfr") -> Optional[nx.Graph]:
    """Generate a single LFR graph"""
    
    # Extract parameters with defaults
    avg_degree = params.get('avg_degree', 10)
    max_degree = params.get('max_degree', 50)
    mixing_parameter = params.get('mixing_parameter', 0.1)
    weight_mixing = params.get('weight_mixing', 0.0)
    degree_exponent = params.get('degree_exponent', 2.0)
    community_exponent = params.get('community_exponent', 1.0)
    weight_exponent = params.get('weight_exponent', 1.5)
    min_community_size = params.get('min_community_size', 10)
    max_community_size = params.get('max_community_size', 50)
    timeout = params.get('timeout', 30)

    # Adjust constraints based on node count
    max_degree = min(max_degree, nodes - 1)
    max_community_size = min(max_community_size, nodes - 1)

    try:
        lfr_executable = find_lfr_executable()
    except FileNotFoundError:
        print(f"❌ LFR executable not found")
        return None

    # Setup temp directory
    temp_dir = Path(temp_root) / graph_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        lfr_executable,
        "-N", str(nodes),
        "-k", str(int(avg_degree)),
        "-maxk", str(max_degree),
        "-mut", str(mixing_parameter),
        "-muw", str(weight_mixing),
        "-t1", str(degree_exponent),
        "-t2", str(community_exponent),
        "-beta", str(weight_exponent),
        "-minc", str(min_community_size),
        "-maxc", str(max_community_size),
    ]

    # Run LFR
    try:
        result = subprocess.run(cmd, cwd=temp_dir, capture_output=True, timeout=timeout)
        
        # Check for output files (LFR may return error code but still generate files)
        network_file = temp_dir / "network.dat"
        community_file = temp_dir / "community.dat"
        
        if not (network_file.exists() and community_file.exists()):
            return None

        # Read network file
        G = nx.Graph()
        with open(network_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        u, v = int(parts[0]), int(parts[1])
                        G.add_edge(u, v)
                    except ValueError:
                        continue

        # Read community file
        communities = {}
        with open(community_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        node, comm = int(parts[0]), int(parts[1])
                        communities[node] = comm
                    except ValueError:
                        continue

        # Validate and remap nodes
        if not G.nodes or not nx.is_connected(G):
            return None

        node_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(G.nodes()))}
        G_remapped = nx.Graph()
        
        for u, v in G.edges():
            G_remapped.add_edge(node_mapping[u], node_mapping[v])
        
        for old_id, comm_id in communities.items():
            if old_id in node_mapping:
                G_remapped.nodes[node_mapping[old_id]]["community"] = comm_id

        # Set metadata
        G_remapped.graph.update({
            'id': graph_id,
            'type': 'Community',
            'num_communities': len(set(communities.values())),
            'has_communities': True,
            'params': params.copy()
        })

        return G_remapped

    except Exception:
        return None
    finally:
        # Clean up
        if not params.get('keep_failed_attempts', False):
            shutil.rmtree(temp_dir, ignore_errors=True)

def generate_community_graphs_variable(num_graphs: int,
                                     num_nodes_config: Union[int, str, Dict],
                                     **params) -> List[nx.Graph]:
    """Generate community graphs with variable parameters - continues until target number is reached"""
    
    min_nodes, max_nodes = parse_node_range(num_nodes_config)
    seed = params.get('seed', 42)
    max_attempts = params.get('max_attempts', num_graphs * 5)  # 最大尝试次数，防止无限循环
    
    # Set seed
    random.seed(seed)
    np.random.seed(seed)

    # Parse parameter ranges
    param_ranges = {}
    for param_name in ['avg_degree', 'max_degree', 'mixing_parameter', 'min_community_size', 'max_community_size']:
        if param_name in params:
            param_ranges[param_name] = parse_range_parameter(params[param_name])

    graphs = []
    attempts = 0
    failed_attempts = 0
    
    # 使用进度条显示成功生成的图数量
    pbar = tqdm(total=num_graphs, desc="Generating graphs")
    
    while len(graphs) < num_graphs and attempts < max_attempts:
        attempts += 1
        
        # Sample parameters for this graph
        nodes_for_graph = random.randint(min_nodes, max_nodes)
        
        graph_params = params.copy()
        for param_name, (min_val, max_val) in param_ranges.items():
            if min_val != max_val:
                graph_params[param_name] = sample_parameter(min_val, max_val, seed + attempts + hash(param_name))
        
        # Generate graph
        G = generate_single_lfr_graph(f"Community_{len(graphs):04d}", nodes_for_graph, graph_params)
        
        if G is not None:
            graphs.append(G)
            pbar.update(1)  # 只有成功生成时才更新进度条
        else:
            failed_attempts += 1
            # 可选：显示失败信息
            if params.get('verbose', False):
                pbar.set_description(f"Generating graphs (failed: {failed_attempts})")
    
    pbar.close()

    # 检查是否达到目标
    if len(graphs) < num_graphs:
        print(f"⚠️  Warning: Only generated {len(graphs)}/{num_graphs} graphs after {attempts} attempts")
        print(f"   Failed attempts: {failed_attempts}")
        if attempts >= max_attempts:
            print(f"   Reached maximum attempts limit ({max_attempts})")
    
    # Print summary
    if graphs:
        node_counts = [G.number_of_nodes() for G in graphs]
        edge_counts = [G.number_of_edges() for G in graphs]
        comm_counts = [G.graph['num_communities'] for G in graphs]
        
        print(f"\n✅ Successfully generated {len(graphs)} graphs")
        print(f"   Total attempts: {attempts} (success rate: {len(graphs)/attempts*100:.1f}%)")
        print(f"   Nodes: {min(node_counts)}-{max(node_counts)} (avg {np.mean(node_counts):.1f})")
        print(f"   Edges: {min(edge_counts)}-{max(edge_counts)} (avg {np.mean(edge_counts):.1f})")
        print(f"   Communities: {min(comm_counts)}-{max(comm_counts)} (avg {np.mean(comm_counts):.1f})")

    return graphs