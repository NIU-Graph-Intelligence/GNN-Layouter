"""
Community graph generation utilities using LFR benchmark algorithm.
"""

import os
import subprocess
import shutil
import networkx as nx
import numpy as np
from typing import List, Optional, Union, Dict, Any
from tqdm import tqdm
from pathlib import Path


def parse_node_range(num_nodes_config: Union[int, str, Dict[str, int]]) -> tuple:
    """Parse node count configuration into (min_nodes, max_nodes) tuple."""
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
    """Parse parameter range into (min_val, max_val) tuple."""
    if isinstance(param, (int, float)):
        return float(param), float(param)
    elif isinstance(param, str) and "-" in param:
        min_val, max_val = map(float, param.split("-"))
        return min_val, max_val
    else:
        return float(param), float(param)


def find_lfr_executable() -> str:
    """Find LFR benchmark executable."""
    for exe_name in ["lfrbench_udwov", "benchmark", "lfr_benchmark"]:
        if shutil.which(exe_name):
            return exe_name
        local_exe = os.path.join(os.getcwd(), exe_name)
        if os.path.exists(local_exe) and os.access(local_exe, os.X_OK):
            return local_exe
    raise FileNotFoundError("LFR executable not found")


def generate_single_lfr_graph(graph_id: str, 
                             nodes: int, 
                             params: Dict[str, Any],
                             work_dir: Path) -> Optional[nx.Graph]:
    """Generate a single LFR graph."""
    
    try:
        lfr_executable = find_lfr_executable()
    except FileNotFoundError:
        return None
    
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
    
    # Adjust constraints
    max_degree = min(max_degree, nodes - 1)
    max_community_size = min(max_community_size, nodes - 1)
    
    # Build LFR command
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
    
    try:
        # Run LFR in shared work directory (seed auto-managed)
        result = subprocess.run(cmd, cwd=work_dir, capture_output=True, timeout=timeout)
        
        # Check for output files
        network_file = work_dir / "network.dat"
        community_file = work_dir / "community.dat"
        
        if not (network_file.exists() and community_file.exists()):
            return None
        
        # Parse network file
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
                        weight = float(parts[2]) if len(parts) > 2 else 1.0
                        G.add_edge(u, v, weight=weight)
                    except ValueError:
                        continue
        
        # Parse community file
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
        
        # Validate graph
        if not G.nodes or not nx.is_connected(G):
            return None
        
        # Remap nodes to 0-based indexing
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(G.nodes()))}
        G_remapped = nx.Graph()
        
        for u, v, data in G.edges(data=True):
            G_remapped.add_edge(node_mapping[u], node_mapping[v], **data)
        
        for old_id, comm_id in communities.items():
            if old_id in node_mapping:
                G_remapped.nodes[node_mapping[old_id]]["community"] = comm_id
        
        # Read current seed for tracking
        seed_file = work_dir / "seed.txt"
        current_seed = None
        if seed_file.exists():
            try:
                with open(seed_file, 'r') as f:
                    current_seed = int(f.read().strip())
            except:
                pass
        
        # Set graph metadata
        G_remapped.graph.update({
            'id': graph_id,
            'type': 'Community',
            'num_communities': len(set(communities.values())),
            'has_communities': True,
            'seed': current_seed,
            'params': params.copy()
        })
        
        return G_remapped
        
    except Exception:
        return None


def generate_community_graphs_variable(num_graphs: int,
                                     num_nodes_config: Union[int, str, Dict],
                                     **params) -> List[nx.Graph]:
    """Generate multiple LFR community graphs with shared seed management."""
    
    min_nodes, max_nodes = parse_node_range(num_nodes_config)
    
    # Setup random number generator for parameter sampling
    seed = params.get('seed', 42)
    rng = np.random.RandomState(seed)
    
    # Parse parameter ranges
    param_ranges = {}
    for param_name in ['avg_degree', 'max_degree', 'mixing_parameter', 
                      'min_community_size', 'max_community_size']:
        if param_name in params:
            param_ranges[param_name] = parse_range_parameter(params[param_name])
    
    # Create shared work directory for LFR seed management
    work_dir = Path("temp_lfr_shared")
    work_dir.mkdir(exist_ok=True)
    
    # Initialize seed file
    seed_file = work_dir / "seed.txt"
    if not seed_file.exists():
        with open(seed_file, 'w') as f:
            f.write(str(seed))
    
    # Generation parameters
    max_attempts = params.get('max_attempts', num_graphs * 3)
    verbose = params.get('verbose', False)
    
    graphs = []
    attempts = 0
    failed_attempts = 0
    
    pbar = tqdm(total=num_graphs, desc="Generating community graphs")
    
    try:
        while len(graphs) < num_graphs and attempts < max_attempts:
            attempts += 1
            
            # Sample parameters for this graph
            nodes_for_graph = rng.randint(min_nodes, max_nodes + 1)
            
            graph_params = params.copy()
            for param_name, (min_val, max_val) in param_ranges.items():
                if min_val != max_val:
                    graph_params[param_name] = rng.uniform(min_val, max_val)
                else:
                    graph_params[param_name] = min_val
            
            if verbose:
                print(f"\nAttempt {attempts}: {nodes_for_graph} nodes")
            
            # Generate graph using shared work directory
            G = generate_single_lfr_graph(
                f"Community_{len(graphs):04d}",
                nodes_for_graph,
                graph_params,
                work_dir
            )
            
            if G is not None:
                graphs.append(G)
                pbar.update(1)
                if verbose:
                    print(f"  ✅ Success! Graph {len(graphs)}/{num_graphs}")
            else:
                failed_attempts += 1
                if verbose:
                    print(f"  ❌ Failed (total failures: {failed_attempts})")
        
        pbar.close()
        
        # Print generation summary
        if graphs:
            node_counts = [G.number_of_nodes() for G in graphs]
            edge_counts = [G.number_of_edges() for G in graphs]
            comm_counts = [G.graph['num_communities'] for G in graphs]
            seeds_used = [G.graph.get('seed') for G in graphs if G.graph.get('seed')]
            
            print(f"\n✅ Generated {len(graphs)} community graphs")
            print(f"   Success rate: {len(graphs)/attempts*100:.1f}% ({attempts} attempts)")
            print(f"   Nodes: {min(node_counts)}-{max(node_counts)} (avg {np.mean(node_counts):.1f})")
            print(f"   Edges: {min(edge_counts)}-{max(edge_counts)} (avg {np.mean(edge_counts):.1f})")
            print(f"   Communities: {min(comm_counts)}-{max(comm_counts)} (avg {np.mean(comm_counts):.1f})")
            
            if seeds_used:
                unique_seeds = len(set(seeds_used))
                print(f"   Seeds: {min(seeds_used)}-{max(seeds_used)} (unique: {unique_seeds}/{len(seeds_used)})")
                if unique_seeds < len(seeds_used):
                    print("   ⚠️  Warning: Some graphs may have used duplicate seeds!")
        
        if len(graphs) < num_graphs:
            print(f"⚠️  Warning: Only generated {len(graphs)}/{num_graphs} graphs")
            print(f"   Failed attempts: {failed_attempts}")
    
    finally:
        # Cleanup shared directory unless requested to keep
        if not params.get('keep_temp_files', False):
            shutil.rmtree(work_dir, ignore_errors=True)
    
    return graphs