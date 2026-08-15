"""
data/processed/generate_fa2_final_dataset.py

Despite the filename (written for FA2 first, reused as-is for KK -- the
--iters_file/--output args are the only thing that changes per teacher, none
of the logic below is FA2-specific), this builds the counterpart of
comm_5k_v2_with_encodings.pt for any teacher whose iteration pickle matches
generate_fr_iterations.py's schema: same 5000 graphs, same node ordering,
same train/val/test split (both come from the same seed-42 shuffle over the
same graph list) -- only the trajectory teacher differs.

Reuses generate_final_dataset.py's build_graph_sample() unmodified for the
x/edge_index/y/y_traj/y_mean/y_std/k/community fields (same normalization
convention as the FR dataset -- see the scale-normalization convention's addendum
on why glide/executor/data.py has to undo it before training). Structural
encodings (lap_pe, eigvals, rrwp_node, rrwp_edge, fiedler_se) are NOT
recomputed: they depend only on graph topology, which is identical to the
FR dataset's, so they're copied across by graph_idx instead of paying for a
second eigendecomposition of the same 5000 graphs.

Usage:
    python data/processed/generate_fa2_final_dataset.py \
        --graphs_file data/graphs/comm_5k_5000graphs_20-50nodes_20260322_163135.pkl \
        --iters_file  data/layouts/comm_5k_fa2_iters.pkl \
        --encodings_source data/processed/comm_5k_v2_with_encodings.pt \
        --output      data/processed/comm_5k_fa2_with_encodings.pt
"""

import argparse
import os
import pickle
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # data/generate_final_dataset.py
from generate_final_dataset import build_graph_sample, load_graphs, load_iterations  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_file", required=True)
    parser.add_argument("--iters_file", required=True)
    parser.add_argument("--encodings_source", required=True,
                        help="Already-built FR dataset to copy structural encodings from")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_graphs", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading graphs from {args.graphs_file} ...")
    graphs = load_graphs(args.graphs_file)

    print(f"Loading FA2 iteration records from {args.iters_file} ...")
    iter_records = load_iterations(args.iters_file)["graphs"]

    assert len(graphs) == len(iter_records), (
        f"Mismatch: {len(graphs)} graphs vs {len(iter_records)} records"
    )

    print(f"Loading structural encodings from {args.encodings_source} ...")
    fr_dataset = torch.load(args.encodings_source, weights_only=False)
    assert len(fr_dataset) == len(graphs), (
        f"Encoding source has {len(fr_dataset)} graphs, expected {len(graphs)} -- "
        f"graph_idx-based copy requires identical graph lists/ordering"
    )

    if args.max_graphs is not None:
        graphs = graphs[: args.max_graphs]
        iter_records = iter_records[: args.max_graphs]
        fr_dataset = fr_dataset[: args.max_graphs]

    dataset = []
    for graph_idx, (G, record, fr_sample) in enumerate(
        tqdm(zip(graphs, iter_records, fr_dataset), total=len(graphs), desc="Building dataset")
    ):
        try:
            sample = build_graph_sample(graph_idx, G, record)
            # Structural encodings are topology-only -- identical between the
            # FR and FA2 datasets for the same graph_idx. Verified below via
            # the num_nodes match rather than assumed.
            assert sample.num_nodes == fr_sample.num_nodes, (
                f"graph_idx {graph_idx}: node count mismatch "
                f"({sample.num_nodes} vs {fr_sample.num_nodes}) -- graph lists are out of sync"
            )
            sample.lap_pe = fr_sample.lap_pe
            sample.eigvals = fr_sample.eigvals
            sample.rrwp_node = fr_sample.rrwp_node
            sample.rrwp_edge = fr_sample.rrwp_edge
            sample.fiedler_se = fr_sample.fiedler_se
            dataset.append(sample)
        except Exception as e:
            print(f"\n[WARN] Skipped graph {graph_idx}: {e}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(dataset, args.output)

    node_counts = [d.num_nodes for d in dataset]
    print(f"\nDataset saved -> {args.output}")
    print(f"  Total graphs : {len(dataset)}")
    print(f"  Node range   : {min(node_counts)} - {max(node_counts)}")
    print(f"  x shape      : {dataset[0].x.shape}")
    print(f"  y_traj shape : {dataset[0].y_traj.shape}")
    print(f"  lap_pe shape : {dataset[0].lap_pe.shape}  (copied from FR dataset)")


if __name__ == "__main__":
    main()
