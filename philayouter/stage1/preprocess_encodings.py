"""
preprocess_encodings.py

One-time preprocessing script. Run this ONCE before Stage 1 training.

Loads the original community graph dataset, computes all GST encodings
for every graph, stores them as fields in the Data objects, and saves
an enriched dataset to disk.

After this runs, training never recomputes eigenvectors or RRWP again.

Fields added to each Data object:
    data.lap_pe      [N, lap_k]   Laplacian eigenvectors (raw, SignNet handles signs)
    data.eigvals     [1, lap_k]   Laplacian eigenvalues  (graph-level, stored as [1,k]
                                  so PyG batches to [B, lap_k])
    data.rrwp_node   [N, rw_k]   RRWP diagonal entries  (local SE)
    data.rrwp_edge   [E, rw_k]   RRWP edge entries      (relative PE)
    data.fiedler_se  [E, 1]      Fiedler relative SE    (community signal)

Why all shapes are batchable:
    [N, k]  → PyG concatenates along dim 0 → [sum_N, k]  ✓ (like data.x)
    [E, k]  → PyG concatenates along dim 0 → [sum_E, k]  ✓ (like edge_attr)
    [1, k]  → PyG concatenates along dim 0 → [B, k]      ✓ (graph-level)

The old batching failure was [N, N, k] (full pairwise matrix). None of our
encodings have that shape — we only store diagonal and edge entries.

Usage:
    python -m philayouter.stage1.preprocess_encodings \
        --input_path  data/processed/community_5k_Nodes20-50_force_directed.pt \
        --output_path data/processed/community_5k_with_encodings.pt \
        --lap_k 10 \
        --rw_k  16
"""

import os
import sys
import argparse
import time
import warnings
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from philayouter.stage1.gst.encodings import compute_all_encodings


def preprocess(
    input_path: str,
    output_path: str,
    lap_k: int = 10,
    rw_k: int = 16,
):
    # ----------------------------------------------------------------
    # Load original dataset
    # ----------------------------------------------------------------
    print(f"Loading: {input_path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data_list = torch.load(input_path, weights_only=False)

    print(f"  Graphs: {len(data_list)}")
    print(f"  Sample fields: {list(data_list[0].keys())}")
    print(f"  Node features: {data_list[0].x.shape[1]}")
    print()

    # ----------------------------------------------------------------
    # Compute and attach encodings
    # ----------------------------------------------------------------
    print(f"Computing encodings (lap_k={lap_k}, rw_k={rw_k})...")
    t0 = time.time()
    failed = 0
    for idx, data in enumerate(data_list):
        if idx % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx:5d}/{len(data_list)}]  {elapsed:.1f}s elapsed")

        edge_index = data.edge_index
        N = data.x.shape[0]

        # Compute on CPU (eigsh is CPU-only via scipy)
        # DataLoader will move tensors to GPU during training
        try:
            enc = compute_all_encodings(
                edge_index=edge_index.cpu(),
                num_nodes=N,
                lap_k=lap_k,
                rw_k=rw_k,
            )

            # Attach to Data object
            # eigvecs [N, lap_k] — batches like x
            data.lap_pe = enc['eigvecs'].float()

            # eigvals [1, lap_k] — stored as 2D so PyG batches to [B, lap_k]
            data.eigvals = enc['eigvals'].float().unsqueeze(0)

            # rrwp_node [N, rw_k] — batches like x
            data.rrwp_node = enc['rrwp_node'].float()

            # rrwp_edge [E, rw_k] — batches like edge_attr
            data.rrwp_edge = enc['rrwp_edge'].float()

            # fiedler_se [E, 1] — batches like edge_attr
            data.fiedler_se = enc['fiedler_se'].float()

        except Exception as e:
            print(f"  WARNING: graph {idx} failed ({e}), padding with zeros")
            E = edge_index.shape[1]
            data.lap_pe     = torch.zeros(N, lap_k)
            data.eigvals    = torch.zeros(1, lap_k)
            data.rrwp_node  = torch.zeros(N, rw_k)
            data.rrwp_edge  = torch.zeros(E, rw_k)
            data.fiedler_se = torch.zeros(E, 1)
            failed += 1

    elapsed = time.time() - t0
    print(f"\nDone. {elapsed:.1f}s total ({elapsed/len(data_list)*1000:.1f}ms per graph)")
    if failed:
        print(f"WARNING: {failed} graphs failed and were zero-padded")

    # ----------------------------------------------------------------
    # Verify shapes on first graph
    # ----------------------------------------------------------------
    d = data_list[0]
    print(f"\nVerification (graph 0, N={d.x.shape[0]}, E={d.edge_index.shape[1]}):")
    print(f"  lap_pe:     {d.lap_pe.shape}    — [N, {lap_k}]")
    print(f"  eigvals:    {d.eigvals.shape}   — [1, {lap_k}]")
    print(f"  rrwp_node:  {d.rrwp_node.shape}  — [N, {rw_k}]")
    print(f"  rrwp_edge:  {d.rrwp_edge.shape}  — [E, {rw_k}]")
    print(f"  fiedler_se: {d.fiedler_se.shape} — [E, 1]")

    # ----------------------------------------------------------------
    # Quick batching test — confirm PyG handles all shapes correctly
    # ----------------------------------------------------------------
    print(f"\nBatching test (first 4 graphs)...")
    from torch_geometric.data import Batch
    batch = Batch.from_data_list(data_list[:4])
    print(f"  x:          {batch.x.shape}")
    print(f"  lap_pe:     {batch.lap_pe.shape}")
    print(f"  eigvals:    {batch.eigvals.shape}")
    print(f"  rrwp_node:  {batch.rrwp_node.shape}")
    print(f"  rrwp_edge:  {batch.rrwp_edge.shape}")
    print(f"  fiedler_se: {batch.fiedler_se.shape}")
    print(f"  batch vec:  {batch.batch.shape}")
    print("  ✓ All shapes batch correctly")

    # ----------------------------------------------------------------
    # Save enriched dataset
    # ----------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    print(f"\nSaving to: {output_path}")
    torch.save(data_list, output_path)

    # File size
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  File size: {size_mb:.1f} MB")
    print(f"\nDone. Use {output_path} as --dataset_path for python -m philayouter.stage1.train")


def main():
    parser = argparse.ArgumentParser(
        description='Precompute GST encodings and attach to dataset'
    )
    parser.add_argument(
        '--input_path', type=str, required=True,
        help='Path to original .pt dataset file'
    )
    parser.add_argument(
        '--output_path', type=str, required=True,
        help='Path to save enriched .pt dataset file'
    )
    parser.add_argument('--lap_k', type=int, default=10,
                        help='Number of Laplacian eigenvectors (default 10)')
    parser.add_argument('--rw_k',  type=int, default=16,
                        help='Number of RRWP steps (default 16)')
    args = parser.parse_args()

    preprocess(
        input_path=args.input_path,
        output_path=args.output_path,
        lap_k=args.lap_k,
        rw_k=args.rw_k,
    )


if __name__ == '__main__':
    main()
