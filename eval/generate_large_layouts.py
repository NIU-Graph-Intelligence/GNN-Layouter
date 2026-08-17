"""
eval/generate_large_layouts.py

R1 (rider brief 2026-08-16): produce the three layout sources that R1.2 scores
on the same synthetic corpora at N = 10^5 and 10^6:

  phi      -- the Phi-layouter rollout (evaluate_scale.py rollout_scale, same
              k-unit init and T=50 as the Q14 size-extrapolation runs), saved
              to disk instead of only being timed.
  sfdp     -- graphviz sfdp (multilevel), run to convergence; layout saved.
  cuGraph FA2 -- cugraph force_atlas2 (one batched GPU call of max_iter), run
              in the RAPIDS venv; layout saved.

Each (source, family, size) layout is written as out/<source>_<family>_<n>.npy
([N,2] float64). A manifest JSON records wall-clock per layout and the phi
divergence step when the rollout did not stay finite (scale-free family).

Usage:
    .venv/bin/python eval/generate_large_layouts.py \\
        --checkpoint checkpoints/phi1_laponly_v1/executor_best.pt \\
        --corpora data/corpora --sizes 100000 1000000 \\
        --device cuda:0 --T 50 --chunk 500000 --skip_encoding \\
        --sfdp_bin /home/lei/miniconda3/bin/sfdp \\
        --cugraph_python /tmp/opencode/cugraph_venv/bin/python \\
        --out eval/results/large_layouts
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

CUG_LD = ("/tmp/opencode/cugraph_venv/lib/python3.12/site-packages/libcugraph/lib64:"
          "/tmp/opencode/cugraph_venv/lib/python3.12/site-packages/libcudf/lib64:"
          "/tmp/opencode/cugraph_venv/lib/python3.12/site-packages/librmm/lib64")


def phi_layout(encoder, model, edge_index, n, device, T, skip_encoding, chunk):
    """Roll the executor out, return (pos [N,2] or None if diverged,
    diverged_at step, step_mags, wall_clock)."""
    from philayouter.executor.evaluate_scale import rollout_scale

    t0 = time.time()
    final, step_mags, div_at = rollout_scale(
        encoder, model, edge_index, n, device, T=T,
        skip_encoding=skip_encoding)
    dt = time.time() - t0
    return final, div_at, step_mags, dt


def rollout_with_features(model, edge_index, n, device, T, node_feat):
    """Roll the executor out using PRE-COMPUTED real LapPE node features
    (cached by the encoding step), i.e. the paper's full model at scale.
    Mirrors evaluate_scale.rollout_scale exactly, replacing the random
    node_feat of --skip_encoding with the real structural encoding."""
    from philayouter.executor.evaluate_scale import k_unit_init

    k = float(np.sqrt(1.0 / n))
    pos = k_unit_init(n, device)
    edge_index = edge_index.to(device)
    temps = np.maximum(1.2 - np.arange(T) * (1.2 / (T + 1)), 1e-6).astype(np.float32)
    node_feat = torch.from_numpy(node_feat.astype(np.float32)).to(device)
    step_mags = []
    with torch.no_grad():
        for t in range(T):
            tau_val = float(temps[t])
            tau = torch.full((n, 1), tau_val, device=device)
            dX = model(node_feat, edge_index, pos, tau)
            pos = pos + dX
            step_mags.append(dX.norm(dim=-1).mean().item())
            if not torch.isfinite(pos).all():
                return None, np.array(step_mags), t
    return (pos * k).cpu().numpy(), np.array(step_mags), -1


def sfdp_layout(edge_index: np.ndarray, n: int, sfdp_bin: str, timeout=3600):
    """Run graphviz sfdp to convergence; return (pos [N,2], wall_clock)."""
    lines = ["graph G {", "  node [shape=point width=0];", "  edge [len=1.0];"]
    for e in range(edge_index.shape[1]):
        lines.append(f"  {int(edge_index[0, e])} -- {int(edge_index[1, e])};")
    lines.append("}")
    dot = "\n".join(lines)
    with tempfile.NamedTemporaryFile("w", suffix=".dot", delete=False) as f:
        f.write(dot)
        dotpath = f.name
    try:
        t0 = time.perf_counter()
        r = subprocess.run([sfdp_bin, "-T", "plain", dotpath],
                           capture_output=True, text=True, timeout=timeout)
        dt = time.perf_counter() - t0
        if r.returncode != 0:
            raise RuntimeError(f"sfdp rc={r.returncode}: {r.stderr.strip()[-300:]}")
        pos = np.full((n, 2), np.nan)
        for ln in r.stdout.splitlines():
            toks = ln.split()
            if len(toks) >= 4 and toks[0] == "node":
                pos[int(toks[1])] = (float(toks[2]), float(toks[3]))
        if np.isnan(pos).any():
            raise RuntimeError(f"sfdp output missing {int(np.isnan(pos).sum())} nodes")
        return pos, dt
    finally:
        os.unlink(dotpath)


def fa2_layout(edge_index: np.ndarray, n: int, rapids_python: str, max_iter=50,
               timeout=3600):
    """Run cugraph ForceAtlas2 in the RAPIDS venv; return (pos [N,2],
    wall_clock). Positions sorted by vertex id to align with 0..n-1."""
    edges_np = np.stack([edge_index[0], edge_index[1]], axis=1)
    code = textwrap.dedent(
        f"""
        import os
        import numpy as np
        import cudf, cugraph, time
        import cugraph.layout as cl
        arr = np.load(os.environ['NPZ'])['ei']
        edges = cudf.DataFrame({{'src': arr[:, 0], 'dst': arr[:, 1]}})
        G = cugraph.Graph(directed=False)
        G.from_cudf_edgelist(edges, source='src', destination='dst')
        t0 = time.perf_counter()
        pos = cl.force_atlas2(G, max_iter={max_iter}, jitter_tolerance=1.0)
        print(float(time.perf_counter() - t0))
        df = pos.sort_values('vertex')
        np.save(os.environ['OUT'], df[['x', 'y']].to_numpy().astype(np.float64))
        """
    )
    with tempfile.NamedTemporaryFile("wb", suffix=".npz", delete=False) as f:
        np.savez(f, ei=edges_np)
        npzpath = f.name
    with tempfile.NamedTemporaryFile("wb", suffix=".npy", delete=False) as f:
        outpath = f.name
    try:
        env = dict(os.environ, NPZ=npzpath, OUT=outpath,
                   LD_LIBRARY_PATH=CUG_LD)
        run = subprocess.run([rapids_python, "-c", code], env=env,
                             capture_output=True, text=True, timeout=timeout)
        if run.returncode != 0 or not os.path.exists(outpath):
            raise RuntimeError(f"FA2 failed: {run.stderr.strip()[-300:]}")
        pos = np.load(outpath)
        if pos.shape[0] != n:
            raise RuntimeError(f"FA2 returned {pos.shape[0]} rows for n={n}")
        return pos, float(run.stdout.strip().splitlines()[-1])
    finally:
        for p in (npzpath, outpath):
            if os.path.exists(p):
                os.unlink(p)


def main():
    ap = argparse.ArgumentParser(description="R1.2: save phi/sfdp/FA2 layouts")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--corpora", default="data/corpora")
    ap.add_argument("--sizes", type=int, nargs="+", default=[100000, 1000000])
    ap.add_argument("--families", nargs="+",
                    default=["grid", "rgg", "scale_free", "er"])
    ap.add_argument("--out", default="eval/results/large_layouts")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--skip_encoding", action="store_true")
    ap.add_argument("--sfdp_bin", default=None)
    ap.add_argument("--cugraph_python", default=None)
    ap.add_argument("--sources", nargs="+", default=["phi", "sfdp", "fa2"],
                    help="which sources to produce (phi needs --checkpoint; "
                         "sfdp needs --sfdp_bin; fa2 needs --cugraph_python; "
                         "phi_enc additionally needs --enc_dir)")
    ap.add_argument("--enc_dir", default=None,
                    help="dir of cached LapPE node-feature files "
                         "(<fam>_<n>.npy). When set, `phi_enc` is produced: "
                         "the FULL model rollout with real structural "
                         "encodings (the paper's Phi-layouter at scale), as "
                         "opposed to `phi`, which uses --skip_encoding random "
                         "features and is an ablation, not the model.")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    encoder = model = None
    if "phi" in args.sources or "phi_enc" in args.sources:
        from philayouter.executor.evaluate_scale import rollout_scale  # noqa
        from philayouter.executor.model import EquivariantExecutor
        from philayouter.executor.structural import StructuralEncoder

        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        cargs = ckpt["args"]
        encoder = StructuralEncoder(
            out_dim=cargs.get("hidden_dim", 64),
            lap_k=cargs.get("lap_k", 10),
            rw_k=cargs.get("rw_k", 16),
            fast_eigsh=True, use_rrwp=False)
        num_teachers = len(ckpt["teachers"]) if isinstance(ckpt.get("teachers"), dict) else 1
        model = EquivariantExecutor(
            node_feat_dim=cargs.get("hidden_dim", 64),
            hidden_dim=cargs.get("hidden_dim", 64),
            k_geo=cargs.get("k_geo", 10),
            num_teachers=num_teachers, knn_backend="kd", chunk=args.chunk,
        ).to(device)
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        model.load_state_dict(ckpt["model_state_dict"])
        encoder.eval()
        model.eval()
        print(f"loaded checkpoint {args.checkpoint}")

    os.makedirs(args.out, exist_ok=True)
    manifest = {}
    if os.path.isfile(os.path.join(args.out, "manifest.json")):
        with open(os.path.join(args.out, "manifest.json")) as f:
            manifest = json.load(f)      # merge, don't clobber
    for fam in args.families:
        for n in args.sizes:
            path = os.path.join(args.corpora, f"{fam}_{n}.pt")
            if not os.path.isfile(path):
                print(f"  (missing corpus {path})")
                continue
            ei = torch.load(path, weights_only=True)
            e_np = ei.numpy()
            entry = {"n": n}
            if "phi" in args.sources:
                final, div_at, step_mags, dt = phi_layout(
                    encoder, model, ei, n, device, args.T, args.skip_encoding, args.chunk)
                if final is None:
                    entry["phi"] = {"diverged_at": div_at, "wall_clock": round(dt, 2)}
                    print(f"  {fam:>10} {n:>9,}  phi DIVERGED @t{div_at}  {dt:.1f}s")
                else:
                    np.save(os.path.join(args.out, f"phi_{fam}_{n}.npy"), final)
                    entry["phi"] = {"wall_clock": round(dt, 2),
                                    "late_step_mag": float(np.mean(step_mags[-5:]))}
                    print(f"  {fam:>10} {n:>9,}  phi ok  {dt:.1f}s")
            if "phi_enc" in args.sources:
                if not args.enc_dir:
                    raise SystemExit("--enc_dir required for phi_enc source")
                feats_path = os.path.join(args.enc_dir, f"{fam}_{n}.npy")
                if not os.path.isfile(feats_path):
                    print(f"  {fam:>10} {n:>9,}  phi_enc (no cached features {feats_path})")
                else:
                    feats = np.load(feats_path)
                    t0 = time.time()
                    final, step_mags, div_at = rollout_with_features(
                        model, ei, n, device, args.T, feats)
                    dt = time.time() - t0
                    if final is None:
                        entry["phi_enc"] = {"diverged_at": div_at, "wall_clock": round(dt, 2)}
                        print(f"  {fam:>10} {n:>9,}  phi_enc DIVERGED @t{div_at}  {dt:.1f}s")
                    else:
                        np.save(os.path.join(args.out, f"phi_enc_{fam}_{n}.npy"), final)
                        entry["phi_enc"] = {"wall_clock": round(dt, 2),
                                            "late_step_mag": float(np.mean(step_mags[-5:]))}
                        print(f"  {fam:>10} {n:>9,}  phi_enc ok  {dt:.1f}s")
            if "sfdp" in args.sources:
                if not args.sfdp_bin:
                    raise SystemExit("--sfdp_bin required for sfdp source")
                t0 = time.time()
                pos, dt = sfdp_layout(e_np, n, args.sfdp_bin)
                np.save(os.path.join(args.out, f"sfdp_{fam}_{n}.npy"), pos)
                entry["sfdp"] = {"wall_clock": round(dt, 2)}
                print(f"  {fam:>10} {n:>9,}  sfdp ok  {dt:.1f}s")
            if "fa2" in args.sources:
                if not args.cugraph_python:
                    raise SystemExit("--cugraph_python required for fa2 source")
                t0 = time.time()
                pos, dt = fa2_layout(e_np, n, args.cugraph_python, max_iter=args.T)
                np.save(os.path.join(args.out, f"fa2_{fam}_{n}.npy"), pos)
                entry["fa2"] = {"wall_clock": round(dt, 2)}
                print(f"  {fam:>10} {n:>9,}  fa2 ok  {dt:.1f}s")
            manifest[f"{fam}_{n}"] = entry

    mpath = os.path.join(args.out, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest -> {mpath}")


if __name__ == "__main__":
    main()
