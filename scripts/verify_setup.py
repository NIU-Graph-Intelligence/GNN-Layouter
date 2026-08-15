#!/usr/bin/env python3
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXPECTED_SHA256 = {
    "data/processed/comm_5k_final_v2.pt":
        "1948a051e90d634d2c766af26000a00f1bb5da9f9d544f76fd0e79cf71da0539",
    "data/processed/comm_5k_v2_with_encodings.pt":
        "a5f1d5626b5a0ef96bc9b58e20e4749694fde1de4b0c6c3725db68ac10333498",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    import torch
    import torch_geometric

    print(f"Python       {sys.version.split()[0]}")
    print(f"PyTorch      {torch.__version__}")
    print(f"PyG          {torch_geometric.__version__}")
    print(f"CUDA         {torch.cuda.is_available()} ({torch.cuda.device_count()} GPUs)")

    for relative, expected in EXPECTED_SHA256.items():
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f"Missing required asset: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"Checksum mismatch: {relative}\n{actual}")
        print(f"asset ok     {relative}")

    dataset_path = ROOT / "data/processed/comm_5k_v2_with_encodings.pt"
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if isinstance(dataset, dict):
        dataset = dataset["dataset"]
    if len(dataset) != 5000:
        raise SystemExit(f"Expected 5000 graphs, found {len(dataset)}")
    sample = dataset[0]
    required = ("x", "y", "y_traj", "edge_index", "community")
    missing = [name for name in required if getattr(sample, name, None) is None]
    if missing:
        raise SystemExit(f"Dataset sample lacks: {', '.join(missing)}")
    print(f"dataset ok   {len(dataset)} graphs, x={tuple(sample.x.shape)}, "
          f"trajectory={tuple(sample.y_traj.shape)}")


if __name__ == "__main__":
    main()
