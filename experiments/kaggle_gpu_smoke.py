"""Exercise the shared neural model runner on a verified Kaggle CUDA device."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.kaggle_preflight import cuda_probe

try:
    import torch
except ImportError:  # CPU-only developers can still run the contract tests.
    torch = None


def smoke_frame(rows: int) -> pd.DataFrame:
    """Build a deterministic graded-label frame with profile/content features."""
    return pd.DataFrame(
        {
            "profile_id": [f"p{i % 4}" for i in range(rows)],
            "content_id": [f"c{i % 6}" for i in range(rows)],
            "type_id": [str(i % 2) for i in range(rows)],
            "package_code": ["A" if i % 2 else "B" for i in range(rows)],
            "tag_names": ["sports,movie" if i % 2 else "kids" for i in range(rows)],
            "historical_interaction_count": [i + 1 for i in range(rows)],
            "content_duration": [100 + i for i in range(rows)],
            "label": [i % 2 for i in range(rows)],
            "completion_gain": [float(i % 2) for i in range(rows)],
            "watch_time_gain": [float(i % 3) / 2 for i in range(rows)],
            "target_semantics": ["engagement_v1"] * rows,
        }
    )


def run_smoke(output_dir: Path) -> dict[str, Any]:
    """Train one epoch, reload the checkpoint, and predict strictly on CUDA."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if torch is None:
        raise RuntimeError("CUDA unavailable: PyTorch is not installed")
    gpu = cuda_probe(torch)

    # Import only after CUDA is verified; the gate never falls through to CPU.
    from recsys_vnpt.model_runner import (
        TrainingConfig,
        load_v2_checkpoint,
        predict,
        train_model,
    )

    config = TrainingConfig(
        model_family="embedding_fcn",
        epochs=1,
        patience=1,
        batch_size=4,
        hidden_dims=(8,),
        embedding_dim=4,
        numeric_projection_dim=4,
    )
    train, validation = smoke_frame(24), smoke_frame(8)
    _, _, metrics = train_model(
        train,
        validation,
        "all",
        output_dir,
        config,
        device="cuda",
        require_cuda=True,
    )

    checkpoint_path = output_dir / "best_model_v2.pth"
    contract_path = output_dir / "feature_contract_v2.json"
    result_path = output_dir / "gpu_smoke.json"
    model, contract, _ = load_v2_checkpoint(
        checkpoint_path,
        contract_path,
        device="cuda",
    )
    scores = predict(
        model,
        contract,
        validation,
        "embedding_fcn",
        device="cuda",
    )
    if len(scores) != 8 or not np.isfinite(scores).all():
        raise RuntimeError("CUDA smoke predictions are invalid")

    result = {
        **gpu,
        "rows_scored": len(scores),
        "prediction_sha256": hashlib.sha256(
            np.asarray(scores, dtype=np.float64).tobytes()
        ).hexdigest(),
        "training_data_hash": metrics["training_data_hash"],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "contract_path": str(contract_path.resolve()),
        "result_path": str(result_path.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    }
    result_path.write_text(json.dumps(result, sort_keys=True, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_smoke(args.output_dir)
    except Exception as error:
        print(f"Kaggle CUDA smoke failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
