"""FCN Official checkpoint loading and bounded candidate scoring."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recsys_vnpt.feature_contract import FeatureContract
from recsys_vnpt.model_runner import (
    load_v2_checkpoint,
    predict,
    prepare_inference_input,
)


@dataclass(frozen=True)
class FCNScorerModel:
    model: Any
    contract: FeatureContract
    checkpoint: dict[str, Any]
    model_family: str
    device: str
    artifact_hash: str = ""


def _resolve_device(device: str) -> str:
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("FCN device must be auto, cpu, or cuda")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for FCN scoring") from error
    available = bool(torch.cuda.is_available())
    if device == "cuda" and not available:
        raise RuntimeError("FCN device 'cuda' requested but CUDA is unavailable")
    return "cuda" if device == "cuda" or (device == "auto" and available) else "cpu"


def load_fcn_scorer(
    checkpoint_path: str | Path,
    contract_path: str | Path,
    device: str = "auto",
    model_family: str = "fcn_official",
) -> FCNScorerModel:
    """Load FCN Official through the existing checkpoint/FeatureContract gate."""
    if model_family != "fcn_official":
        raise ValueError("content FCN scorer requires model_family='fcn_official'")
    checkpoint = Path(checkpoint_path)
    contract = Path(contract_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"FCN checkpoint is missing: {checkpoint}")
    if not contract.is_file():
        raise FileNotFoundError(f"FCN feature contract is missing: {contract}")
    active_device = _resolve_device(device)
    model, feature_contract, checkpoint_data = load_v2_checkpoint(
        checkpoint, contract, device=active_device
    )
    if feature_contract.model_family != model_family:
        raise ValueError(
            "FCN checkpoint FeatureContract family mismatch: "
            f"{feature_contract.model_family} != {model_family}"
        )
    return FCNScorerModel(
        model=model,
        contract=feature_contract,
        checkpoint=checkpoint_data,
        model_family=model_family,
        device=active_device,
        artifact_hash=_sha256_file(checkpoint),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_fcn_candidates(
    scorer: FCNScorerModel,
    feature_frame: pd.DataFrame,
    batch_size: int = 4096,
    amp: bool = False,
) -> np.ndarray:
    """Transform using the checkpoint contract and predict in bounded batches."""
    if batch_size < 1:
        raise ValueError("FCN scoring batch_size must be positive")
    prepared = prepare_inference_input(
        scorer.contract, feature_frame, scorer.model_family
    )
    return predict(
        scorer.model,
        scorer.contract,
        feature_frame,
        scorer.model_family,
        batch_size=batch_size,
        device=scorer.device,
        amp=amp,
        prepared=prepared,
    )
