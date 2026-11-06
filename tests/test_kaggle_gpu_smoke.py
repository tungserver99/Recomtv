import json
from types import SimpleNamespace

import pytest

from experiments import kaggle_gpu_smoke
from experiments.kaggle_gpu_smoke import run_smoke, smoke_frame


def test_smoke_frame_has_profile_identity_and_features():
    frame = smoke_frame(8)

    assert len(frame) == 8
    assert {"profile_id", "content_id", "tag_names", "label"} <= set(frame)
    assert {"completion_gain", "watch_time_gain", "target_semantics"} <= set(frame)
    assert frame["profile_id"].iloc[0] == "p0"


def test_smoke_fails_without_cuda(tmp_path, monkeypatch):
    cpu_only_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setattr(kaggle_gpu_smoke, "torch", cpu_only_torch)

    with pytest.raises(RuntimeError, match="CUDA"):
        run_smoke(tmp_path)


def test_smoke_saves_checkpoint_and_hashes_finite_predictions_on_cuda(tmp_path):
    """Exercise real GPU training only when this machine actually has CUDA."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA integration test runs on the Kaggle GPU gate")

    result = run_smoke(tmp_path)

    assert result["gpu_verified"] is True
    assert result["rows_scored"] == 8
    assert len(result["prediction_sha256"]) == 64
    assert len(result["checkpoint_sha256"]) == 64
    assert (tmp_path / "best_model_v2.pth").is_file()
    assert (tmp_path / "feature_contract_v2.json").is_file()
    assert json.loads((tmp_path / "gpu_smoke.json").read_text())["gpu_verified"] is True
