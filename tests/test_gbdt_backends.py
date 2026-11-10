"""Both gradient-boosting backends, and the GPU fallback the XGBoost one relies on."""

import json
from pathlib import Path

import numpy as np
import pytest

import recsys_vnpt.batch_runner as runner
from recsys_vnpt import gbdt
from recsys_vnpt.batch_contracts import BatchConfig, BatchError
from recsys_vnpt.export import read_current_delivery
from recsys_vnpt.output_validation import validate_client_output
from recsys_vnpt.scorer_bundle import BundleError, load_bundle, promoted_bundle
from recsys_vnpt.smoke_fixture import write_smoke_batch

HEAD = dict(seed=42, learning_rate=0.06, max_iter=60, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0)


def _data(rows: int = 4_000):
    rng = np.random.default_rng(0)
    x = rng.random((rows, 4), dtype=np.float32)
    x[:, 0] = rng.integers(0, 12, rows)
    x[rng.random(rows) < 0.05, 0] = np.nan  # a category the model never saw
    y = (np.nan_to_num(x[:, 0]) % 3 == 0) + x[:, 1]
    return x, y.astype(np.float32), np.where(y > 1, 3.0, 1.0).astype(np.float32)


@pytest.mark.parametrize("backend", gbdt.GBDT_BACKENDS)
def test_each_backend_learns_categorical_and_numeric_signal(backend):
    x, y, weights = _data()
    head = gbdt.make_head(backend, "cpu", [True, False, False, False], **HEAD)
    head.fit(x, y, sample_weight=weights)
    predicted = head.predict(x)
    assert head.n_iter_ >= 1
    assert np.isfinite(predicted).all()
    assert np.corrcoef(predicted, y)[0, 1] > 0.8


def test_the_sklearn_head_keeps_the_production_configuration():
    head = gbdt.make_head("sklearn", "auto", [True, False], **HEAD)
    params = head.get_params()
    assert params["loss"] == "squared_error" and params["early_stopping"] is True
    assert (params["validation_fraction"], params["n_iter_no_change"], params["random_state"]) == (0.1, 8, 42)
    assert params["categorical_features"] == [True, False]


def test_xgboost_training_is_deterministic():
    x, y, weights = _data()
    first = gbdt.make_head("xgboost", "cpu", [True, False, False, False], **HEAD).fit(x, y, weights)
    second = gbdt.make_head("xgboost", "cpu", [True, False, False, False], **HEAD).fit(x, y, weights)
    assert np.array_equal(first.predict(x), second.predict(x))


def test_a_gpu_request_without_a_gpu_fails_training_but_serving_falls_back(monkeypatch):
    x, y, weights = _data()
    fitted = gbdt.make_head("xgboost", "auto", [True, False, False, False], **HEAD).fit(x, y, weights)
    monkeypatch.setattr(gbdt, "cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CUDA GPU"):
        gbdt.make_head("xgboost", "cuda", [True, False, False, False], **HEAD).fit(x, y, weights)
    fitted.device = "cuda"
    assert np.isfinite(fitted.predict(x)).all()
    assert gbdt.resolve_device("auto", strict=True) == "cpu"


def test_strict_serving_device_refuses_cuda_fallback(monkeypatch):
    head = gbdt.XGBoostHead(
        [False], "auto", seed=42, learning_rate=0.06, max_iter=2,
        max_leaf_nodes=3, min_samples_leaf=1, l2_regularization=1.0,
    )
    model = type("Model", (), {"heads": {"head": head}})()
    monkeypatch.setattr(gbdt, "cuda_available", lambda: False)

    with pytest.raises(RuntimeError, match="no CUDA GPU"):
        gbdt.serve_on(model, "cuda")


def test_unknown_backend_or_device_is_refused():
    with pytest.raises(ValueError):
        gbdt.make_head("lightgbm", "cpu", [False], **HEAD)
    with pytest.raises(ValueError):
        gbdt.make_head("sklearn", "tpu", [False], **HEAD)
    with pytest.raises(BatchError):
        BatchConfig("i", "w", "d", "m", gbdt_backend="lightgbm")


def _gate_passes(monkeypatch) -> None:
    """The plumbing under test is retrain -> promote -> serve. Whether XGBoost
    beats the hand-built ordering is measured on the full data: on the 30
    synthetic profiles the hand-built ordering is already near its ceiling and
    either backend lands within noise of it."""
    real = runner.retrain_gate

    def gate(*args, **kwargs):
        return {**real(*args, **kwargs), "passed": True}

    monkeypatch.setattr(runner, "retrain_gate", gate)


def _xgboost_config(tmp_path, device: str) -> BatchConfig:
    return BatchConfig(
        incoming_root=str(tmp_path / "incoming"),
        work_root=str(tmp_path / "work"),
        delivery_root=str(tmp_path / "delivery"),
        model_registry=str(tmp_path / "registry"),
        min_free_bytes=0,
        input_wait_seconds=0,
        num_shards=2,
        gbdt_backend="xgboost",
        gbdt_device=device,
    )


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not gbdt.cuda_available(), reason="no CUDA GPU"))],
)
def test_the_batch_retrains_and_serves_with_xgboost(tmp_path, device, monkeypatch):
    _gate_passes(monkeypatch)
    write_smoke_batch(tmp_path / "incoming", "b1")
    config = _xgboost_config(tmp_path, device)

    retrained = runner.run_retrain_batch(config, "b1")
    assert retrained["status"] == "promoted"
    bundle = promoted_bundle(config.model_registry)
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    assert manifest["rule_scorer"]["rule_scorer_backend"] == "xgboost"
    assert manifest["content_scorer"]["content_scorer_device"] == device
    assert manifest["libraries"]["xgboost"]

    report = runner.run_inference_batch(config, "b1")
    assert report["status"] == "published"
    current = read_current_delivery(config.delivery_root)
    directory = Path(current["directory"])
    names = sorted(name for name in current["files"] if name.startswith("f_"))
    validation = validate_client_output(
        json.loads((directory / names[0]).read_text(encoding="utf-8")),
        (directory / names[1]).read_text(encoding="utf-8"),
        config=config.assembly,
    )
    assert validation.passed, validation.errors[:3]


def test_a_bundle_from_another_xgboost_release_is_refused(tmp_path, monkeypatch):
    _gate_passes(monkeypatch)
    write_smoke_batch(tmp_path / "incoming", "b1")
    config = _xgboost_config(tmp_path, "cpu")
    runner.run_retrain_batch(config, "b1")
    bundle = promoted_bundle(config.model_registry)
    monkeypatch.setattr("recsys_vnpt.scorer_bundle._xgboost_version", lambda: "0.0.0")
    with pytest.raises(BundleError, match="xgboost"):
        load_bundle(bundle)
