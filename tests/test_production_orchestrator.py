import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from production.config import load_production_config
from production.orchestrator import run_production
from production.staging import StagingResult
from production.state_store import StateStore
from recsys_vnpt.batch_contracts import BatchError


VN = ZoneInfo("Asia/Ho_Chi_Minh")


def _config(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    raw = {
        "schema_version": 1,
        "timezone": "Asia/Ho_Chi_Minh",
        "schedule": {"enabled": True, "run_times": ["00:00", "12:00"]},
        "source_root": str(source),
        "batch": {
            "schema_version": 1,
            "domain": "all",
            "incoming_root": str(tmp_path / "incoming"),
            "work_root": str(tmp_path / "work"),
            "delivery_root": str(tmp_path / "delivery"),
            "model_registry": str(tmp_path / "models"),
            "combination": "GxEp",
            "input_wait_seconds": 0,
            "min_free_bytes": 0,
            "num_shards": 1,
            "workers": 1,
            "max_profiles": 1000,
        },
    }
    path = tmp_path / "production.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return load_production_config(path)


def _run(tmp_path, *, gate_error=None, retrain_error=None, infer_error=None, model=True):
    config = _config(tmp_path)
    store = StateStore(tmp_path / "state.db")
    slot = datetime(2026, 11, 13, 12, tzinfo=VN)
    builder = lambda cfg, batch_id, started: StagingResult(
        cfg.incoming_root / batch_id, "input-hash", "source-hash"
    )

    def retrain(cfg, batch_id):
        if gate_error:
            raise gate_error
        if retrain_error:
            raise retrain_error
        return {"status": "promoted", "bundle": "bundle-new", "gate": {"passed": True}}

    def infer(cfg, batch_id):
        if infer_error:
            raise infer_error
        return {"status": "published", "generation": "generation-1", "files": ["a.csv", "b.csv", "c.csv"]}

    def promoted(cfg):
        if not model:
            raise BatchError("MODEL_UNAVAILABLE", "there is no valid promoted model")
        return Path("previous-bundle"), {"bundle_id": "bundle-old", "model_sha256": "abc"}

    result = run_production(
        config,
        slot,
        store=store,
        source_builder=builder,
        retrain=retrain,
        infer=infer,
        promoted_resolver=promoted,
    )
    return result, store, config, slot


def test_published_slot_is_verified_before_staging_or_retraining(tmp_path, monkeypatch):
    config = _config(tmp_path)
    store = StateStore(tmp_path / "state.db")
    slot = datetime(2026, 11, 13, 12, tzinfo=VN)
    current = {
        "batch_id": "20261113_1200",
        "data_cutoff": "2026-11-13T08:00:00+07:00",
        "generation": "generation-already-published",
        "bundle_id": "bundle-old",
        "input_fingerprint": "published-input",
        "directory": str(tmp_path / "delivery" / "generation"),
        "files": {"home.csv": {}, "rule.csv": {}, "content.csv": {}},
    }
    monkeypatch.setattr("recsys_vnpt.export.read_current_delivery", lambda _: current)

    def forbidden(*args, **kwargs):
        raise AssertionError("published slots must not restage or retrain")

    result = run_production(
        config,
        slot,
        store=store,
        source_builder=forbidden,
        retrain=forbidden,
        infer=forbidden,
    )
    assert result["already_published"] is True
    assert result["retrain_status"] == "PENDING"
    assert store.jobs(limit=1)[0]["status"] == "SUCCESS"


def test_prepare_only_job_stops_before_retrain_and_inference(tmp_path, monkeypatch):
    from recsys_vnpt import batch_runner

    config = _config(tmp_path)
    store = StateStore(tmp_path / "state.db")
    slot = datetime(2026, 11, 13, 12, tzinfo=VN)
    builder = lambda cfg, batch_id, started: StagingResult(
        cfg.incoming_root / batch_id, "input-hash", "source-hash"
    )
    monkeypatch.setattr(
        batch_runner,
        "run_prepare_data_batch",
        lambda cfg, batch_id: {
            "status": "prepared",
            "batch_id": batch_id,
            "prepared_dataset": "prepared.json",
            "reused": False,
        },
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("prepare-only must not retrain or infer")

    result = run_production(
        config,
        slot,
        store=store,
        source_builder=builder,
        prepare_only=True,
        retrain=unexpected,
        infer=unexpected,
    )

    assert result["status"] == "PREPARED"
    assert result["preparation_status"] == "READY"
    assert result["prepared_dataset"] == "prepared.json"
    assert store.jobs(limit=1)[0]["status"] == "PREPARED"

    production_result = run_production(
        config,
        slot,
        store=store,
        source_builder=builder,
        retrain=lambda *_: {"status": "promoted", "bundle": "new-bundle", "gate": {}},
        infer=lambda *_: {"status": "published", "generation": "generation-1", "files": []},
    )
    assert production_result["preparation_status"] == "READY"
    assert production_result["prepared_dataset"] == "prepared.json"


def test_newer_published_delivery_blocks_batch_before_staging(tmp_path, monkeypatch):
    config = _config(tmp_path)
    store = StateStore(tmp_path / "state.db")
    slot = datetime(2026, 11, 13, 12, tzinfo=VN)
    current = {
        "batch_id": "newer-slot",
        "batch_started_at": "2026-11-13T09:00:00+07:00",
        "generation": "newer-generation",
        "directory": str(tmp_path / "delivery" / "generation"),
        "files": {},
    }
    monkeypatch.setattr("recsys_vnpt.export.read_current_delivery", lambda _: current)

    def forbidden(*args, **kwargs):
        raise AssertionError("an older batch must stop before staging")

    with pytest.raises(BatchError, match="does not advance past published batch"):
        run_production(
            config, slot, store=store, source_builder=forbidden,
            retrain=forbidden, infer=forbidden,
        )
    report = store.jobs(limit=1)[0]["report"]
    assert report["input_status"] == "PENDING"
    assert report["retrain_status"] == "PENDING"


def test_input_failure_does_not_claim_retraining_or_inference_ran(tmp_path):
    config = _config(tmp_path)
    store = StateStore(tmp_path / "state.db")
    slot = datetime(2026, 11, 13, 12, tzinfo=VN)

    def broken_source(*args):
        raise BatchError("INPUT_INTEGRITY", "source has date-only events")

    with pytest.raises(BatchError, match="INPUT_INTEGRITY"):
        run_production(config, slot, store=store, source_builder=broken_source)
    report = store.jobs(limit=1)[0]["report"]
    assert report["input_status"] == "FAILED"
    assert report["retrain_status"] == "PENDING"
    assert report["inference_status"] == "PENDING"


def test_retry_keeps_original_batch_started_at(tmp_path):
    config = _config(tmp_path)
    store = StateStore(tmp_path / "state.db")
    slot = datetime(2026, 11, 13, 12, tzinfo=VN)
    observed_starts = []

    def builder(cfg, batch_id, started):
        observed_starts.append(started)
        return StagingResult(cfg.incoming_root / batch_id, "same-input", "same-source")

    def fail_retrain(*_):
        raise BatchError("RETRAIN_FAILED", "temporary GPU failure")

    with pytest.raises(BatchError, match="RETRAIN_FAILED"):
        run_production(config, slot, store=store, source_builder=builder, retrain=fail_retrain)

    run_production(
        config,
        slot,
        store=store,
        source_builder=builder,
        retrain=lambda *_: {"status": "promoted", "bundle": "bundle-new", "gate": {"passed": True}},
        infer=lambda *_: {"status": "published", "generation": "generation-2", "files": []},
    )
    assert observed_starts[0] == observed_starts[1]
    assert observed_starts[0].tzinfo is not None
    assert store.get_job("20261113_1200")["batch_started_at"] == observed_starts[0].isoformat()


def test_saved_data_settings_are_applied_to_the_next_batch(tmp_path):
    config = _config(tmp_path)
    store = StateStore(tmp_path / "state.db")
    store.save_data_settings({
        "time_train": 12,
        "time_valid": 6,
        "time_test": 3,
        "max_profiles": 500,
        "gate_evaluation_profiles": 250,
    })
    slot = datetime(2026, 11, 13, 12, tzinfo=VN)
    seen = []
    builder = lambda cfg, batch_id, started: StagingResult(
        cfg.incoming_root / batch_id, "input-hash", "source-hash"
    )

    def retrain(batch_config, _):
        seen.append(batch_config)
        return {"status": "promoted", "bundle": "bundle", "gate": {"passed": True}}

    run_production(
        config,
        slot,
        store=store,
        source_builder=builder,
        retrain=retrain,
        infer=lambda *_: {"status": "published", "generation": "generation", "files": []},
    )

    assert seen[0].time_train == 12
    assert seen[0].time_valid == 6
    assert seen[0].time_test == 3
    assert seen[0].max_profiles == 500
    assert seen[0].gate_evaluation_profiles == 250


def test_gate_pass_uses_newly_promoted_model_and_writes_report(tmp_path):
    result, store, config, _ = _run(tmp_path)
    assert result["retrain_status"] == "PROMOTED"
    assert result["model_used"] == "NEWLY_PROMOTED"
    assert result["inference_status"] == "SUCCESS"
    assert result["publish_status"] == "SUCCESS"
    assert result["input_fingerprint"] == "input-hash"
    assert len(result["output_files"]) == 3
    assert len(result["output_file_paths"]) == 3
    assert (config.work_root / "batches" / result["batch_id"] / "production" / "report.json").is_file()
    assert store.jobs(limit=1)[0]["status"] == "SUCCESS"


def test_gate_rejection_falls_back_to_existing_promoted_model(tmp_path):
    gate = BatchError("QUALITY_GATE_FAILED", "new model regressed")
    result, _, _, _ = _run(tmp_path, gate_error=gate)
    assert result["retrain_status"] == "REJECTED_BY_QUALITY_GATE"
    assert result["model_used"] == "PREVIOUS_PROMOTED"
    assert result["model_bundle_id"] == "bundle-old"
    assert result["inference_status"] == "SUCCESS"
    assert result["status"] == "SUCCESS"


def test_gate_rejection_without_usable_promoted_model_is_failure(tmp_path):
    gate = BatchError("QUALITY_GATE_FAILED", "no positives")
    with pytest.raises(BatchError, match="MODEL_UNAVAILABLE"):
        _run(tmp_path, gate_error=gate, model=False)


def test_non_gate_retraining_failure_is_not_swallowed(tmp_path):
    error = BatchError("RETRAIN_FAILED", "training crashed")
    with pytest.raises(BatchError, match="RETRAIN_FAILED"):
        _run(tmp_path, retrain_error=error)


def test_inference_failure_never_reports_success(tmp_path):
    error = BatchError("INFERENCE_FAILED", "shard failure")
    with pytest.raises(BatchError, match="INFERENCE_FAILED"):
        _run(tmp_path, infer_error=error)

