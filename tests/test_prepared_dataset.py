from __future__ import annotations

import json
from datetime import datetime, timezone
from dataclasses import replace

import pytest

import recsys_vnpt.batch_runner as runner
from recsys_vnpt.batch_cli import main as batch_cli_main
from recsys_vnpt.batch_contracts import BatchError, ReadyManifest
from recsys_vnpt.prepared_dataset import (
    load_prepared_dataset,
    prepared_dataset_paths,
    write_prepared_dataset_manifest,
)
from recsys_vnpt.source_manifest import sha256_file


def _config(tmp_path, **overrides):
    return runner.BatchConfig(
        incoming_root=str(tmp_path / "incoming"),
        work_root=str(tmp_path / "work"),
        delivery_root=str(tmp_path / "delivery"),
        model_registry=str(tmp_path / "registry"),
        min_free_bytes=0,
        input_wait_seconds=0,
        **overrides,
    )


def test_inference_requires_prepared_data_and_never_falls_back_to_preparation(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    batch = tmp_path / "incoming" / "b1"
    batch.mkdir(parents=True)
    (batch / "_READY.json").write_text("{}", encoding="utf-8")
    manifest = ReadyManifest(
        "b1", datetime(2025, 3, 1, tzinfo=timezone.utc),
        datetime(2025, 3, 1, tzinfo=timezone.utc), (),
    )
    monkeypatch.setattr(runner, "_validate_input", lambda *_, **__: manifest)

    def unexpected_prepare(*args, **kwargs):
        raise AssertionError("implicit preparation must not run")

    monkeypatch.setattr(runner, "_prepare", unexpected_prepare)

    with pytest.raises(BatchError, match="prepare-data") as error:
        runner.run_inference_batch(config, "b1")

    assert error.value.code == "PREPARED_DATA_UNAVAILABLE"


def _manifest(config, batch_id="b1"):
    source = config.root("incoming_root") / batch_id / "duration" / "d.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source")
    return ReadyManifest(
        batch_id,
        datetime(2025, 3, 1, tzinfo=timezone.utc),
        datetime(2025, 3, 1, tzinfo=timezone.utc),
        ({"path": "duration/d.csv", "size_bytes": source.stat().st_size, "sha256": sha256_file(source)},),
    )


def _materialize_prepared_tree(paths, domain="all"):
    canonical = paths.artifact_root / "canonical" / domain
    temporal = canonical / "temporal"
    temporal.mkdir(parents=True)
    (canonical / "users.parquet").write_bytes(b"users")
    (canonical / "interactions.parquet").mkdir()
    (canonical / "interactions.parquet" / "part-0.parquet").write_bytes(b"interactions")
    (canonical / "profile_mapping.parquet").write_bytes(b"mapping")
    (canonical / "contents.parquet").write_bytes(b"contents")
    (canonical / "content_versions.parquet").write_bytes(b"versions")
    (canonical / "content_rule_membership.parquet").write_bytes(b"membership")
    (canonical / "playback_events.parquet").write_bytes(b"events")
    event_part = canonical / "playback_events_by_date" / "event_date=2025-01-01" / "part-00000.parquet"
    event_part.parent.mkdir(parents=True)
    event_part.write_bytes(b"event partition")
    (canonical / "label_contract_v4.json").write_text("{}", encoding="utf-8")
    (temporal / "train_1.parquet").write_bytes(b"train")
    (temporal / "train_2.parquet").write_bytes(b"train")
    (temporal / "train_3.parquet").write_bytes(b"train")
    (temporal / "validation.parquet").write_bytes(b"valid")
    (temporal / "test.parquet").write_bytes(b"test")
    (temporal / "windows.json").write_text(
        json.dumps({"schema_version": 2, "time_config": {}, "windows": []}), encoding="utf-8"
    )


def _write_ready_placeholder(config, batch_id="b1"):
    batch = config.root("incoming_root") / batch_id
    batch.mkdir(parents=True, exist_ok=True)
    (batch / "_READY.json").write_text("{}", encoding="utf-8")
    return _manifest(config, batch_id)


def test_prepared_dataset_is_pinned_to_input_and_prepare_configuration(tmp_path):
    config = _config(tmp_path)
    manifest = _manifest(config)
    paths = prepared_dataset_paths(config, manifest)
    _materialize_prepared_tree(paths, config.domain)
    write_prepared_dataset_manifest(config, manifest, status="READY")

    dataset = load_prepared_dataset(config, manifest)

    assert dataset.artifact_root == paths.artifact_root
    assert dataset.manifest["prepared_at"]
    changed_config = replace(config, time_train=config.time_train + 1)
    with pytest.raises(BatchError, match="prepare-data") as error:
        load_prepared_dataset(changed_config, manifest)
    assert error.value.code == "PREPARED_DATA_UNAVAILABLE"


def test_prepared_dataset_rejects_missing_or_changed_artifacts(tmp_path):
    config = _config(tmp_path)
    manifest = _manifest(config)
    paths = prepared_dataset_paths(config, manifest)
    _materialize_prepared_tree(paths, config.domain)
    write_prepared_dataset_manifest(config, manifest, status="READY")
    (paths.artifact_root / "canonical" / "all" / "users.parquet").write_bytes(b"tampered")

    with pytest.raises(BatchError, match="artifact") as error:
        load_prepared_dataset(config, manifest)
    assert error.value.code == "PREPARED_DATA_UNAVAILABLE"


def test_retrain_metrics_pointer_changes_do_not_invalidate_prepared_data(tmp_path):
    config = _config(tmp_path)
    manifest = _manifest(config)
    paths = prepared_dataset_paths(config, manifest)
    _materialize_prepared_tree(paths, config.domain)
    metrics = paths.artifact_root / "metrics" / "latest.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text('{"phase":"prepare"}', encoding="utf-8")
    manifest_path = write_prepared_dataset_manifest(config, manifest, status="READY")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["artifacts"].append({
        "path": "metrics/latest.json",
        "size_bytes": metrics.stat().st_size,
    })
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    metrics.write_text('{"phase":"retrain","run_id":"a-new-run-with-longer-metadata"}', encoding="utf-8")

    dataset = load_prepared_dataset(config, manifest)
    assert dataset.artifact_root == paths.artifact_root


def test_prepared_dataset_rejects_changed_raw_input(tmp_path):
    config = _config(tmp_path)
    manifest = _manifest(config)
    paths = prepared_dataset_paths(config, manifest)
    _materialize_prepared_tree(paths, config.domain)
    write_prepared_dataset_manifest(config, manifest, status="READY")
    source = config.root("incoming_root") / "b1" / "duration" / "d.csv"
    source.write_bytes(b"changed input")

    with pytest.raises(BatchError, match="raw input changed") as error:
        load_prepared_dataset(config, manifest)

    assert error.value.code == "PREPARED_DATA_UNAVAILABLE"


def test_prepare_data_job_writes_ready_manifest_and_reuses_unchanged_input(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    manifest = _write_ready_placeholder(config)
    monkeypatch.setattr(runner, "_validate_input", lambda *_, **__: manifest)
    calls = []

    def fake_prepare(config_arg, manifest_arg, *, temporal, progress, data_root, timings):
        calls.append((temporal, data_root))
        paths = prepared_dataset_paths(config_arg, manifest_arg)
        _materialize_prepared_tree(paths, config_arg.domain)
        timings["canonical_seconds"] = 0.1
        timings["temporal_seconds"] = 0.2
        return data_root

    monkeypatch.setattr(runner, "_prepare", fake_prepare)

    first = runner.run_prepare_data_batch(config, "b1")
    second = runner.run_prepare_data_batch(config, "b1")

    assert first["status"] == "prepared"
    assert first["reused"] is False
    assert second["reused"] is True
    assert len(calls) == 1
    assert first["prepared_dataset"] == second["prepared_dataset"]


def test_prepare_data_job_retries_after_interruption_using_same_artifact_key(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    manifest = _write_ready_placeholder(config)
    monkeypatch.setattr(runner, "_validate_input", lambda *_, **__: manifest)
    roots = []

    def flaky_prepare(config_arg, manifest_arg, *, temporal, progress, data_root, timings):
        roots.append(data_root)
        if len(roots) == 1:
            raise RuntimeError("simulated interruption")
        _materialize_prepared_tree(prepared_dataset_paths(config_arg, manifest_arg), config_arg.domain)
        return data_root

    monkeypatch.setattr(runner, "_prepare", flaky_prepare)
    with pytest.raises(BatchError):
        runner.run_prepare_data_batch(config, "b1")

    result = runner.run_prepare_data_batch(config, "b1")

    assert result["status"] == "prepared"
    assert roots[0] == roots[1]
    status = json.loads(
        (config.root("work_root") / "batches" / "b1" / "prepare" / "status.json").read_text()
    )
    assert status["attempt"] == 2
    assert status["phase"] == "PREPARED"


def test_batch_cli_exposes_prepare_data_command(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "batch.json"
    config_path.write_text(
        json.dumps(
            {
                "incoming_root": str(tmp_path / "incoming"),
                "work_root": str(tmp_path / "work"),
                "delivery_root": str(tmp_path / "delivery"),
                "model_registry": str(tmp_path / "registry"),
                "min_free_bytes": 0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner,
        "run_prepare_data_batch",
        lambda config, batch_id: {"status": "prepared", "batch_id": batch_id},
    )

    code = batch_cli_main(
        ["prepare-data", "--config", str(config_path), "--batch-id", "b1"]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out)["batch_id"] == "b1"
