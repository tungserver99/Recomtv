import json
import pickle
import sys
from datetime import datetime, timezone

import pytest

from recsys_vnpt.batch_contracts import (
    BatchConfig,
    BatchError,
    load_and_validate_ready,
    load_config,
    wait_for_ready,
    write_ready_manifest,
)
from recsys_vnpt.batch_state import BatchLock, BatchStatus

BATCH_STARTED = datetime(2025, 3, 1, 8, 30, tzinfo=timezone.utc)


def _batch(tmp_path, name="b1", batch_started_at=BATCH_STARTED):
    batch = tmp_path / name
    for relative in (
        "duration/part.csv",
        "mytv_vmp_content/20250101/content_movie.json",
        "tags/20250101/TAG.json",
        "content_tags/20250101/CONTENT_TO_TAG.json",
        "month_mytv_info.parquet",
    ):
        path = batch / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    write_ready_manifest(batch, batch_started_at=batch_started_at)
    return batch


def _rewrite_ready(batch, mutate):
    ready = json.loads((batch / "_READY.json").read_text())
    mutate(ready)
    (batch / "_READY.json").write_text(json.dumps(ready))


def test_ready_manifest_round_trips_with_a_content_fingerprint(tmp_path):
    first = load_and_validate_ready(_batch(tmp_path / "x", "b1"), "b1")
    second = load_and_validate_ready(_batch(tmp_path / "y", "b2"), "b2")
    assert first.batch_started_at == BATCH_STARTED
    assert len(first.files) == 5
    # Same bytes under another batch id is the same input.
    assert first.fingerprint == second.fingerprint


@pytest.mark.parametrize(
    "tamper, message",
    [
        (lambda b: (b / "duration/part.csv").write_text("duration/part.csX"), "checksum"),
        (lambda b: (b / "duration/part.csv").write_text("x"), "size"),
        (lambda b: (b / "duration/extra.csv").write_text("x"), "not in the manifest"),
        (lambda b: _rewrite_ready(b, lambda r: r["files"][0].update(path="../escape")), "unsafe"),
        (lambda b: _rewrite_ready(b, lambda r: r["files"][0].update(path="/abs")), "unsafe"),
        (lambda b: _rewrite_ready(b, lambda r: r["files"].append(dict(r["files"][0]))), "duplicate"),
        (lambda b: _rewrite_ready(b, lambda r: r.update(batch_started_at="2025-03-01T00:00:00")), "timezone"),
        (lambda b: _rewrite_ready(b, lambda r: r.update(batch_id="other")), "batch id"),
        (
            lambda b: _rewrite_ready(
                b, lambda r: r.update(files=[f for f in r["files"] if not f["path"].startswith("tags/")])
            )
            or (b / "tags/20250101/TAG.json").unlink(),
            "required sources",
        ),
    ],
)
def test_ready_manifest_rejects_anything_it_cannot_vouch_for(tmp_path, tamper, message):
    batch = _batch(tmp_path)
    tamper(batch)
    with pytest.raises(BatchError, match=message) as error:
        load_and_validate_ready(batch, "b1")
    assert error.value.exit_code == 11


@pytest.mark.skipif(sys.platform == "win32", reason="requires symlink creation")
def test_ready_manifest_rejects_a_symlink_out_of_the_batch(tmp_path):
    batch = _batch(tmp_path)
    outside = tmp_path / "outside.csv"
    outside.write_text("secret")
    (batch / "duration" / "link.csv").symlink_to(outside)
    with pytest.raises(BatchError, match="not in the manifest"):
        write_ready_manifest(batch, batch_started_at=BATCH_STARTED)


def test_ready_manifest_uses_one_stable_start_and_separate_manifest_completion_time(tmp_path):
    batch = _batch(tmp_path)
    first = load_and_validate_ready(batch, "b1")
    ready_before = (batch / "_READY.json").read_bytes()

    write_ready_manifest(batch)

    second = load_and_validate_ready(batch, "b1")
    assert second.batch_started_at == BATCH_STARTED
    assert second.created_at == first.created_at
    assert (batch / "_READY.json").read_bytes() == ready_before
    with pytest.raises(BatchError, match="conflicts"):
        write_ready_manifest(batch, batch_started_at=datetime(2025, 3, 4, 8, 30, tzinfo=timezone.utc))


def test_ready_manifest_generates_timezone_aware_start_when_not_supplied(tmp_path):
    batch = _batch(tmp_path, "auto", batch_started_at=None)
    manifest = load_and_validate_ready(batch, "auto")
    assert manifest.batch_started_at.tzinfo is not None
    assert manifest.batch_started_at.utcoffset() is not None


def test_legacy_ready_manifest_maps_data_cutoff_to_batch_start_without_rewriting(tmp_path):
    batch = _batch(tmp_path)
    raw = json.loads((batch / "_READY.json").read_text())
    raw["schema_version"] = 1
    raw["data_cutoff"] = raw.pop("batch_started_at")
    (batch / "_READY.json").write_text(json.dumps(raw))

    manifest = load_and_validate_ready(batch, "b1")

    assert manifest.batch_started_at == BATCH_STARTED
    assert json.loads((batch / "_READY.json").read_text())["schema_version"] == 1


def test_batch_timestamp_reader_prefers_new_field_and_falls_back_to_legacy_data_cutoff():
    from recsys_vnpt.batch_contracts import batch_started_at_from_metadata

    legacy = {"data_cutoff": "2025-03-01T08:30:00+00:00"}
    current = {
        "batch_started_at": "2025-03-02T08:30:00+00:00",
        "data_cutoff": "2025-03-01T08:30:00+00:00",
    }

    assert batch_started_at_from_metadata(legacy) == BATCH_STARTED
    assert batch_started_at_from_metadata(current) == datetime(2025, 3, 2, 8, 30, tzinfo=timezone.utc)


def test_waiting_for_input_times_out_without_sleeping_for_real(tmp_path):
    clock = iter(range(0, 1000, 10))
    with pytest.raises(BatchError, match="INPUT_TIMEOUT") as error:
        wait_for_ready(tmp_path, timeout_seconds=30, poll_seconds=10, clock=lambda: next(clock), sleep=lambda _: None)
    assert error.value.exit_code == 10


def test_config_rejects_unknown_keys_and_impossible_pages(tmp_path):
    roots = {name: str(tmp_path) for name in ("incoming_root", "work_root", "delivery_root", "model_registry")}
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**roots, "num_shard": 4}))
    with pytest.raises(BatchError, match="unknown config keys"):
        load_config(path)
    with pytest.raises(BatchError, match="contents_per_rule_max"):
        BatchConfig(**roots, contents_per_rule_max=40)
    with pytest.raises(BatchError, match="CONFIG_INVALID"):
        load_config(path.with_name("missing.json"))


def test_batch_errors_survive_a_process_boundary():
    error = pickle.loads(pickle.dumps(BatchError("OUTPUT_INVALID", "shard 3")))
    assert (error.code, error.message, error.exit_code) == ("OUTPUT_INVALID", "shard 3", 40)


def test_status_moves_forward_only_and_records_the_failure(tmp_path):
    status = BatchStatus(tmp_path / "status.json", "b1", "inference")
    status.begin()
    status.transition("VALIDATING_INPUT")
    with pytest.raises(RuntimeError, match="illegal transition"):
        status.transition("WAITING_INPUT")
    status.fail(BatchError("INPUT_INTEGRITY", "bad hash"))
    document = status.read()
    assert document["phase"] == "FAILED"
    assert document["details"]["error_stage"] == "VALIDATING_INPUT"
    assert (tmp_path / "log.jsonl").read_text().count("\n") == 3


def test_a_held_lock_refuses_a_second_owner(tmp_path):
    with BatchLock(tmp_path, "inference-b1"):
        with pytest.raises(BatchError, match="BATCH_LOCKED"):
            with BatchLock(tmp_path, "inference-b1"):
                pass
    with BatchLock(tmp_path, "inference-b1"):
        pass
