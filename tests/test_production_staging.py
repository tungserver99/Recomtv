import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from production.config import load_production_config
from production.staging import build_input_snapshot
from recsys_vnpt.batch_contracts import BatchError, load_and_validate_ready


VN = ZoneInfo("Asia/Ho_Chi_Minh")


def _config(tmp_path: Path, duration_format="csv", include_behavior=False):
    source = tmp_path / "source"
    for name in ("mytv_vmp_content", "tags", "content_tags"):
        (source / name).mkdir(parents=True, exist_ok=True)
        (source / name / "source.json").write_text("{}", encoding="utf-8")
    (source / "duration").mkdir()
    events = pd.DataFrame(
        {
            "profile_id": ["p1", "p2", "p3"],
            "username": ["u1", "u2", "u3"],
            "content_id": ["c1", "c2", "c3"],
            "duration": [10, 20, 30],
            "create_date": ["2026-11-13 07:59:59", "2026-11-13 08:00:00", "2026-11-13 08:00:01"],
            "date_time": ["2026-11-13 07:59:49", "2026-11-13 07:59:50", "2026-11-13 07:59:51"],
        }
    )
    if duration_format == "csv":
        events.to_csv(source / "duration" / "part.csv", index=False)
    else:
        events.to_parquet(source / "duration" / "part.parquet", index=False)
    (source / "month_mytv_info.parquet").write_bytes(b"test metadata placeholder")
    if include_behavior:
        behavior = source / "behavior"
        behavior.mkdir()
        pd.DataFrame(
            {
                "profile_id": ["p1", "p2", "p3"],
                "event_time": ["2026-11-13 07:59:59", "2026-11-13 08:00:00", "2026-11-13 08:00:01"],
                "action": ["play", "play", "play"],
            }
        ).to_csv(behavior / "events.csv", index=False)
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
            "include_behavior": include_behavior,
            "input_wait_seconds": 0,
            "min_free_bytes": 0,
            "num_shards": 1,
            "workers": 1,
            "gate_evaluation_profiles": 500,
            "max_profiles": 1000,
            "gbdt_device": "cuda",
            "fcn_epochs": 2,
            "fcn_schedule_epochs": 30,
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return load_production_config(path)


def test_csv_staging_copies_all_events_without_using_batch_start_as_a_filter(tmp_path):
    config = _config(tmp_path)
    source_before = (config.source_root / "duration" / "part.csv").read_bytes()
    started = datetime(2026, 11, 13, 12, tzinfo=VN)
    result = build_input_snapshot(config, "20261113_1200", started)

    output = pd.read_csv(result.batch_dir / "duration" / "part.csv")
    assert output["profile_id"].tolist() == ["p1", "p2", "p3"]
    manifest = load_and_validate_ready(result.batch_dir, "20261113_1200")
    assert manifest.batch_started_at == started
    assert (config.source_root / "duration" / "part.csv").read_bytes() == source_before


def test_csv_staging_preserves_identifier_zeros(tmp_path):
    config = _config(tmp_path)
    source_file = config.source_root / "duration" / "part.csv"
    frame = pd.read_csv(source_file, dtype={"profile_id": "string"})
    frame.loc[0, "profile_id"] = "000123"
    frame.to_csv(source_file, index=False)
    result = build_input_snapshot(
        config, "20261113_1200", datetime(2026, 11, 13, 12, tzinfo=VN)
    )
    output = pd.read_csv(result.batch_dir / "duration" / "part.csv", dtype={"profile_id": "string"})
    assert output.loc[0, "profile_id"] == "000123"


def test_parquet_duration_staging_preserves_all_rows(tmp_path):
    pytest.importorskip("pyarrow")
    config = _config(tmp_path, duration_format="parquet")
    result = build_input_snapshot(
        config, "20261113_1200", datetime(2026, 11, 13, 12, tzinfo=VN)
    )
    output = pd.read_parquet(result.batch_dir / "duration" / "part.parquet")
    assert output["profile_id"].tolist() == ["p1", "p2", "p3"]


def test_enabled_behavior_staging_preserves_all_rows(tmp_path):
    config = _config(tmp_path, include_behavior=True)
    result = build_input_snapshot(
        config, "20261113_1200", datetime(2026, 11, 13, 12, tzinfo=VN)
    )
    output = pd.read_csv(result.batch_dir / "behavior" / "events.csv", dtype={"profile_id": "string"})
    assert output["profile_id"].tolist() == ["p1", "p2", "p3"]


def test_staging_keeps_day_only_events_for_the_temporal_builder_to_reject(tmp_path):
    config = _config(tmp_path)
    source_file = config.source_root / "duration" / "part.csv"
    frame = pd.read_csv(source_file)
    frame["create_date"] = ["2026-11-13"] * len(frame)
    frame.to_csv(source_file, index=False)
    result = build_input_snapshot(config, "20261113_1200", datetime(2026, 11, 13, 12, tzinfo=VN))
    assert len(pd.read_csv(result.batch_dir / "duration" / "part.csv")) == 3


def test_missing_event_timestamp_fails_closed(tmp_path):
    config = _config(tmp_path)
    source_file = config.source_root / "duration" / "part.csv"
    frame = pd.read_csv(source_file).drop(columns=["create_date"])
    frame.to_csv(source_file, index=False)
    with pytest.raises(BatchError, match="SCHEMA_DRIFT.*create_date"):
        build_input_snapshot(config, "20261113_1200", datetime(2026, 11, 13, 8, tzinfo=VN))


def test_duration_schema_is_checked_before_ready_manifest_is_committed(tmp_path):
    config = _config(tmp_path)
    source_file = config.source_root / "duration" / "part.csv"
    frame = pd.read_csv(source_file).drop(columns=["duration"])
    frame.to_csv(source_file, index=False)
    with pytest.raises(BatchError, match="SCHEMA_DRIFT"):
        build_input_snapshot(config, "20261113_1200", datetime(2026, 11, 13, 8, tzinfo=VN))
    assert not (config.incoming_root / "20261113_1200" / "_READY.json").exists()


def test_committed_snapshot_is_reused_and_source_change_conflicts(tmp_path):
    config = _config(tmp_path)
    started = datetime(2026, 11, 13, 12, tzinfo=VN)
    first = build_input_snapshot(config, "20261113_1200", started)
    second = build_input_snapshot(config, "20261113_1200", started)
    assert second.reused is True
    assert second.fingerprint == first.fingerprint

    with (config.source_root / "tags" / "source.json").open("w", encoding="utf-8") as handle:
        handle.write('{"changed": true}')
    with pytest.raises(BatchError, match="source snapshot changed"):
        build_input_snapshot(config, "20261113_1200", started)


def test_production_config_forces_gxep_and_resolves_paths(tmp_path):
    config = _config(tmp_path)
    assert config.batch.ordering.rule_ordering == "learned_rrf"
    assert config.batch.ordering.content_ordering == "ensemble"
    assert Path(config.batch.incoming_root).is_absolute()
    assert config.batch.max_profiles == 1000
    assert config.state_root == tmp_path / "state"
