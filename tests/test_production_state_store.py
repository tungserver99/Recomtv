from datetime import datetime
import sqlite3

import pytest

from production.state_store import StateStore


def test_schedule_configuration_persists(tmp_path):
    store = StateStore(tmp_path / "state.db")
    default = {"enabled": True, "run_times": ["00:00", "12:00"], "timezone": "Asia/Ho_Chi_Minh"}
    assert store.get_schedule(default) == default
    changed = {**default, "enabled": False, "run_times": ["03:45", "18:15"]}
    store.save_schedule(changed)
    assert store.get_schedule(default) == changed


def test_data_settings_persist_and_merge_defaults(tmp_path):
    store = StateStore(tmp_path / "state.db")
    defaults = {
        "time_train": 24,
        "time_valid": 24,
        "time_test": 24,
        "max_profiles": 0,
        "gate_evaluation_profiles": 5000,
    }
    assert store.get_data_settings(defaults) == defaults

    changed = {**defaults, "time_train": 12, "time_valid": 6, "time_test": 3, "max_profiles": 500}
    store.save_data_settings(changed)

    assert store.get_data_settings(defaults) == changed


def test_requests_are_claimed_once_and_jobs_are_globally_serialized(tmp_path):
    store = StateStore(tmp_path / "state.db")
    slot = datetime.fromisoformat("2026-11-13T12:00:00+07:00")
    first = store.enqueue_request(slot)
    assert store.enqueue_request(slot) == first
    request = store.claim_next_request()
    assert request["request_id"] == first
    assert store.claim_next_request() is None
    started = datetime.fromisoformat("2026-11-13T12:01:02+07:00")
    assert store.try_start_job("20261113_1200", slot, started) is True
    assert store.try_start_job("20261113_1200", slot, started) is False
    store.finish_job("20261113_1200", {"inference_status": "SUCCESS"})
    assert store.current_job() is None
    assert store.jobs(limit=1)[0]["status"] == "SUCCESS"


def test_restart_marks_only_dead_process_jobs_interrupted(tmp_path):
    store = StateStore(tmp_path / "state.db")
    slot = datetime.fromisoformat("2026-11-13T12:00:00+07:00")
    store.try_start_job("20261113_1200", slot, slot)
    assert store.recover_interrupted(owner_check=lambda *_: True) == []
    assert store.jobs(limit=1)[0]["status"] == "RUNNING"
    store.recover_interrupted(owner_check=lambda *_: False)
    assert store.jobs(limit=1)[0]["status"] == "INTERRUPTED"
    assert store.current_job() is None


def test_only_one_manual_request_can_wait_in_the_queue(tmp_path):
    store = StateStore(tmp_path / "state.db")
    first = datetime.fromisoformat("2026-11-13T12:00:00+07:00")
    second = datetime.fromisoformat("2026-11-14T00:00:00+07:00")
    request_id = store.enqueue_request(first)
    assert store.enqueue_request(second) == request_id
    assert store.claim_next_request()["batch_id"] == "20261113_1200"


def test_prepare_and_production_requests_keep_distinct_actions_for_one_slot(tmp_path):
    store = StateStore(tmp_path / "state.db")
    slot = datetime.fromisoformat("2026-11-13T12:00:00+07:00")

    prepare_id = store.enqueue_request(slot, kind="prepare")
    prepare = store.claim_next_request()
    assert prepare["request_id"] == prepare_id
    assert prepare["kind"] == "prepare"
    store.finish_request(prepare_id)

    run_id = store.enqueue_request(slot, kind="production")
    run = store.claim_next_request()
    assert run["request_id"] == run_id
    assert run["kind"] == "production"
    assert run_id != prepare_id


def test_independent_action_request_preserves_batch_and_manual_cutoff(tmp_path):
    store = StateStore(tmp_path / "state.db")
    slot = datetime.fromisoformat("2025-02-16T00:00:00+07:00")

    request_id = store.enqueue_request(
        slot,
        kind="inference",
        batch_id="data",
        data_end_at="2025-02-16T23:59:59+07:00",
    )
    request = store.claim_next_request()

    assert request["request_id"] == request_id
    assert request["kind"] == "inference"
    assert request["batch_id"] == "data"
    assert request["data_end_at"] == "2025-02-16T23:59:59+07:00"


def test_queued_request_of_another_kind_is_not_returned_as_this_action(tmp_path):
    store = StateStore(tmp_path / "state.db")
    first = datetime.fromisoformat("2026-11-13T12:00:00+07:00")
    second = datetime.fromisoformat("2026-11-14T00:00:00+07:00")
    store.enqueue_request(first, kind="prepare")

    with pytest.raises(ValueError, match="another dashboard request"):
        store.enqueue_request(second, kind="production")


def test_history_records_skipped_slot_without_preventing_manual_retry(tmp_path):
    store = StateStore(tmp_path / "state.db")
    slot = datetime.fromisoformat("2026-11-13T12:00:00+07:00")
    store.record_skipped("20261113_1200", slot, slot, "another production job is active")
    assert store.jobs(limit=1)[0]["status"] == "SKIPPED_OVERLAP"
    assert store.try_start_job("20261113_1200", slot, slot) is True


def test_old_sqlite_jobs_migrate_data_cutoff_as_legacy_batch_started_at(tmp_path):
    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE jobs (
            batch_id TEXT PRIMARY KEY, scheduled_at TEXT NOT NULL, data_cutoff TEXT NOT NULL,
            status TEXT NOT NULL, actual_start TEXT, actual_end TEXT, attempt INTEGER NOT NULL DEFAULT 0,
            report_json TEXT NOT NULL DEFAULT '{}', error_code TEXT, error_message TEXT, updated_at TEXT NOT NULL
        );
        INSERT INTO jobs(batch_id,scheduled_at,data_cutoff,status,attempt,report_json,updated_at)
        VALUES('old','2025-03-01T12:00:00+00:00','2025-03-01T08:00:00+00:00','SUCCESS',1,
               '{"data_cutoff":"2025-03-01T08:00:00+00:00"}','2025-03-01T12:00:00+00:00');
        """
    )
    connection.close()

    store = StateStore(path)
    job = store.get_job("old")

    assert job["batch_started_at"] == "2025-03-01T08:00:00+00:00"
    assert job["report"]["batch_started_at"] == "2025-03-01T08:00:00+00:00"
    with sqlite3.connect(path) as migrated:
        columns = {row[1] for row in migrated.execute("PRAGMA table_info(jobs)")}
    assert "data_cutoff" not in columns


def test_retry_keeps_the_database_batch_started_at(tmp_path):
    store = StateStore(tmp_path / "state.db")
    slot = datetime.fromisoformat("2026-11-13T12:00:00+07:00")
    started = datetime.fromisoformat("2026-11-13T12:03:00+07:00")
    store.try_start_job("20261113_1200", slot, started)
    store.finish_job("20261113_1200", {"status": "FAILED"}, status="FAILED")

    assert store.try_start_job("20261113_1200", slot, started.replace(minute=9))

    assert store.get_job("20261113_1200")["batch_started_at"] == started.isoformat()
