from datetime import datetime
import json
import socket
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from production.schedule import apply_schedule, next_scheduled_slot
from production.scheduler import _acquire_scheduler_lock, _run_manual
from production.state_store import StateStore
from recsys_vnpt.batch_contracts import BatchError


VN = ZoneInfo("Asia/Ho_Chi_Minh")


class FakeScheduler:
    def __init__(self):
        self.jobs = {}

    def get_jobs(self):
        return [type("Job", (), {"id": key})() for key in self.jobs]

    def add_job(self, func, trigger, **kwargs):
        self.jobs[kwargs["id"]] = {"func": func, "trigger": trigger, **kwargs}

    def remove_job(self, job_id):
        self.jobs.pop(job_id, None)


def test_next_scheduled_slot_is_next_future_occurrence():
    now = datetime(2026, 11, 13, 12, 7, tzinfo=VN)
    assert next_scheduled_slot(now, ["00:00", "12:00"]) == datetime(2026, 11, 14, 0, tzinfo=VN)


def test_schedule_changes_replace_jobs_and_disabled_schedule_removes_them():
    scheduler = FakeScheduler()
    first = {"enabled": True, "run_times": ["00:00", "12:00"], "timezone": "Asia/Ho_Chi_Minh"}
    apply_schedule(scheduler, first, lambda _: None)
    assert set(scheduler.jobs) == {"production-slot-0000", "production-slot-1200"}

    changed = {"enabled": True, "run_times": ["03:45"], "timezone": "Asia/Ho_Chi_Minh"}
    apply_schedule(scheduler, changed, lambda _: None)
    assert set(scheduler.jobs) == {"production-slot-0345"}
    apply_schedule(scheduler, {**changed, "enabled": False}, lambda _: None)
    assert scheduler.jobs == {}


def test_scheduler_store_can_enqueue_run_now_request(tmp_path):
    store = StateStore(tmp_path / "state.db")
    slot = datetime(2026, 11, 13, 12, tzinfo=VN)
    request_id = store.enqueue_request(slot)
    request = store.claim_next_request()
    assert request["request_id"] == request_id
    assert request["slot"] == slot.isoformat()


def test_prepare_request_uses_prepare_only_orchestration(tmp_path, monkeypatch):
    import production.scheduler as scheduler

    config = SimpleNamespace(
        schedule_enabled=True,
        run_times=("00:00", "12:00"),
        timezone="Asia/Ho_Chi_Minh",
    )
    store = StateStore(tmp_path / "state.db")
    observed = {}
    monkeypatch.setattr(
        scheduler,
        "run_production",
        lambda config, slot, **kwargs: observed.update(kwargs),
    )

    _run_manual(
        config,
        store,
        {
            "request_id": "r1",
            "slot": datetime(2026, 11, 13, 12, tzinfo=VN).isoformat(),
            "kind": "prepare",
        },
    )

    assert observed["prepare_only"] is True
    assert store.get_schedule({})["timezone"] == "Asia/Ho_Chi_Minh"


def test_scheduler_reclaims_only_a_lock_whose_local_owner_is_confirmed_dead(tmp_path):
    lock_root = tmp_path / "locks"
    stale = lock_root / "production-scheduler"
    stale.mkdir(parents=True)
    (stale / "owner.json").write_text(
        json.dumps({"pid": 12345, "host": socket.gethostname()}), encoding="utf-8"
    )
    lock, reclaimed = _acquire_scheduler_lock(lock_root, owner_check=lambda *_: False)
    assert reclaimed is True
    lock.__exit__(None, None, None)

    active = lock_root / "production-scheduler"
    active.mkdir(parents=True)
    (active / "owner.json").write_text(
        json.dumps({"pid": 12345, "host": socket.gethostname()}), encoding="utf-8"
    )
    try:
        _acquire_scheduler_lock(lock_root, owner_check=lambda *_: True)
    except BatchError as error:
        assert error.code == "BATCH_LOCKED"
    else:
        raise AssertionError("a live scheduler lock must not be reclaimed")
