from __future__ import annotations

import argparse
import logging
import json
import shutil
import socket
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from production.config import load_production_config
from production.orchestrator import run_batch_action, run_production
from production.schedule import apply_schedule
from production.state_store import StateStore
from recsys_vnpt.batch_state import BatchLock
from recsys_vnpt.batch_contracts import BatchError


def _acquire_scheduler_lock(locks_root: Path, owner_check=None):
    from production.state_store import _process_is_alive

    owner_check = owner_check or _process_is_alive
    lock_path = Path(locks_root) / "production-scheduler"
    lock = BatchLock(locks_root, "production-scheduler")
    try:
        lock.__enter__()
        return lock, False
    except BatchError as error:
        owner_path = lock_path / "owner.json"
        try:
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
            pid, host = int(owner["pid"]), str(owner["host"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            raise error
        if host.casefold() != socket.gethostname().casefold() or owner_check(pid, host):
            raise error
        shutil.rmtree(lock_path)
        lock.__enter__()
        return lock, True


LOG = logging.getLogger("production.scheduler")


def _default_schedule(config) -> dict:
    inference_times = getattr(config, "inference_run_times", config.run_times)
    return {
        "enabled": config.schedule_enabled,
        "run_times": list(inference_times),
        "inference_run_times": list(inference_times),
        "retrain_run_times": list(getattr(config, "retrain_run_times", ())),
        "timezone": config.timezone,
    }


def _run_manual(config, store, request):
    from production.cutoff import resolve_slot

    try:
        slot = resolve_slot(request["slot"], store.get_schedule(_default_schedule(config))["timezone"],
                           store.get_schedule(_default_schedule(config))["run_times"])
        kind = request.get("kind", "production")
        if kind in {"retrain", "inference"}:
            run_batch_action(
                config,
                request["batch_id"],
                kind,
                store=store,
                scheduled_at=slot,
                data_end_at=request.get("data_end_at"),
                run_id=request["request_id"],
            )
        else:
            result = run_production(config, slot, store=store, prepare_only=kind == "prepare")
            if kind == "prepare":
                result.update({"kind": "prepare", "run_id": request["request_id"], "mode": "Automatic"})
                store.update_job(result.get("batch_id", request["batch_id"]), result, result.get("status"))
        store.finish_request(request["request_id"])
    except Exception as error:
        LOG.exception("manual request %s failed", request["request_id"])
        store.finish_request(request["request_id"], str(error))


def run_scheduler(config_path: str | Path | None = None, poll_seconds: float = 2.0) -> None:
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except ImportError as error:
        raise RuntimeError("APScheduler is required; install the pinned production dependencies") from error

    config_path = Path(config_path or Path(__file__).with_name("config.json")).resolve()
    config = load_production_config(config_path)
    store = StateStore(config.database_path)
    lock, stale_scheduler = _acquire_scheduler_lock(config.work_root / "locks")
    try:
        store.recover_interrupted(scheduler_interrupted=stale_scheduler)
        _serve(config, store, BackgroundScheduler, poll_seconds)
    finally:
        lock.__exit__(None, None, None)


def _serve(config, store, scheduler_factory, poll_seconds: float) -> None:
    schedule = store.get_schedule(_default_schedule(config))
    scheduler = scheduler_factory(timezone=ZoneInfo(schedule["timezone"]))

    def dispatch(kind: str, run_time: str) -> None:
        current = store.get_schedule(_default_schedule(config))
        times = current.get(f"{kind}_run_times", current.get("run_times", []))
        if not current.get("enabled") or run_time not in times:
            return
        zone = ZoneInfo(current["timezone"])
        now = datetime.now(zone)
        hour, minute = map(int, run_time.split(":"))
        slot = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if slot > now:
            return
        try:
            prepared = next((
                item for item in store.jobs(limit=500)
                if item["status"] == "PREPARED"
                and item["report"].get("preparation_status") == "READY"
            ), None)
            if prepared is None:
                LOG.warning("scheduled %s skipped: no prepared dataset", kind)
                return
            run_batch_action(
                config, prepared["report"].get("batch_id", prepared["batch_id"]), kind,
                store=store, scheduled_at=slot,
            )
        except Exception:
            LOG.exception("scheduled %s slot %s failed", kind, slot.isoformat())

    from production.schedule import apply_action_schedules

    apply_action_schedules(scheduler, schedule, dispatch)
    scheduler.start()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="production-manual")
    manual_future: Future | None = None
    last_schedule = schedule
    LOG.info("production scheduler started with %s", schedule)
    try:
        while True:
            current_schedule = store.get_schedule(_default_schedule(config))
            if current_schedule != last_schedule:
                apply_action_schedules(scheduler, current_schedule, dispatch)
                last_schedule = current_schedule
                LOG.info("schedule updated: %s", current_schedule)
            if manual_future is None or manual_future.done():
                if manual_future is not None:
                    try:
                        manual_future.result()
                    except Exception:
                        LOG.exception("manual run worker failed")
                    manual_future = None
                request = store.claim_next_request() if store.current_job() is None else None
                if request:
                    manual_future = executor.submit(_run_manual, config, store, request)
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        LOG.info("scheduler shutdown requested")
    finally:
        scheduler.shutdown(wait=True)
        executor.shutdown(wait=True, cancel_futures=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the RecSys-VNPT production scheduler")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_scheduler(args.config, args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

