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
from production.orchestrator import run_production
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
    return {
        "enabled": config.schedule_enabled,
        "run_times": list(config.run_times),
        "timezone": config.timezone,
    }


def _run_manual(config, store, request):
    from production.cutoff import resolve_slot

    try:
        slot = resolve_slot(request["slot"], store.get_schedule(_default_schedule(config))["timezone"],
                           store.get_schedule(_default_schedule(config))["run_times"])
        run_production(
            config,
            slot,
            store=store,
            prepare_only=request.get("kind", "production") == "prepare",
        )
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

    def dispatch(run_time: str) -> None:
        current = store.get_schedule(_default_schedule(config))
        if not current.get("enabled") or run_time not in current["run_times"]:
            return
        zone = ZoneInfo(current["timezone"])
        now = datetime.now(zone)
        hour, minute = map(int, run_time.split(":"))
        slot = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if slot > now:
            return
        try:
            run_production(config, slot, store=store)
        except Exception:
            LOG.exception("scheduled slot %s failed", slot.isoformat())

    apply_schedule(scheduler, schedule, dispatch)
    scheduler.start()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="production-manual")
    manual_future: Future | None = None
    last_schedule = schedule
    LOG.info("production scheduler started with %s", schedule)
    try:
        while True:
            current_schedule = store.get_schedule(_default_schedule(config))
            if current_schedule != last_schedule:
                apply_schedule(scheduler, current_schedule, dispatch)
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

