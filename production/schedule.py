from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Callable
from zoneinfo import ZoneInfo

from production.cutoff import validate_run_times


def next_scheduled_slot(now: datetime, run_times: list[str] | tuple[str, ...]) -> datetime:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    times = validate_run_times(run_times)
    zone = now.tzinfo
    candidates = []
    local_now = now.astimezone(zone)
    for offset in range(-1, 3):
        day = local_now.date() + timedelta(days=offset)
        for value in times:
            hour, minute = map(int, value.split(":"))
            candidate = datetime.combine(day, time(hour, minute), tzinfo=zone)
            if candidate > local_now:
                candidates.append(candidate)
    return min(candidates)


def apply_schedule(scheduler, schedule: dict, callback: Callable[[str], None]) -> None:
    """Replace only this service's cron jobs; safe to call after each DB poll."""
    desired = {}
    if schedule.get("enabled"):
        timezone = ZoneInfo(schedule["timezone"])
        for value in validate_run_times(schedule["run_times"]):
            hour, minute = map(int, value.split(":"))
            job_id = f"production-slot-{hour:02d}{minute:02d}"
            desired[job_id] = (value, hour, minute, timezone)
    existing = {job.id for job in scheduler.get_jobs() if job.id.startswith("production-slot-")}
    for job_id in existing - set(desired):
        scheduler.remove_job(job_id)
    for job_id, (value, hour, minute, timezone) in desired.items():
        scheduler.add_job(
            callback,
            trigger="cron",
            hour=hour,
            minute=minute,
            timezone=timezone,
            args=[value],
            id=job_id,
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=1,
        )

