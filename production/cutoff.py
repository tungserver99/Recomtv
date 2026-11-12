from __future__ import annotations

import re
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


_TIME = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")


def validate_run_times(run_times: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if not run_times:
        raise ValueError("at least one run time is required")
    normalized = []
    for value in run_times:
        if not isinstance(value, str) or not _TIME.fullmatch(value):
            raise ValueError(f"invalid run time {value!r}; expected HH:MM")
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ValueError("run times must be unique")
    return tuple(sorted(normalized))


def _zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise ValueError(f"unknown timezone {name!r}") from error


def resolve_slot(
    slot: str | datetime | None,
    timezone: str,
    run_times: list[str] | tuple[str, ...],
    now: datetime | None = None,
) -> datetime:
    zone = _zone(timezone)
    times = validate_run_times(run_times)
    if slot is not None:
        parsed = datetime.fromisoformat(slot[:-1] + "+00:00" if slot.endswith("Z") else slot) if isinstance(slot, str) else slot
        if parsed.tzinfo is None:
            raise ValueError("explicit slot must include a timezone offset")
        localized = parsed.astimezone(zone).replace(second=0, microsecond=0)
        if parsed.second or parsed.microsecond or localized.strftime("%H:%M") not in times:
            raise ValueError(f"slot {parsed.isoformat()} does not match a configured run time")
        return localized

    current = now or datetime.now(zone)
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    current = current.astimezone(zone)
    candidates = []
    for day_delta in (0, -1, -2):
        day = current.date() + timedelta(days=day_delta)
        for value in times:
            hour, minute = map(int, value.split(":"))
            candidate = datetime.combine(day, time(hour, minute), tzinfo=zone)
            if candidate <= current:
                candidates.append(candidate)
    if not candidates:
        raise ValueError("could not resolve a scheduled slot")
    return max(candidates)


def batch_id_for_slot(slot: datetime) -> str:
    if slot.tzinfo is None:
        raise ValueError("slot must be timezone-aware")
    return slot.strftime("%Y%m%d_%H%M")



