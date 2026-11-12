from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from production.cutoff import batch_id_for_slot, resolve_slot, validate_run_times


VN = ZoneInfo("Asia/Ho_Chi_Minh")


def test_slot_id_and_resolution_are_stable():
    slot = datetime(2026, 11, 13, 12, tzinfo=VN)
    assert batch_id_for_slot(slot) == "20261113_1200"
    assert resolve_slot(slot.isoformat(), "Asia/Ho_Chi_Minh", ["00:00", "12:00"]) == slot


def test_implicit_slot_uses_nearest_allowed_slot_not_later_than_now():
    now = datetime(2026, 11, 13, 12, 7, tzinfo=VN)
    assert resolve_slot(None, "Asia/Ho_Chi_Minh", ["00:00", "12:00"], now=now) == datetime(
        2026, 11, 13, 12, tzinfo=VN
    )


@pytest.mark.parametrize("times", [["24:00"], ["12:0"], ["12:00", "12:00"], []])
def test_invalid_run_times_are_rejected(times):
    with pytest.raises(ValueError):
        validate_run_times(times)


def test_explicit_slot_must_match_configured_local_schedule():
    with pytest.raises(ValueError, match="does not match"):
        resolve_slot("2026-11-13T13:00:00+07:00", "Asia/Ho_Chi_Minh", ["00:00", "12:00"])


def test_timezone_offset_is_converted_to_configured_timezone():
    slot = resolve_slot("2026-11-13T05:00:00Z", "Asia/Ho_Chi_Minh", ["12:00"])
    assert slot == datetime(2026, 11, 13, 12, tzinfo=VN)
