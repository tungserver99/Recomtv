from __future__ import annotations

import json

import pandas as pd
import pytest

from recsys_vnpt.engagement import (
    LabelContract,
    apply_atomic_duration,
    build_playback_events,
    historical_sequence_features,
    label_engagement_window,
)


def _interaction(
    content_id: str,
    end: str,
    duration: float,
    uuid: str | None = "launch-1",
) -> dict[str, object]:
    return {
        "profile_id": "profile-1",
        "content_id": content_id,
        "create_date": end,
        "duration": duration,
        "uuid": uuid,
    }


def _contract() -> LabelContract:
    return LabelContract(
        domain="all",
        fit_cutoff="2025-01-31 23:59:59",
        duration_groups={
            "movie|series": {
                "duration_lower_seconds": 30.0,
                "duration_upper_seconds": 600.0,
                "watch_cap_seconds": 300.0,
                "content_rows": 1,
                "playback_rows": 1,
            },
            "all|unknown": {
                "duration_lower_seconds": 10.0,
                "duration_upper_seconds": 1000.0,
                "watch_cap_seconds": 200.0,
                "content_rows": 1,
                "playback_rows": 1,
            },
        },
        global_duration_lower_seconds=10.0,
        global_duration_upper_seconds=1000.0,
        global_watch_cap_seconds=200.0,
    )


def test_exact_and_same_end_duplicates_do_not_create_repeat_events() -> None:
    rows = [
        _interaction("content-1", "2025-01-01 00:02:00", 60),
        _interaction("content-1", "2025-01-01 00:02:00", 60),
        _interaction("content-1", "2025-01-01 00:02:00", 90),
    ]

    events, sessions, metrics = build_playback_events(pd.DataFrame(rows))

    assert len(events) == 1
    assert events.iloc[0]["watch_seconds"] == pytest.approx(90.0)
    assert events.iloc[0]["source_event_count"] == 1
    assert len(sessions) == 1
    assert metrics["exact_duplicate_rows_removed"] == 1
    assert metrics["same_end_duplicate_rows_removed"] == 1
    assert metrics["duplicate_leakage"] == 0


def test_overlapping_playback_intervals_are_merged_not_summed() -> None:
    rows = [
        _interaction("content-1", "2025-01-01 00:01:40", 100),
        _interaction("content-1", "2025-01-01 00:02:30", 100),
    ]

    events, _, _ = build_playback_events(pd.DataFrame(rows))

    assert len(events) == 1
    assert events.iloc[0]["source_event_count"] == 2
    assert events.iloc[0]["watch_seconds"] == pytest.approx(150.0)


def test_session_boundary_uses_uuid_and_falls_back_to_profile_gap() -> None:
    rows = [
        _interaction("content-1", "2025-01-01 00:01:00", 30, "launch-1"),
        _interaction("content-2", "2025-01-01 00:10:00", 30, "launch-1"),
        _interaction("content-3", "2025-01-01 00:11:00", 30, "launch-2"),
        _interaction("content-4", "2025-01-01 01:00:00", 30, None),
        _interaction("content-5", "2025-01-01 01:40:00", 30, None),
    ]

    events, sessions, _ = build_playback_events(
        pd.DataFrame(rows), session_gap_minutes=30
    )

    assert len(events) == 5
    assert len(sessions) == 4
    assert sorted(sessions["distinct_contents"].tolist()) == [1, 1, 1, 2]


def test_series_atomic_duration_and_outlier_fallback() -> None:
    items = pd.DataFrame(
        [
            {
                "content_id": "series-1",
                "content_family": "movie",
                "type_id": "series",
                "content_duration": 1200,
                "total_episode": 12,
            },
            {
                "content_id": "invalid-1",
                "content_family": "movie",
                "type_id": "series",
                "content_duration": -1,
                "total_episode": 1,
            },
        ]
    )

    output = apply_atomic_duration(items, _contract()).set_index("content_id")

    assert output.loc["series-1", "atomic_duration_seconds"] == 100.0
    assert not bool(output.loc["series-1", "atomic_duration_fallback"])
    assert output.loc["invalid-1", "atomic_duration_seconds"] == 30.0
    assert bool(output.loc["invalid-1", "atomic_duration_fallback"])


def test_duplicate_rows_do_not_change_graded_gain() -> None:
    row = _interaction("content-1", "2025-01-05 00:02:00", 100)
    items = pd.DataFrame(
        [
            {
                "content_id": "content-1",
                "content_family": "all",
                "type_id": "unknown",
                "content_duration": 200,
            }
        ]
    )
    single, _, _ = build_playback_events(pd.DataFrame([row]))
    duplicated, _, _ = build_playback_events(pd.DataFrame([row, row]))

    single_gain = label_engagement_window(
        single, items, _contract(), "2025-01-01", "2025-01-10"
    )
    duplicate_gain = label_engagement_window(
        duplicated, items, _contract(), "2025-01-01", "2025-01-10"
    )

    assert single_gain["completion_gain"].tolist() == duplicate_gain[
        "completion_gain"
    ].tolist()
    assert single_gain["watch_time_gain"].tolist() == duplicate_gain[
        "watch_time_gain"
    ].tolist()
    assert single_gain.iloc[0]["episode_count"] == 1


def test_historical_sequence_never_reads_after_cutoff() -> None:
    events, _, _ = build_playback_events(
        pd.DataFrame(
            [
                _interaction("before", "2025-01-01 00:01:00", 20),
                _interaction("after", "2025-02-01 00:01:00", 20),
            ]
        )
    )

    features = historical_sequence_features(events, "2025-01-31 23:59:59")

    assert features.iloc[0]["historical_content_sequence"] == "before"
    assert features.iloc[0]["historical_playback_count"] == 1


def test_label_contract_detects_tampering(tmp_path) -> None:
    path = _contract().save(tmp_path / "label_contract_v4.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["global_watch_cap_seconds"] = 999.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        LabelContract.load(path)


def test_timestamp_resolution_does_not_change_watch_time() -> None:
    # Parquet inputs arrive as datetime64[us]; CSV inputs parse to nanoseconds.
    rows = [
        _interaction("content-1", "2025-01-01 00:02:00", 60),
        _interaction("content-1", "2025-01-01 00:03:30", 90),
        _interaction("content-2", "2025-01-01 01:00:00", 120),
    ]
    from_text = pd.DataFrame(rows)
    from_parquet = from_text.assign(
        create_date=pd.to_datetime(from_text["create_date"]).astype("datetime64[us]")
    )

    expected, expected_sessions, _ = build_playback_events(from_text)
    actual, actual_sessions, _ = build_playback_events(from_parquet)

    assert actual["watch_seconds"].tolist() == expected["watch_seconds"].tolist()
    assert actual["watch_seconds"].tolist() == [150.0, 120.0]
    assert actual["session_id"].tolist() == expected["session_id"].tolist()
    assert len(actual_sessions) == len(expected_sessions)
