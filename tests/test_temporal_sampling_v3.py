import pandas as pd
import pytest

from recsys_vnpt.canonical import (
    read_partitioned_events,
    select_content_metadata_as_of,
    write_partitioned_interactions,
)
from recsys_vnpt.temporal import (
    DEFAULT_TEMPORAL_WINDOWS,
    derive_relative_temporal_windows,
    load_temporal_windows,
    write_temporal_windows,
    exclude_consumed_pairs,
    future_positive_pairs,
    mixed_negative_samples,
    observed_negative_pairs,
    select_smoke_profile_cohort,
    split_aware_catalog_cap,
)


def test_temporal_windows_include_subsecond_events_on_the_final_day():
    window = DEFAULT_TEMPORAL_WINDOWS[-1]
    interactions = pd.DataFrame(
        {
            "profile_id": ["p1", "p1"],
            "content_id": ["included", "excluded"],
            "create_date": [
                "2025-02-20 23:59:59.500000",
                "2025-02-21 00:00:00",
            ],
            "label": [1, 1],
        }
    )
    positives = future_positive_pairs(interactions, window)
    assert positives["content_id"].tolist() == ["included"]


def test_consumed_pairs_are_removed_from_future_training_truth():
    future = pd.DataFrame(
        {
            "profile_id": ["p1", "p1", "p2"],
            "content_id": ["seen", "new", "other"],
            "label": [1, 1, 0],
        }
    )
    consumed = pd.DataFrame(
        {"profile_id": ["p1"], "content_id": ["seen"]}
    )
    filtered = exclude_consumed_pairs(future, consumed)
    assert set(filtered["content_id"]) == {"new", "other"}


def test_mixed_negatives_exclude_consumed_and_use_sample_weights():
    positives = pd.DataFrame({"profile_id": ["p1"], "content_id": ["c1"]})
    catalog = pd.DataFrame(
        {
            "content_id": [f"c{i}" for i in range(1, 9)],
            "popularity": list(range(8)),
        }
    )
    membership = pd.DataFrame(
        {
            "content_id": ["c1", "c2", "c3", "c4"],
            "rule_id": ["r1", "r1", "r1", "r2"],
        }
    )
    consumed = pd.DataFrame({"profile_id": ["p1"], "content_id": ["c8"]})
    sampled = mixed_negative_samples(
        positives, catalog, consumed, membership, ratio=4, seed=42
    )
    repeated = mixed_negative_samples(
        positives, catalog, consumed, membership, ratio=4, seed=42
    )
    assert 1 <= len(sampled) <= 4
    assert not set(sampled["content_id"]) & {"c1", "c8"}
    assert sampled["sample_weight"].eq(0.25).all()
    assert "same_rule_hard_negative" in set(sampled["sample_type"])
    pd.testing.assert_frame_equal(sampled, repeated)


def test_observed_low_watch_is_retained_with_full_weight():
    labeled = pd.DataFrame(
        {
            "profile_id": ["p1", "p1"],
            "content_id": ["c1", "c2"],
            "label": [1, 0],
        }
    )
    negatives = observed_negative_pairs(labeled)
    assert negatives.to_dict("records") == [
        {
            "profile_id": "p1",
            "content_id": "c2",
            "label": 0,
            "sample_type": "observed_low_watch",
            "sample_weight": 1.0,
        }
    ]


def test_smoke_cohort_and_content_cap_are_selected_after_labels():
    windows = {
        "train_1": pd.DataFrame(
            {"profile_id": ["train"], "content_id": ["c1"], "label": [1]}
        ),
        "test": pd.DataFrame(
            {"profile_id": ["test"], "content_id": ["c9"], "label": [1]}
        ),
    }
    assert select_smoke_profile_cohort(windows, 1, seed=42) == {"test"}

    items = pd.DataFrame({"content_id": [f"c{i}" for i in range(1, 13)]})
    membership = pd.DataFrame(
        {"content_id": [f"c{i}" for i in range(1, 13)], "rule_id": ["r1"] * 12}
    )
    capped_items, _ = split_aware_catalog_cap(
        items,
        membership,
        [windows["test"]],
        {f"c{i}": i for i in range(1, 13)},
        maximum_contents=10,
    )
    assert "c9" in set(capped_items["content_id"])
    assert len(capped_items) == 10


def test_as_of_metadata_and_date_partitions(tmp_path):
    versions = pd.DataFrame(
        {
            "content_id": ["c1", "c1", "c2"],
            "content_name": ["old", "new", "future-known"],
            "content_status": [1, 0, 1],
            "publish_date": ["2024-01-01", "2024-01-01", "2025-01-01"],
            "_source": ["a", "b", "c"],
            "_source_timestamp": pd.to_datetime(
                ["2025-01-01", "2025-03-01", "2025-03-01"]
            ),
        }
    )
    selected = select_content_metadata_as_of(versions, "2025-02-01")
    old = selected.set_index("content_id").loc["c1"]
    backfill = selected.set_index("content_id").loc["c2"]
    assert old["content_name"] == "old"
    assert not bool(old["metadata_backfill"])
    assert backfill["content_name"] == "future-known"
    assert pd.isna(backfill["content_status"])
    assert bool(backfill["metadata_backfill"])

    interactions = pd.DataFrame(
        {
            "profile_id": ["p1", "p1"],
            "content_id": ["c1", "c2"],
            "create_date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
        }
    )
    paths = write_partitioned_interactions(interactions, tmp_path / "parts")
    assert len(paths) == 2
    assert all(path.exists() for path in paths)


def test_relative_windows_derive_from_data_and_roundtrip(tmp_path):
    import pandas as pd

    windows = derive_relative_temporal_windows(
        pd.Timestamp("2026-11-12 20:00"), time_train=24, time_valid=24, time_test=24
    )
    assert [w.name for w in windows] == [
        "train_1", "train_2", "train_3", "validation", "test",
    ]
    assert [w.split for w in windows] == [
        "train", "train", "train", "validation", "test",
    ]
    assert windows[0].label_start == pd.Timestamp("2026-11-07 20:00")
    assert windows[0].label_end == pd.Timestamp("2026-11-08 19:59:59.999999999")
    assert windows[-1].label_start == pd.Timestamp("2026-11-11 20:00")
    assert windows[-1].label_end == pd.Timestamp("2026-11-12 20:00")
    for earlier, later in zip(windows, windows[1:]):
        # contiguous, non-overlapping hour windows; history ends just before labels
        assert later.label_start == earlier.label_end + pd.Timedelta(1, unit="ns")
        assert later.history_end < later.label_start
    # persisted definitions round-trip; missing file falls back to pinned defaults
    time_config = {"mode": "relative", "time_train": 24, "time_valid": 24, "time_test": 24}
    write_temporal_windows(tmp_path, windows, time_config=time_config)
    assert load_temporal_windows(tmp_path) == windows
    import json
    assert json.loads((tmp_path / "windows.json").read_text())["time_config"] == time_config
    assert load_temporal_windows(tmp_path / "absent") == DEFAULT_TEMPORAL_WINDOWS


def test_partitioned_event_read_loads_only_intersecting_dates_and_exact_times(tmp_path, monkeypatch):
    root = tmp_path / "events"
    for date in ("2025-01-01", "2025-01-02", "2025-01-03"):
        part = root / f"event_date={date}" / "part-00000.parquet"
        part.parent.mkdir(parents=True)
        part.touch()
    data = {
        "2025-01-01": pd.DataFrame({"event_end": ["2025-01-01 23:00"], "id": [1]}),
        "2025-01-02": pd.DataFrame({"event_end": ["2025-01-02 12:00", "2025-01-02 16:00"], "id": [2, 3]}),
    }
    read_paths = []

    def fake_read_parquet(path):
        read_paths.append(Path(path))
        return data[Path(path).parent.name.split("=", 1)[1]]

    from pathlib import Path
    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    result = read_partitioned_events(root, "2025-01-02 00:00", "2025-01-02 12:00")

    assert len(read_paths) == 1
    assert read_paths[0].parent.name == "event_date=2025-01-02"
    assert result["id"].tolist() == [2]


def test_relative_windows_use_unequal_hours_across_month_boundary():
    import pandas as pd

    windows = derive_relative_temporal_windows(
        pd.Timestamp("2027-01-01 01:15"), time_train=12, time_valid=6, time_test=3
    )
    assert [(window.name, window.label_start, window.label_end) for window in windows] == [
        ("train_1", pd.Timestamp("2026-12-30 04:15"), pd.Timestamp("2026-12-30 16:14:59.999999999")),
        ("train_2", pd.Timestamp("2026-12-30 16:15"), pd.Timestamp("2026-12-31 04:14:59.999999999")),
        ("train_3", pd.Timestamp("2026-12-31 04:15"), pd.Timestamp("2026-12-31 16:14:59.999999999")),
        ("validation", pd.Timestamp("2026-12-31 16:15"), pd.Timestamp("2026-12-31 22:14:59.999999999")),
        ("test", pd.Timestamp("2026-12-31 22:15"), pd.Timestamp("2027-01-01 01:15")),
    ]


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_relative_windows_reject_non_positive_integer_hour_values(value):
    import pandas as pd
    import pytest

    with pytest.raises(ValueError, match="positive integer"):
        derive_relative_temporal_windows(pd.Timestamp("2026-11-12 20:00"), time_train=value)


def test_temporal_window_fingerprint_changes_when_hours_change():
    import pandas as pd

    from recsys_vnpt.temporal import temporal_window_fingerprint

    maximum = pd.Timestamp("2026-11-12 20:00")
    daily = derive_relative_temporal_windows(maximum, 24, 24, 24)
    shorter = derive_relative_temporal_windows(maximum, 12, 6, 3)

    assert temporal_window_fingerprint(daily, "relative", 24, 24, 24) != temporal_window_fingerprint(
        shorter, "relative", 12, 6, 3
    )


def test_relative_window_coverage_rejects_a_window_without_positive_samples():
    import pandas as pd
    import pytest

    from recsys_vnpt.temporal import validate_window_positives

    frames = {name: pd.DataFrame({"label": [1]}) for name in ("train_1", "train_2", "train_3", "validation", "test")}
    frames["validation"] = pd.DataFrame({"label": [0, 0]})

    with pytest.raises(ValueError, match="validation.*positive"):
        validate_window_positives(frames)


def test_strict_duration_timestamps_reject_day_only_and_filename_fallback():
    import pytest
    from recsys_vnpt.prepare import _normalize_duration_chunk

    base = {"profile_id": ["p"], "username": ["u"], "content_id": ["c"], "duration": [10]}
    with pytest.raises(ValueError, match="requires precise create_date"):
        _normalize_duration_chunk(pd.DataFrame({**base, "filename_date": ["20261112"]}), True)
    with pytest.raises(ValueError, match="hour and minute precision"):
        _normalize_duration_chunk(pd.DataFrame({**base, "create_date": ["2026-11-12"]}), True)
    normalized = _normalize_duration_chunk(
        pd.DataFrame({**base, "create_date": ["2026-11-12 20:15:00"]}), True
    )
    assert normalized.loc[0, "create_date"] == pd.Timestamp("2026-11-12 20:15:00")


def test_test_split_training_windows_include_validation_but_never_test(tmp_path):
    import pandas as pd

    from recsys_vnpt.rule_scorer import training_windows
    from recsys_vnpt.temporal import derive_relative_temporal_windows, write_temporal_windows

    temporal = tmp_path / "temporal"
    windows = derive_relative_temporal_windows(pd.Timestamp("2026-11-12 20:00"))
    write_temporal_windows(temporal, windows)

    assert [window.name for window in training_windows(tmp_path, "test")] == [
        "train_1", "train_2", "train_3", "validation"
    ]


def test_backfill_keeps_descriptive_attributes_but_not_merchandising_ones():
    """The as-of guard must not delete the content semantics the model needs.

    Every content snapshot in this dataset postdates every window cutoff, so
    every row takes the backfill path. Nulling everything outside the immutable
    set therefore emptied the descriptive attributes on all training rows. Those
    must survive, while attributes that move after a cutoff are still withheld.
    """
    versions = pd.DataFrame(
        {
            "content_id": ["c1"],
            "content_name": ["known"],
            "tag_names": ["hanh dong,vien tuong"],
            "content_category_name": ["phim le"],
            "content_country": ["VN"],
            "total_episode": [12],
            "content_hot": [1],
            "publish_date": ["2024-01-01"],
            "_source": ["a"],
            "_source_timestamp": pd.to_datetime(["2025-03-01"]),
        }
    )
    row = select_content_metadata_as_of(versions, "2025-02-01").iloc[0]
    assert bool(row["metadata_backfill"])
    # Describes what the content is: kept.
    assert row["tag_names"] == "hanh dong,vien tuong"
    assert row["content_category_name"] == "phim le"
    assert row["content_country"] == "VN"
    # Moves after the cutoff: withheld. total_episode grows while a series
    # airs and content_hot is a weekly merchandising flag.
    assert pd.isna(row["total_episode"])
    assert pd.isna(row["content_hot"])

    strict = select_content_metadata_as_of(
        versions, "2025-02-01", include_slow_moving=False
    ).iloc[0]
    assert pd.isna(strict["tag_names"])
    assert pd.isna(strict["content_category_name"])
