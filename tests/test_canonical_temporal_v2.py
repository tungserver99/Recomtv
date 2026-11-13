import pandas as pd

from recsys_vnpt.canonical import (
    ContentSnapshot,
    canonical_profile_mapping,
    canonicalize_behavior_events,
    canonicalize_content_tags,
    canonicalize_contents,
    interaction_partitions_through,
    historical_behavior_features,
    historical_behavior_features_from_parts,
    normalize_identifier,
    normalize_user_metadata,
    normalize_username,
    normalize_data_end_at,
    read_partitioned_interactions_until,
)
from recsys_vnpt.temporal import deterministic_negative_samples, evaluation_candidates
from recsys_vnpt.temporal import temporal_artifact_root
from recsys_vnpt.pipeline_v2 import (
    _evaluation_truth_selection,
    _interaction_catalog_audit,
)


def test_manual_cutoff_requires_timezone_and_preserves_local_wall_time():
    import pytest

    assert normalize_data_end_at("2025-02-15T23:59:59+07:00") == pd.Timestamp(
        "2025-02-15 23:59:59"
    )
    with pytest.raises(ValueError, match="timezone offset"):
        normalize_data_end_at("2025-02-15 23:59:59")


def test_manual_temporal_cache_uses_the_configured_isolated_root(tmp_path, monkeypatch):
    canonical = tmp_path / "canonical"
    isolated = tmp_path / "temporal" / "cutoff-20250215"
    monkeypatch.setenv("RECSYS_VNPT_TEMPORAL_ROOT", str(isolated))

    assert temporal_artifact_root(canonical) == isolated


def test_partition_selection_does_not_open_dates_after_manual_cutoff(tmp_path):
    root = tmp_path / "interactions_by_date"
    for day in ("2025-02-15", "2025-02-16", "2025-02-20"):
        part = root / f"event_date={day}" / "part-00000.parquet"
        part.parent.mkdir(parents=True)
        part.touch()

    selected, available = interaction_partitions_through(
        root, "2025-02-16T10:30:00+07:00"
    )

    assert available == 3
    assert [path.parent.name for path in selected] == [
        "event_date=2025-02-15", "event_date=2025-02-16"
    ]


def test_manual_cutoff_prunes_later_partitions_and_filters_within_day(tmp_path):
    import pytest
    pl = pytest.importorskip("polars")

    parts = tmp_path / "interactions_by_date"
    for date, rows in {
        "2025-02-15": [
            ("p1", "2025-02-15 12:00:00"),
            ("p2", "2025-02-15 23:59:59"),
        ],
        "2025-02-16": [("p3", "2025-02-16 00:00:00")],
    }.items():
        partition = parts / f"event_date={date}"
        partition.mkdir(parents=True)
        pl.DataFrame(
            {
                "profile_id": [row[0] for row in rows],
                "content_id": ["c1"] * len(rows),
                "create_date": [row[1] for row in rows],
                "duration": [10] * len(rows),
            }
        ).with_columns(pl.col("create_date").str.to_datetime()).write_parquet(
            partition / "part-00000.parquet"
        )

    loaded = read_partitioned_interactions_until(
        parts, "2025-02-15T23:59:58+07:00"
    )

    assert loaded["profile_id"].tolist() == ["p1"]
    assert len(list(parts.glob("event_date=*/part-*.parquet"))) == 2


def test_identifier_normalization_removes_only_numeric_float_artifacts():
    values = pd.Series([4294895.0, "50008412.0", "account.0", None])

    assert normalize_identifier(values).tolist()[:3] == [
        "4294895",
        "50008412",
        "account.0",
    ]
    assert normalize_username(pd.Series([" HNITV001 "])).iloc[0] == "hnitv001"


def test_interaction_catalog_audit_separates_scope_and_reports_time_quality(tmp_path):
    interactions = tmp_path / "interactions.parquet"
    interactions.mkdir()
    pd.DataFrame(
        {
            "profile_id": ["p1", "p2", "p3"],
            "content_id": ["eligible", "old", "unknown"],
            "refer_path": ["HOME+poster", "CHANNEL_HOME", None],
            "date_time": [None, None, None],
            "duration": [100.0, 200.0, 25 * 60 * 60.0],
        }
    ).to_parquet(interactions / "part_00000.parquet", index=False)

    metrics = _interaction_catalog_audit(
        interactions,
        {"eligible"},
        {"eligible", "old"},
    )

    by_class = metrics["by_catalog_class"]
    assert by_class["eligible"]["rows"] == 1
    assert by_class["historical_version_only"]["rows"] == 1
    assert by_class["unknown"]["rows"] == 1
    assert by_class["historical_version_only"]["channel_path_hint_rows"] == 1
    assert metrics["eligible_missing_start_time_rate"] == 1.0
    assert metrics["eligible_duration_over_24h_rows"] == 0


def test_evaluation_truth_selection_caps_profiles_without_dropping_their_rows(tmp_path):
    temporal = tmp_path / "temporal"
    temporal.mkdir()
    pd.DataFrame(
        {
            "profile_id": ["p1", "p1", "p2", "p2", "p3", "p3"],
            "content_id": ["a", "x", "b", "y", "c", "z"],
            "label": [1, 0, 1, 0, 1, 0],
        }
    ).to_parquet(temporal / "validation_candidates.parquet", index=False)

    selected, cutoff, metrics = _evaluation_truth_selection(
        temporal, "validation", 2, seed=42
    )

    assert selected["profile_id"].nunique() == 2
    assert len(selected) == 4
    assert cutoff == pd.Timestamp("2025-01-31 23:59:59.999999999")
    assert metrics["evaluation_profiles_selected"] == 2
    assert len(metrics["evaluation_profile_selection_hash"]) == 64


def test_user_content_and_behavior_canonicalization_are_deterministic():
    users = pd.DataFrame(
        {
            "username": [" u1 ", "u2", "u2"],
            "sex": [1, 9, 0],
            "birthday": ["19900101", "bad", "20000101"],
            "package_code": ["A", "B", "C"],
        }
    )
    normalized = normalize_user_metadata(users)
    assert normalized.frame["username"].tolist() == ["u1", "u2"]
    assert normalized.frame.set_index("username").loc["u1", "age"] == 35
    assert normalized.metrics["duplicate_usernames_removed"] == 1

    snapshots = [
        ContentSnapshot(
            "raw/content_movie_20250101_000000.json",
            pd.Timestamp("2025-01-01"),
            pd.DataFrame({"content_id": ["1", "2"], "value": ["old", "keep"]}),
        ),
        ContentSnapshot(
            "raw/content_movie_20250103_000000.json",
            pd.Timestamp("2025-01-03"),
            pd.DataFrame({"content_id": ["1", "3"], "value": ["republished", "new"]}),
        ),
    ]
    tombstones = pd.DataFrame(
        {
            "content_id": ["1", "2"],
            "_source_timestamp": [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-04")],
        }
    )
    contents = canonicalize_contents(snapshots, tombstones, require_recommendable=False)
    assert contents.frame.set_index("content_id")["value"].to_dict() == {
        "1": "republished",
        "3": "new",
    }
    assert contents.metrics["duplicate_canonical_ids"] == 0

    behavior = canonicalize_behavior_events(
        pd.DataFrame(
            {
                "profile_id": ["p1", "p1", None],
                "user_name": ["u1", "u1", "u2"],
                "content_id": ["1", "2", "3"],
                "action": ["opaque_a", "opaque_b", "opaque_c"],
                "duration": [3, 4, 5],
                "create_date": ["2025-01-01", "2025-02-01", "bad"],
            }
        )
    )
    assert behavior.metrics["behavior_canonical_rows"] == 2
    historical = historical_behavior_features(behavior.frame, "2025-01-15")
    assert historical.loc[0, "historical_behavior_count"] == 1
    assert historical.loc[0, "historical_behavior_duration"] == 3


def test_profile_mapping_uses_most_frequent_account_and_reports_typed_ids():
    interactions = pd.DataFrame(
        {
            "profile_id": [1, 1, 1, 2],
            "username": ["b", "a", "a", "c"],
        }
    )
    mapping = canonical_profile_mapping(interactions)
    assert mapping.set_index("profile_id").loc["1", "username"] == "a"
    assert mapping.set_index("profile_id").loc["1", "mapping_interactions"] == 2


def test_uppercase_content_tag_schema_and_unseen_negative_sampling():
    mapping = pd.DataFrame({"CONTENT_ID": [1, 1, 2], "TAG_ID": [10, 11, 10]})
    tags = pd.DataFrame(
        {"TAG_ID": [10, 11], "TAG_NAME": ["sports", "inactive"], "STATUS": [1, 0]}
    )
    canonical = canonicalize_content_tags(mapping, tags)
    assert canonical.to_dict("records") == [
        {"content_id": "1", "tag_id": "10", "tag_name": "sports"},
        {"content_id": "2", "tag_id": "10", "tag_name": "sports"},
    ]

    positives = pd.DataFrame({"profile_id": ["p1"], "content_id": ["c1"]})
    consumed = pd.DataFrame({"profile_id": ["p1"], "content_id": ["c2"]})
    catalog = pd.DataFrame(
        {"content_id": ["c1", "c2", "c3", "c4"], "popularity": [4, 3, 2, 1]}
    )
    first = deterministic_negative_samples(positives, catalog, consumed, ratio=2, seed=42)
    second = deterministic_negative_samples(positives, catalog, consumed, ratio=2, seed=42)
    assert first.equals(second)
    assert set(first["content_id"]) == {"c3", "c4"}

    candidates = evaluation_candidates(
        positives, catalog, consumed=consumed, negatives_per_profile=2, seed=42
    )
    assert set(candidates[candidates["label"].eq(0)]["content_id"]) == {"c3", "c4"}


def test_behavior_part_aggregation_is_cutoff_and_profile_safe(tmp_path):
    pd.DataFrame(
        {
            "profile_id": ["p1", "p2"],
            "duration": [3.0, 4.0],
            "create_date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
        }
    ).to_parquet(tmp_path / "part_1.parquet", index=False)
    pd.DataFrame(
        {
            "profile_id": ["p1", "p1"],
            "duration": [5.0, 100.0],
            "create_date": pd.to_datetime(["2025-01-02", "2025-02-01"]),
        }
    ).to_parquet(tmp_path / "part_2.parquet", index=False)
    features = historical_behavior_features_from_parts(
        sorted(tmp_path.glob("*.parquet")),
        "2025-01-15",
        profile_ids={"p1"},
    )
    assert features["profile_id"].tolist() == ["p1"]
    assert features.loc[0, "historical_behavior_count"] == 2
    assert features.loc[0, "historical_behavior_duration"] == 8.0
