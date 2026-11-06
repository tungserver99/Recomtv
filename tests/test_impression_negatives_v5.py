import pandas as pd

from recsys_vnpt.impressions import build_impressions, window_impression_negatives
from recsys_vnpt.temporal import mixed_negative_samples


def _behavior_part(tmp_path, rows):
    path = tmp_path / "part-00000.gz.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_ingest_scopes_to_vod_display_rows_and_dedups_per_day(tmp_path):
    part = _behavior_part(
        tmp_path,
        [
            # kept: display action, VOD-shaped, in catalog
            {"profile_id": "p1", "content_id": "151474", "action": "impressed",
             "start_time": "20250108 13:28:07", "other": "x"},
            # kept: same pair later the same day -> deduped to one row
            {"profile_id": "p1", "content_id": "151474", "action": "poster",
             "start_time": "20250108 19:02:00", "other": "x"},
            # kept: next day counts separately
            {"profile_id": "p1", "content_id": "151474", "action": "impressed",
             "start_time": "20250109 08:00:00", "other": "x"},
            # dropped: live-TV channel number (too short)
            {"profile_id": "p1", "content_id": "423", "action": "impressed",
             "start_time": "20250108 13:30:00", "other": "x"},
            # dropped: ad event action
            {"profile_id": "p1", "content_id": "152503", "action": "ima_start",
             "start_time": "20250108 13:31:00", "other": "x"},
            # kept but tier 1: attention action, noisier evidence
            {"profile_id": "p2", "content_id": "152503", "action": "focus",
             "start_time": "20250108 13:32:00", "other": "x"},
            # dropped: VOD-shaped but outside the catalog
            {"profile_id": "p2", "content_id": "999999", "action": "impressed",
             "start_time": "20250108 13:33:00", "other": "x"},
            # dropped: unparseable timestamp
            {"profile_id": "p2", "content_id": "152503", "action": "impressed",
             "start_time": "not-a-date", "other": "x"},
        ],
    )

    impressions, metrics = build_impressions([part], {"151474", "152503"})

    assert list(impressions.columns) == [
        "profile_id", "content_id", "event_date", "source_tier",
    ]
    assert len(impressions) == 3  # two p1 days + one p2 focus row
    assert metrics["behavior_raw_rows"] == 8
    assert metrics["impression_display_rows"] == 7  # + focus, still no ima_*
    assert metrics["impression_vod_shaped_rows"] == 6  # channel number dropped
    assert metrics["impression_catalog_matched_rows"] == 5  # 999999 dropped
    assert metrics["impression_dated_rows"] == 4  # bad timestamp dropped
    # p1 same-day pair keeps the cleaner tier despite the later poster row
    p1_day = impressions[impressions["profile_id"].eq("p1")].iloc[0]
    assert p1_day["source_tier"] == 0
    assert impressions[impressions["profile_id"].eq("p2")]["source_tier"].tolist() == [1]


def test_window_negatives_exclude_watched_and_consumed_pairs():
    impressions = pd.DataFrame(
        {
            "profile_id": ["p1", "p1", "p1", "p1"],
            "content_id": ["shown", "watched", "consumed", "outside"],
            "event_date": pd.to_datetime(
                ["2025-02-03", "2025-02-03", "2025-02-04", "2025-03-01"]
            ),
            "source_tier": [0, 0, 0, 0],
        }
    )
    watched = pd.DataFrame({"profile_id": ["p1"], "content_id": ["watched"]})
    consumed = pd.DataFrame({"profile_id": ["p1"], "content_id": ["consumed"]})

    negatives = window_impression_negatives(
        impressions,
        watched,
        pd.Timestamp("2025-02-01"),
        pd.Timestamp("2025-02-10 23:59:59"),
        consumed=consumed,
    )

    assert negatives == {"p1": ["shown"]}


def test_impression_negative_replaces_popularity_slot_keeping_count():
    positives = pd.DataFrame({"profile_id": ["p1"], "content_id": ["c1"]})
    catalog = pd.DataFrame(
        {"content_id": [f"c{i}" for i in range(1, 13)], "popularity": range(12)}
    )
    membership = pd.DataFrame(
        {"content_id": ["c1", "c2", "c3"], "rule_id": ["r1", "r1", "r1"]}
    )

    without = mixed_negative_samples(positives, catalog, None, membership, seed=42)
    with_impression = mixed_negative_samples(
        positives,
        catalog,
        None,
        membership,
        seed=42,
        impression_negatives={"p1": ["c9"]},
    )

    assert len(with_impression) == len(without)  # equal-count replacement
    types = set(with_impression["sample_type"])
    assert "shown_not_watched" in types
    assert "popularity_matched_negative" not in types
    real = with_impression[with_impression["sample_type"].eq("shown_not_watched")]
    assert real["content_id"].tolist() == ["c9"]
    assert real["sample_weight"].tolist() == [1.0]  # observed evidence
    sampled = with_impression[with_impression["sample_type"].ne("shown_not_watched")]
    assert sampled["sample_weight"].eq(0.25).all()


def test_impression_negative_falls_back_when_profile_has_none():
    positives = pd.DataFrame({"profile_id": ["p1"], "content_id": ["c1"]})
    catalog = pd.DataFrame(
        {"content_id": [f"c{i}" for i in range(1, 13)], "popularity": range(12)}
    )

    baseline = mixed_negative_samples(positives, catalog, seed=42)
    other_profile = mixed_negative_samples(
        positives, catalog, seed=42, impression_negatives={"p2": ["c9"]}
    )

    pd.testing.assert_frame_equal(baseline, other_profile)


def test_candidates_are_ordered_clean_evidence_first():
    impressions = pd.DataFrame(
        {
            "profile_id": ["p1", "p1"],
            "content_id": ["focus_only", "impressed_one"],
            "event_date": pd.to_datetime(["2025-02-03", "2025-02-03"]),
            "source_tier": [1, 0],
        }
    )

    negatives = window_impression_negatives(
        impressions,
        pd.DataFrame(columns=["profile_id", "content_id"]),
        pd.Timestamp("2025-02-01"),
        pd.Timestamp("2025-02-10 23:59:59"),
    )

    assert negatives["p1"] == ["impressed_one", "focus_only"]
