import pandas as pd

from recsys_vnpt.evaluation import evaluate_scored_candidates
from recsys_vnpt.metrics import (
    hierarchical_relative_harmonic_score,
    paired_bootstrap_confidence_interval,
)
from recsys_vnpt.retrieval import (
    RetrievalConfig,
    build_personalized_candidate_pools,
    candidate_recall_by_profile,
    candidate_recall_curve,
    positive_hit_source_counts,
    retrieval_truth_attribution,
    pair_keys_hash,
    prepare_retrieval_index,
    prepare_retrieval_truth,
)
from recsys_vnpt.rule_assembly import (
    aggregate_model_rule_scores,
    rank_scored_rules,
    rank_scored_rules_and_all_with_diagnostics,
    rank_scored_rules_with_diagnostics,
)


def _catalog():
    contents = pd.DataFrame(
        {
            "content_id": [f"c{i:02d}" for i in range(24)],
            "type_id": ["2"] * 24,
            "content_status": ["1"] * 24,
            "tag_names": ["sports" if i < 12 else "movie" for i in range(24)],
        }
    )
    membership = pd.DataFrame(
        [
            {"content_id": content_id, "rule_id": str(rule)}
            for rule in range(1, 21)
            for content_id in contents["content_id"]
        ]
    )
    return contents, membership


def test_shared_retrieval_index_gives_each_shard_the_same_candidates():
    contents, membership = _catalog()
    popularity = {f"c{i:02d}": i for i in range(24)}
    interactions = pd.DataFrame(
        {
            "profile_id": ["p1", "p1", "p2", "p3", "p3"],
            "content_id": ["c00", "c01", "c20", "c01", "c20"],
            "create_date": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-03", "2025-01-04"]
            ),
        }
    )
    config = RetrievalConfig(candidate_budget=200, candidate_rules=20, reserved_per_rule=10)
    together, _ = build_personalized_candidate_pools(
        pd.DataFrame({"profile_id": ["p1", "p2", "p3"]}),
        contents, membership, popularity, interactions, cutoff="2025-02-01", config=config,
    )
    index = prepare_retrieval_index(
        contents, membership, popularity, interactions, cutoff="2025-02-01", config=config
    )
    shards = [
        build_personalized_candidate_pools(
            pd.DataFrame({"profile_id": shard}), contents, membership, popularity,
            interactions, cutoff="2025-02-01", config=config, index=index,
        )[0]
        for shard in (["p2"], ["p1", "p3"])
    ]
    columns = ["profile_id", "content_id", "candidate_rule_id", "retrieval_score"]
    ordered = lambda frame: frame[columns].sort_values(columns[:3]).reset_index(drop=True)
    pd.testing.assert_frame_equal(ordered(together), ordered(pd.concat(shards)))


def test_personalized_retrieval_reports_actual_k_and_profile_recall():
    contents, membership = _catalog()
    profiles = pd.DataFrame({"profile_id": ["p1", "p2"]})
    interactions = pd.DataFrame(
        {
            "profile_id": ["p1", "p1", "p2"],
            "content_id": ["c00", "c01", "c20"],
            "create_date": pd.to_datetime(
                ["2025-01-01", "2025-01-02", "2025-01-01"]
            ),
        }
    )
    candidates, metrics = build_personalized_candidate_pools(
        profiles,
        contents,
        membership,
        {f"c{i:02d}": i for i in range(24)},
        interactions,
        cutoff="2025-02-01",
        config=RetrievalConfig(
            candidate_budget=200,
            candidate_rules=20,
            reserved_per_rule=10,
        ),
    )
    assert candidates.groupby("profile_id")["content_id"].nunique().max() <= 200
    assert metrics["max_candidates_per_profile"] <= 200
    candidate_keys = set(
        candidates[["profile_id", "content_id"]].itertuples(
            index=False, name=None
        )
    )
    assert not candidate_keys & {("p1", "c00"), ("p1", "c01"), ("p2", "c20")}
    assert metrics["exclude_consumed"]
    assert metrics["consumed_candidate_overlap"] == 0
    positives = pd.DataFrame(
        {"profile_id": ["p1", "p2"], "content_id": ["c23", "missing"]}
    )
    recall, by_profile = candidate_recall_by_profile(candidates, positives)
    assert recall == 0.5
    assert by_profile == {"p1": 1.0, "p2": 0.0}

    reversed_candidates, reversed_metrics = build_personalized_candidate_pools(
        profiles,
        contents,
        membership,
        {f"c{i:02d}": i for i in range(24)},
        interactions.iloc[::-1].reset_index(drop=True),
        cutoff="2025-02-01",
        config=RetrievalConfig(
            candidate_budget=200,
            candidate_rules=20,
            reserved_per_rule=10,
        ),
    )
    columns = ["profile_id", "content_id"]
    pd.testing.assert_frame_equal(
        candidates[columns].reset_index(drop=True),
        reversed_candidates[columns].reset_index(drop=True),
    )
    assert metrics["candidate_pairs_hash"] == reversed_metrics["candidate_pairs_hash"]


def test_recency_reservation_adds_cold_catalog_items_without_expanding_budget():
    contents = pd.DataFrame(
        {
            "content_id": [f"c{i:03d}" for i in range(150)],
            "type_id": ["2"] * 150,
            "content_status": ["1"] * 150,
            "tag_names": ["movie"] * 150,
            "publish_date": pd.date_range("2024-01-01", periods=150, freq="D"),
        }
    )
    membership = pd.DataFrame(
        [
            {"content_id": content_id, "rule_id": str(rule_id)}
            for rule_id in range(1, 21)
            for content_id in contents["content_id"]
        ]
    )
    popularity = {
        content_id: float(150 - index)
        for index, content_id in enumerate(contents["content_id"])
    }
    profiles = pd.DataFrame({"profile_id": ["p1"]})

    baseline, _ = build_personalized_candidate_pools(
        profiles,
        contents,
        membership,
        popularity,
        config=RetrievalConfig(
            candidate_budget=100,
            candidate_rules=20,
            reserved_per_rule=5,
        ),
    )
    recency, metrics = build_personalized_candidate_pools(
        profiles,
        contents,
        membership,
        popularity,
        config=RetrievalConfig(
            candidate_budget=100,
            candidate_rules=20,
            reserved_per_rule=5,
            recency_candidates_per_rule=5,
            recency_reserved_per_rule=5,
        ),
    )

    newest = set(contents.tail(5)["content_id"])
    assert not newest & set(baseline["content_id"])
    assert newest <= set(recency["content_id"])
    assert recency["content_id"].nunique() == 100
    assert metrics["catalog_recency_source_enabled"]
    assert metrics["recency_reserved_per_rule"] == 5
    assert recency.loc[
        recency["content_id"].isin(newest), "retrieval_sources"
    ].str.contains("catalog_recency").all()

    age_limited, _ = build_personalized_candidate_pools(
        profiles,
        contents,
        membership,
        popularity,
        cutoff="2024-05-29",
        config=RetrievalConfig(
            candidate_budget=100,
            candidate_rules=20,
            reserved_per_rule=5,
            recency_candidates_per_rule=5,
            recency_reserved_per_rule=5,
            recency_max_age_days=2,
        ),
    )
    allowed_recent = set(contents.iloc[147:150]["content_id"])
    recency_only = age_limited[
        age_limited["retrieval_sources"].str.contains("catalog_recency")
    ]
    assert set(recency_only["content_id"]) <= allowed_recent


def test_recency_reservation_never_uses_content_published_after_cutoff():
    contents = pd.DataFrame(
        {
            "content_id": [
                *[f"c{i:03d}" for i in range(120)],
                "future",
            ],
            "type_id": ["2"] * 121,
            "content_status": ["1"] * 121,
            "tag_names": ["movie"] * 121,
            "publish_date": [
                *pd.date_range("2024-01-01", periods=120, freq="D"),
                pd.Timestamp("2025-01-01"),
            ],
        }
    )
    membership = pd.DataFrame(
        [
            {"content_id": content_id, "rule_id": str(rule_id)}
            for rule_id in range(1, 21)
            for content_id in contents["content_id"]
        ]
    )
    popularity = {
        content_id: float(121 - index)
        for index, content_id in enumerate(contents["content_id"])
    }

    candidates, _ = build_personalized_candidate_pools(
        pd.DataFrame({"profile_id": ["p1"]}),
        contents,
        membership,
        popularity,
        cutoff="2024-04-29",
        config=RetrievalConfig(
            candidate_budget=100,
            candidate_rules=20,
            reserved_per_rule=5,
            recency_candidates_per_rule=5,
            recency_reserved_per_rule=5,
            recency_max_age_days=0,
        ),
    )

    future = candidates[candidates["content_id"] == "future"]
    assert future.empty or not future["retrieval_sources"].str.contains(
        "catalog_recency"
    ).any()


def test_retrieval_truth_separates_repeat_novel_and_unavailable_pairs():
    positives = pd.DataFrame(
        {
            "profile_id": ["p1", "p1", "p2"],
            "content_id": ["seen", "eligible", "missing"],
            "label": [1, 1, 1],
        }
    )
    history = pd.DataFrame(
        {
            "profile_id": ["p1"],
            "content_id": ["seen"],
            "create_date": pd.to_datetime(["2025-01-01"]),
        }
    )
    truth = prepare_retrieval_truth(
        positives,
        history,
        {"eligible"},
        cutoff="2025-02-01",
    )
    assert set(truth.raw["content_id"]) == {"seen", "eligible", "missing"}
    assert set(truth.novel["content_id"]) == {"eligible", "missing"}
    assert truth.eligible["content_id"].tolist() == ["eligible"]
    assert truth.metrics["repeat_positive_pairs_excluded"] == 1
    assert truth.metrics["unavailable_novel_positive_pairs"] == 1
    assert truth.metrics["candidate_truth_catalog_coverage"] == 0.5


def test_candidate_recall_curve_uses_explicit_rank_and_attributes_hits():
    candidates = pd.DataFrame(
        {
            "profile_id": ["p1", "p1", "p2"],
            "content_id": ["a", "b", "c"],
            "candidate_rank": [1, 2, 1],
            "retrieval_source": ["pop", "cowatch", "pop"],
        }
    )
    positives = pd.DataFrame(
        {
            "profile_id": ["p1", "p1", "p2"],
            "content_id": ["a", "b", "missing"],
        }
    )
    curve = candidate_recall_curve(candidates, positives, k_values=(1, 2))
    assert curve == {"candidate_recall@1": 1 / 3, "candidate_recall@2": 2 / 3}
    assert positive_hit_source_counts(candidates, positives) == {
        "pop": 1,
        "cowatch": 1,
    }
    assert pair_keys_hash(positives) == pair_keys_hash(
        positives.iloc[::-1].drop_duplicates()
    )


def test_retrieval_truth_attribution_separates_overlapping_sources_and_cohorts():
    candidates = pd.DataFrame(
        {
            "profile_id": ["p1", "p1", "p2"],
            "content_id": ["a", "b", "c"],
            "retrieval_source": ["pop", "pop", "tower"],
            "retrieval_sources": ["pop|tower", "pop", "tower"],
            "is_incumbent_candidate": [True, True, False],
            "is_external_candidate": [True, False, True],
        }
    )
    positives = pd.DataFrame(
        {
            "profile_id": ["p1", "p1", "p2", "p2"],
            "content_id": ["a", "b", "c", "missing"],
            "completion_gain": [1.0, 0.5, 0.25, 0.75],
            "watch_time_gain": [0.8, 0.4, 0.2, 0.6],
        }
    )
    attributes = pd.DataFrame(
        {
            "profile_id": ["p1", "p1", "p2", "p2"],
            "content_id": ["a", "b", "c", "missing"],
            "content_oov": [False, False, True, True],
            "activity_bucket": ["high", "high", "cold", "cold"],
        }
    )

    metrics = retrieval_truth_attribution(candidates, positives, attributes)

    assert metrics["truth_pairs"] == 4
    assert metrics["truth_hits"] == 3
    assert metrics["candidate_recall"] == 0.75
    assert metrics["primary_source"]["pop"]["truth_hits"] == 2
    assert metrics["inclusive_witness"]["tower"]["truth_hits"] == 2
    assert metrics["fractional_credit"]["pop"]["truth_hit_credit"] == 1.5
    assert metrics["fractional_credit"]["tower"]["truth_hit_credit"] == 1.5
    assert metrics["exclusive_source"]["tower"]["truth_hits"] == 1
    cold_content = metrics["cohorts"]["content_oov"]["true"]
    assert cold_content["truth_pairs"] == 2
    assert cold_content["truth_hits"] == 1
    assert cold_content["candidate_recall"] == 0.5
    assert cold_content["completion_gain_recall"] == 0.25


def test_evaluator_excludes_underfilled_rules_and_reports_contract_rate():
    contents, membership = _catalog()
    scored = pd.DataFrame(
        {
            "profile_id": ["p1"] * 9,
            "content_id": [f"c{i:02d}" for i in range(9)],
            "type_id": ["2"] * 9,
            "label": [1] + [0] * 8,
            "score": [1.0 - i / 10 for i in range(9)],
        }
    )
    metrics, ranked = evaluate_scored_candidates(scored, membership)
    assert ranked["p1"] == []
    assert metrics["exact_20_normal_rule_rate"] == 0.0
    assert metrics["underfilled_rule_groups_excluded"] == 20
    assert metrics["overall_recall@200"] == 0.0
    assert metrics["flat_candidate_recall@200"] == 1.0


def test_evaluator_cohort_metrics_ignore_unneeded_wide_columns():
    contents, membership = _catalog()
    scored = contents[["content_id", "type_id"]].copy()
    scored["profile_id"] = "p1"
    scored["label"] = [1] + [0] * (len(scored) - 1)
    scored["score"] = [1.0 - index / 100 for index in range(len(scored))]
    scored["profile_oov"] = False
    scored["content_oov"] = [index % 2 == 0 for index in range(len(scored))]
    scored["activity_bucket"] = "high"
    scored["content_family"] = "movie"
    for index in range(50):
        scored[f"unused_object_{index}"] = f"value-{index}"

    metrics, _ = evaluate_scored_candidates(scored, membership)

    high = metrics["cohorts"]["activity_bucket=high"]
    assert high["candidate_pairs"] == len(scored)
    assert high["positive_pairs"] == 1
    assert high["positive_mean_score"] == 1.0


def test_direct_rule_score_orders_rules_before_content_tiebreakers():
    rows = []
    for rule_id, rule_score, content_score in (
        ("1", 0.1, 0.9),
        ("2", 0.9, 0.5),
    ):
        for index in range(10):
            rows.append(
                {
                    "profile_id": "p1",
                    "rule_id": rule_id,
                    "content_id": f"{rule_id}-{index}",
                    "type_id": "2",
                    "score": content_score,
                    "rule_score": rule_score,
                }
            )
    ranked = rank_scored_rules(pd.DataFrame(rows), rule_k=2, content_k=10)
    assert [row["rule_id"] for row in ranked["p1"]] == ["2", "1"]


def test_model_rule_scores_use_actual_many_to_many_membership() -> None:
    scored = pd.DataFrame(
        {
            "profile_id": ["p1", "p1"],
            "content_id": ["shared", "only-a"],
            "candidate_rule_id": ["a", "a"],
            "model_rule_score": [0.8, 0.2],
        }
    )
    membership = pd.DataFrame(
        {
            "content_id": ["shared", "shared", "only-a"],
            "rule_id": ["a", "b", "a"],
        }
    )

    scores = aggregate_model_rule_scores(scored, membership, content_k=10)
    by_rule = scores.set_index("rule_id")["rule_score"].to_dict()

    assert by_rule == {"a": 0.5, "b": 0.8}


def test_vectorized_rule_ranking_deduplicates_and_reports_underfilled_groups():
    rows = []
    for rule_id, count in (("1", 10), ("2", 9)):
        for index in range(count):
            rows.append(
                {
                    "profile_id": "p1",
                    "rule_id": rule_id,
                    "content_id": f"{rule_id}-{index}",
                    "type_id": "2",
                    "score": 1.0 - index / 100,
                }
            )
    rows.append({**rows[0], "score": 0.0})
    ranked, diagnostics = rank_scored_rules_with_diagnostics(
        pd.DataFrame(rows), rule_k=20, content_k=10
    )
    assert [row["rule_id"] for row in ranked["p1"]] == ["1"]
    assert ranked["p1"][0]["content_ids"][0] == "1-0"
    assert diagnostics == {
        "rule_groups": 2,
        "underfilled_rule_groups_excluded": 1,
        "eligible_rule_groups": 1,
    }


def test_selected_and_all_rule_maps_do_not_share_mutable_content_lists():
    rows = [
        {
            "profile_id": "p1",
            "rule_id": rule_id,
            "content_id": f"{rule_id}-{index}",
            "type_id": "2",
            "score": 1.0 - index / 100,
        }
        for rule_id in ("1", "2")
        for index in range(10)
    ]

    selected, all_eligible, _ = rank_scored_rules_and_all_with_diagnostics(
        pd.DataFrame(rows), rule_k=1, content_k=10
    )
    selected["p1"][0]["content_ids"].pop()

    assert len(selected["p1"][0]["content_ids"]) == 9
    assert len(all_eligible["p1"][0]["content_ids"]) == 10


def test_paired_bootstrap_and_harmonic_promotion_score():
    interval = paired_bootstrap_confidence_interval(
        {"p1": 0.1, "p2": 0.2, "p3": 0.3},
        {"p1": 0.2, "p2": 0.3, "p3": 0.4},
        iterations=200,
    )
    assert interval["lower"] > 0
    assert interval["paired_profiles"] == 3
    assert hierarchical_relative_harmonic_score(0.6, 0.3, 0.5, 0.25) == 1.2


def test_evaluator_reads_only_the_columns_its_metrics_need():
    """The serving path hands this the full candidate frame.

    Copying all 49 columns materialised 2.14 GiB of object dtype for a run that
    reads five columns plus the cohort keys, and ran a 16 GB machine out of
    memory on a 100-profile smoke test.
    """

    from recsys_vnpt.evaluation import evaluation_input_columns

    scored = pd.DataFrame(
        {
            "profile_id": ["p1"],
            "content_id": ["c00"],
            "type_id": ["2"],
            "label": [1],
            "score": [1.0],
            "activity_bucket": ["high"],
            "content_family": ["movie"],
            "completion_gain": [0.5],
            "watch_time_gain": [0.5],
            "retrieval_sources": ["a|b"],
            "external_fusion_score": [0.1],
        }
    )
    kept = evaluation_input_columns(scored, ground_truth_frame=None)
    assert "retrieval_sources" not in kept
    assert "external_fusion_score" not in kept
    assert {"profile_id", "content_id", "type_id", "label", "score"} <= set(kept)
    assert {"activity_bucket", "content_family"} <= set(kept)
    # Gains are the truth when no separate truth frame is supplied.
    assert {"completion_gain", "watch_time_gain"} <= set(kept)
    # With an explicit truth frame the gains are read from there instead.
    supplied = evaluation_input_columns(scored, ground_truth_frame=scored)
    assert "completion_gain" not in supplied


def test_evaluator_metrics_survive_dropping_unused_columns():
    contents, membership = _catalog()
    lean = pd.DataFrame(
        {
            "profile_id": ["p1"] * 9,
            "content_id": [f"c{i:02d}" for i in range(9)],
            "type_id": ["2"] * 9,
            "label": [1] + [0] * 8,
            "score": [1.0 - i / 10 for i in range(9)],
        }
    )
    wide = lean.copy()
    for index in range(30):
        wide[f"unused_object_{index}"] = f"value-{index}"

    lean_metrics, _ = evaluate_scored_candidates(lean, membership)
    wide_metrics, _ = evaluate_scored_candidates(wide, membership)
    assert lean_metrics["overall_recall@200"] == wide_metrics["overall_recall@200"]
    assert lean_metrics["cohorts"] == wide_metrics["cohorts"]
