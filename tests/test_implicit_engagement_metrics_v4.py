from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from recsys_vnpt.evaluation import evaluate_scored_candidates
from recsys_vnpt.implicit_baselines import (
    ImplicitBaselineConfig,
    score_implicit_baseline,
)
from recsys_vnpt.hybrid_retrieval import (
    HybridRetrievalConfig,
    _deterministic_topk,
    _leave_one_out_history,
    train_and_rank_hybrid_retriever,
)
from recsys_vnpt.metrics import (
    gain_recall_at_k,
    graded_ndcg_at_k,
)
from recsys_vnpt.model_runner import TrainingConfig, train_model
from recsys_vnpt.retrieval import (
    RetrievalConfig,
    build_personalized_candidate_pools,
    candidate_gain_recall_by_profile,
    fuse_content_scores_with_external_rank,
    pair_keys_hash,
    ranking_map_incremental_metrics,
    ranking_map_recall_metrics,
)
from recsys_vnpt.sequence_retrieval import (
    SASRecLiteConfig,
    train_and_rank_sasrec_lite,
)


def _events() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    histories = {
        "p1": [("a", "s1"), ("b", "s1"), ("a", "s2")],
        "p2": [("a", "s3"), ("c", "s3")],
        "p3": [("b", "s4"), ("d", "s4")],
    }
    for profile_id, values in histories.items():
        for index, (content_id, session_id) in enumerate(values):
            rows.append(
                {
                    "profile_id": profile_id,
                    "content_id": content_id,
                    "session_id": session_id,
                    "event_end": pd.Timestamp("2025-01-01")
                    + pd.Timedelta(index, unit="m"),
                    "watch_seconds": 30.0 + index,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "family",
    (
        "graded_global_popularity",
        "session_item_knn",
        "bpr_mf",
        "lightgcn",
    ),
)
def test_implicit_baselines_emit_rank_scores_and_artifacts(tmp_path, family) -> None:
    candidates = pd.DataFrame(
        [(profile, content) for profile in ("p1", "p2") for content in "abcd"],
        columns=["profile_id", "content_id"],
    )
    config = ImplicitBaselineConfig(
        family=family,
        embedding_dim=4,
        epochs=1,
        batch_size=4,
        lightgcn_layers=1,
        seed=42,
    )

    result = score_implicit_baseline(
        _events(),
        candidates,
        "2025-01-31",
        config,
        output_dir=tmp_path,
        device="cpu",
    )

    assert len(result.scores) == len(candidates)
    assert np.isfinite(result.scores).all()
    assert result.metrics["classification_metrics_valid"] is False
    assert "not a true negative" in result.metrics["unobserved_semantics"]
    assert result.artifact_path is not None and result.artifact_path.exists()


def test_per_rule_popularity_uses_serving_retrieval_score() -> None:
    candidates = pd.DataFrame(
        {
            "profile_id": ["p1", "p1"],
            "content_id": ["a", "b"],
            "retrieval_score": [0.2, 0.9],
        }
    )

    result = score_implicit_baseline(
        _events(),
        candidates,
        "2025-01-31",
        ImplicitBaselineConfig(family="graded_per_rule_popularity"),
        device="cpu",
    )

    assert result.scores.tolist() == pytest.approx([0.2, 0.9])
    assert result.metrics["score_source"] == "serving_personalized_retrieval_score"
    with pytest.raises(ValueError, match="requires serving retrieval_score"):
        score_implicit_baseline(
            _events(),
            candidates.drop(columns="retrieval_score"),
            "2025-01-31",
            ImplicitBaselineConfig(family="graded_per_rule_popularity"),
            device="cpu",
        )


def test_bpr_per_rule_fusion_fuses_serving_rank_with_bpr_rank() -> None:
    candidates = pd.DataFrame(
        [(profile, content) for profile in ("p1", "p2") for content in "abcd"],
        columns=["profile_id", "content_id"],
    )
    candidates["retrieval_score"] = [0.9, 0.8, 0.7, 0.6, 0.9, 0.8, 0.7, 0.6]

    def _config(weight: float) -> ImplicitBaselineConfig:
        return ImplicitBaselineConfig(
            family="bpr_per_rule_fusion",
            embedding_dim=4,
            epochs=1,
            batch_size=4,
            fusion_bpr_weight=weight,
            seed=42,
        )

    disabled = score_implicit_baseline(
        _events(), candidates, "2025-01-31", _config(0.0), device="cpu"
    )
    assert disabled.scores.tolist() == pytest.approx(
        candidates["retrieval_score"].tolist()
    )

    fused = score_implicit_baseline(
        _events(), candidates, "2025-01-31", _config(1.0), device="cpu"
    )
    assert np.isfinite(fused.scores).all()
    assert fused.metrics["score_source"] == "per_rule_retrieval_bpr_rank_rrf_fusion"
    assert fused.metrics["fusion_bpr_rank_coverage"] == pytest.approx(1.0)
    assert fused.metrics["epochs_completed"] == 1
    # RRF fusion bounds every fused score by the two rank components.
    assert float(fused.scores.max()) <= 2.0 / 61.0 + 1e-9

    with pytest.raises(ValueError, match="requires serving retrieval_score"):
        score_implicit_baseline(
            _events(),
            candidates.drop(columns="retrieval_score"),
            "2025-01-31",
            _config(1.0),
            device="cpu",
        )


def test_knn_per_rule_fusion_fuses_serving_rank_with_knn_rank() -> None:
    candidates = pd.DataFrame(
        [(profile, content) for profile in ("p1", "p2") for content in "abcd"],
        columns=["profile_id", "content_id"],
    )
    candidates["retrieval_score"] = [0.9, 0.8, 0.7, 0.6, 0.9, 0.8, 0.7, 0.6]

    def _config(weight: float) -> ImplicitBaselineConfig:
        return ImplicitBaselineConfig(
            family="knn_per_rule_fusion", fusion_bpr_weight=weight, seed=42
        )

    disabled = score_implicit_baseline(
        _events(), candidates, "2025-01-31", _config(0.0), device="cpu"
    )
    assert disabled.scores.tolist() == pytest.approx(
        candidates["retrieval_score"].tolist()
    )

    fused = score_implicit_baseline(
        _events(), candidates, "2025-01-31", _config(1.0), device="cpu"
    )
    assert np.isfinite(fused.scores).all()
    assert (
        fused.metrics["score_source"]
        == "per_rule_retrieval_session_knn_rank_rrf_fusion"
    )
    assert fused.metrics["fusion_bpr_rank_coverage"] == pytest.approx(1.0)
    assert float(fused.scores.max()) <= 2.0 / 61.0 + 1e-9


def test_per_rule_popularity_does_not_deep_copy_wide_candidate_frame() -> None:
    class NoCopyFrame(pd.DataFrame):
        def copy(self, *args, **kwargs):  # pragma: no cover - failure path
            raise AssertionError("candidate frame must not be deep-copied")

    candidates = NoCopyFrame(
        {
            "profile_id": ["p1", "p1"],
            "content_id": ["a", "b"],
            "retrieval_score": [0.2, 0.9],
            **{
                f"wide_object_{index}": ["x", "y"]
                for index in range(50)
            },
        }
    )

    result = score_implicit_baseline(
        _events(),
        candidates,
        "2025-01-31",
        ImplicitBaselineConfig(family="graded_per_rule_popularity"),
        device="cpu",
    )

    assert result.scores.tolist() == pytest.approx([0.2, 0.9])
    assert result.metrics["candidate_input_columns"] == 53
    assert result.metrics["candidate_frame_deep_copied"] is False


def test_external_rank_fusion_is_scale_free_and_deterministic() -> None:
    candidates = pd.DataFrame(
        {
            "profile_id": ["p1", "p1", "p1"],
            "content_id": ["a", "b", "c"],
            "external_rank": [np.nan, 2, 1],
        }
    )
    base = np.asarray([1.0, 0.9, 0.1])

    disabled, disabled_metrics = fuse_content_scores_with_external_rank(
        candidates, base, external_weight=0.0
    )
    fused, metrics = fuse_content_scores_with_external_rank(
        candidates, base, external_weight=1.0, rrf_constant=60
    )

    assert disabled.tolist() == pytest.approx(base.tolist())
    assert disabled_metrics["content_score_fusion"] == "disabled"
    assert candidates.iloc[int(np.argmax(fused))]["content_id"] == "c"
    assert metrics["content_external_rank_coverage"] == pytest.approx(2 / 3)
    assert metrics["content_score_fusion"] == "base_rank_external_rank_rrf"


def test_graded_ndcg_and_gain_recall_use_continuous_gain() -> None:
    gains = {"high": 1.0, "low": 0.1}

    assert graded_ndcg_at_k(["high", "low"], gains, 2) == pytest.approx(1.0)
    assert graded_ndcg_at_k(["low", "high"], gains, 2) < 1.0
    assert gain_recall_at_k(["high"], gains, 1) == pytest.approx(1.0 / 1.1)


def test_candidate_gain_recall_weights_missed_engagement() -> None:
    candidates = pd.DataFrame(
        {"profile_id": ["p1"], "content_id": ["low"], "candidate_rank": [1]}
    )
    truth = pd.DataFrame(
        {
            "profile_id": ["p1", "p1"],
            "content_id": ["high", "low"],
            "completion_gain": [1.0, 0.1],
        }
    )

    recall, per_profile = candidate_gain_recall_by_profile(
        candidates, truth, "completion_gain", k=1
    )

    assert recall == pytest.approx(0.1 / 1.1)
    assert per_profile["p1"] == pytest.approx(recall)


def test_rank_evaluation_omits_classification_metrics() -> None:
    rows: list[dict[str, object]] = []
    membership_rows: list[dict[str, str]] = []
    for rule in range(20):
        for content_index in range(10):
            content_id = f"c-{rule}-{content_index}"
            membership_rows.append(
                {"content_id": content_id, "rule_id": str(rule)}
            )
            is_truth = rule == 0 and content_index == 0
            rows.append(
                {
                    "profile_id": "p1",
                    "content_id": content_id,
                    "type_id": "1",
                    "label": int(is_truth),
                    "completion_gain": 1.0 if is_truth else 0.0,
                    "watch_time_gain": 0.8 if is_truth else 0.0,
                    "target_semantics": "engagement_v1",
                    "score": 1.0 if is_truth else 0.1,
                }
            )

    metrics, _ = evaluate_scored_candidates(
        pd.DataFrame(rows), pd.DataFrame(membership_rows)
    )

    assert metrics["classification_metrics_valid"] is False
    assert "classification" not in metrics
    assert metrics["target_semantics"] == "engagement_v1"
    assert "completion_gain_rule_ndcg@20" in metrics
    assert "watch_time_gain_macro_content_ndcg@10" in metrics


def test_all_rule_content_metric_is_independent_of_top_20_rule_selection() -> None:
    rows: list[dict[str, object]] = []
    membership_rows: list[dict[str, str]] = []
    for rule in range(21):
        for content_index in range(10):
            content_id = f"c-{rule}-{content_index}"
            is_truth = rule == 20 and content_index == 0
            membership_rows.append(
                {"content_id": content_id, "rule_id": str(rule)}
            )
            rows.append(
                {
                    "profile_id": "p1",
                    "content_id": content_id,
                    "type_id": "1",
                    "label": int(is_truth),
                    "completion_gain": 1.0 if is_truth else 0.0,
                    "watch_time_gain": 1.0 if is_truth else 0.0,
                    "target_semantics": "engagement_v1",
                    "score": 1.0 if is_truth else 0.1,
                }
            )
    scored = pd.DataFrame(rows)
    membership = pd.DataFrame(membership_rows)
    excludes_truth_rule = pd.DataFrame(
        {
            "profile_id": ["p1"] * 21,
            "rule_id": list(map(str, range(21))),
            "rule_score": [1.0] * 20 + [0.0],
        }
    )
    includes_truth_rule = excludes_truth_rule.copy()
    includes_truth_rule.loc[
        includes_truth_rule["rule_id"].eq("20"), "rule_score"
    ] = 2.0

    excluded, _ = evaluate_scored_candidates(
        scored, membership, profile_rule_scores=excludes_truth_rule
    )
    included, _ = evaluate_scored_candidates(
        scored, membership, profile_rule_scores=includes_truth_rule
    )

    assert excluded["completion_gain_macro_content_ndcg@10"] == 0.0
    assert included["completion_gain_macro_content_ndcg@10"] == 1.0
    assert excluded[
        "completion_gain_all_rule_macro_content_ndcg@10"
    ] == pytest.approx(1.0)
    assert included[
        "completion_gain_all_rule_macro_content_ndcg@10"
    ] == pytest.approx(1.0)


def test_hybrid_two_tower_ranks_seen_and_cold_profiles(tmp_path) -> None:
    contents = pd.DataFrame(
        {
            "content_id": list("abcde"),
            "content_family": ["movie"] * 5,
            "type_id": ["1"] * 5,
            "tag_names": ["x", "x,y", "y", "z", "x,z"],
            "rule_ids": ["1", "1", "2", "2", "3"],
        }
    )
    profiles = pd.DataFrame({"profile_id": ["p1", "cold"]})
    metadata = pd.DataFrame(
        {
            "profile_id": ["p1", "cold"],
            "package_code": ["basic", "basic"],
            "sex": ["1", "2"],
            "province": ["hn", "hcm"],
        }
    )
    config = HybridRetrievalConfig(
        embedding_dim=4,
        epochs=1,
        batch_size=2,
        sampled_items=2,
        long_history_length=4,
        recent_history_length=2,
        temporal_holdout_days=0,
        seed=42,
    )

    result = train_and_rank_hybrid_retriever(
        _events(),
        contents,
        profiles,
        "2025-01-31",
        candidate_budget=3,
        profile_metadata=metadata,
        config=config,
        output_dir=tmp_path,
        device="cpu",
    )

    assert set(result.rankings) == {"p1", "cold"}
    assert all(len(values) == 3 for values in result.rankings.values())
    consumed_by_p1 = {"a", "b"}
    assert not consumed_by_p1 & {
        content_id for content_id, _ in result.rankings["p1"]
    }
    assert result.metrics["cold_profile_count"] == 1
    assert result.metrics["classification_metrics_valid"] is False
    assert result.artifact_path is not None and result.artifact_path.exists()


def test_hybrid_eligible_only_scoring_excludes_ineligible_catalog(tmp_path) -> None:
    contents = pd.DataFrame(
        {
            "content_id": list("abcdef"),
            "content_family": ["movie"] * 6,
            "type_id": ["1"] * 6,
            "tag_names": ["x"] * 6,
            "rule_ids": ["1"] * 6,
        }
    )
    eligible = {"c", "d", "e", "f"}
    result = train_and_rank_hybrid_retriever(
        _events(),
        contents,
        pd.DataFrame({"profile_id": ["p1", "cold"]}),
        "2025-01-31",
        candidate_budget=3,
        config=HybridRetrievalConfig(
            embedding_dim=4,
            epochs=1,
            batch_size=2,
            sampled_items=2,
            temporal_holdout_days=0,
            scoring_catalog_policy="eligible_only",
            seed=42,
        ),
        output_dir=tmp_path,
        device="cpu",
        eligible_content_ids=eligible,
    )

    assert all(
        content_id in eligible
        for ranking in result.rankings.values()
        for content_id, _ in ranking
    )
    assert result.metrics["eligible_catalog_contents"] == 4
    assert result.metrics["scored_catalog_contents"] == 4
    assert result.metrics["scored_ineligible_contents"] == 0
    assert result.metrics["scoring_catalog_policy"] == "eligible_only"
    assert result.metrics["retrieved_scored_catalog_coverage"] > 0


def test_hybrid_eligible_only_scoring_requires_catalog() -> None:
    with pytest.raises(ValueError, match="requires eligible_content_ids"):
        train_and_rank_hybrid_retriever(
            _events(),
            pd.DataFrame(
                {
                    "content_id": list("abcde"),
                    "content_family": ["movie"] * 5,
                }
            ),
            pd.DataFrame({"profile_id": ["p1"]}),
            "2025-01-31",
            config=HybridRetrievalConfig(
                embedding_dim=4,
                epochs=1,
                batch_size=2,
                sampled_items=2,
                temporal_holdout_days=0,
                scoring_catalog_policy="eligible_only",
            ),
            device="cpu",
        )


def test_hybrid_temporal_objective_uses_novel_future_targets(tmp_path) -> None:
    events = pd.DataFrame(
        {
            "profile_id": ["p1", "p1", "p1", "p1", "p2", "p2"],
            "content_id": ["a", "b", "a", "c", "a", "d"],
            "event_end": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-25",
                    "2025-01-26",
                    "2025-01-03",
                    "2025-01-27",
                ]
            ),
            "watch_seconds": [10.0, 20.0, 30.0, 40.0, 10.0, 50.0],
        }
    )
    contents = pd.DataFrame(
        {
            "content_id": list("abcde"),
            "content_family": ["movie"] * 5,
            "type_id": ["1"] * 5,
            "tag_names": ["x"] * 5,
            "rule_ids": ["1"] * 5,
            "atomic_duration_seconds": [100.0] * 5,
        }
    )
    result = train_and_rank_hybrid_retriever(
        events,
        contents,
        pd.DataFrame({"profile_id": ["p1", "p2"]}),
        "2025-01-31",
        candidate_budget=2,
        config=HybridRetrievalConfig(
            embedding_dim=4,
            epochs=1,
            batch_size=2,
            sampled_items=2,
            temporal_holdout_days=10,
            item_identity_policy="context_and_target",
            seed=42,
        ),
        output_dir=tmp_path,
        device="cpu",
    )

    assert result.metrics["history_pairs"] == 2
    assert result.metrics["context_events"] == 3
    assert result.metrics["target_events"] == 3
    assert result.metrics["target_semantics"].startswith("temporal_novel")
    assert result.metrics["mean_target_completion_gain"] > 0
    assert result.metrics["target_item_identity_contents"] == 2
    assert result.metrics["item_identity_contents"] == 4
    assert result.metrics["cold_item_identity_policy"] == "context_and_target"


def test_hybrid_topk_breaks_boundary_ties_by_item_index() -> None:
    values, indices = _deterministic_topk(
        torch.tensor([0.9, 0.8, 0.8, 0.8, 0.1]),
        k=3,
    )

    assert indices.tolist() == [0, 1, 2]
    assert values.tolist() == pytest.approx([0.9, 0.8, 0.8])


def test_external_two_tower_rankings_enter_bounded_candidate_pool() -> None:
    content_rows: list[dict[str, object]] = []
    membership_rows: list[dict[str, str]] = []
    popularity: dict[str, float] = {}
    external: list[tuple[str, float]] = []
    for rule_id in range(1, 21):
        for rank in range(30):
            content_id = f"c{rule_id:02d}_{rank:02d}"
            content_rows.append(
                {
                    "content_id": content_id,
                    "type_id": "1",
                    "tag_names": "tag",
                    "content_status": 1,
                }
            )
            membership_rows.append(
                {"content_id": content_id, "rule_id": str(rule_id)}
            )
            popularity[content_id] = float(30 - rank)
        external.append((f"c{rule_id:02d}_00", float(rule_id)))

    baseline, _ = build_personalized_candidate_pools(
        pd.DataFrame({"profile_id": ["p1"]}),
        pd.DataFrame(content_rows),
        pd.DataFrame(membership_rows),
        popularity,
        config=RetrievalConfig(candidate_budget=400),
    )
    candidates, metrics = build_personalized_candidate_pools(
        pd.DataFrame({"profile_id": ["p1"]}),
        pd.DataFrame(content_rows),
        pd.DataFrame(membership_rows),
        popularity,
        config=RetrievalConfig(candidate_budget=400),
        external_rankings={"p1": external},
    )

    assert len(candidates) <= 400
    assert metrics["hybrid_two_tower_enabled"] is True
    assert metrics["hybrid_two_tower_profiles"] == 1
    assert candidates["retrieval_sources"].str.contains("hybrid_two_tower").any()
    assert pair_keys_hash(candidates) == pair_keys_hash(baseline)
    assert metrics["minimum_incumbent_candidate_retention_rate"] == 1.0
    assert metrics["minimum_safety_anchor_retention_rate"] == 1.0


def test_sasrec_lite_only_ranks_profiles_with_sequence_history(tmp_path) -> None:
    events = _events()
    events = pd.concat(
        [
            events,
            pd.DataFrame(
                [
                    {
                        "profile_id": "shallow",
                        "content_id": "e",
                        "session_id": "single",
                        "event_end": pd.Timestamp("2025-01-01"),
                        "watch_seconds": 20.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    config = SASRecLiteConfig(
        embedding_dim=4,
        heads=2,
        layers=1,
        maximum_sequence_length=4,
        sampled_items=2,
        epochs=1,
        batch_size=2,
        seed=42,
    )

    result = train_and_rank_sasrec_lite(
        events,
        pd.DataFrame({"content_id": list("abcdef")}),
        pd.DataFrame({"profile_id": ["p1", "shallow", "cold"]}),
        "2025-01-31",
        candidate_budget=2,
        config=config,
        output_dir=tmp_path,
        device="cpu",
    )

    assert set(result.rankings) == {"p1"}
    assert len(result.rankings["p1"]) == 2
    assert result.metrics["profiles_with_sequence_rankings"] == 1
    assert result.metrics["requested_profiles"] == 3
    assert result.metrics["classification_metrics_valid"] is False
    assert result.artifact_path is not None and result.artifact_path.exists()


def test_sequence_external_source_is_reported_separately() -> None:
    content_rows: list[dict[str, object]] = []
    membership_rows: list[dict[str, str]] = []
    popularity: dict[str, float] = {}
    sequence_ranking: list[tuple[str, float]] = []
    for rule_id in range(1, 21):
        for rank in range(30):
            content_id = f"s{rule_id:02d}_{rank:02d}"
            content_rows.append(
                {
                    "content_id": content_id,
                    "type_id": "1",
                    "tag_names": "tag",
                    "content_status": 1,
                }
            )
            membership_rows.append(
                {"content_id": content_id, "rule_id": str(rule_id)}
            )
            popularity[content_id] = float(30 - rank)
        sequence_ranking.append((f"s{rule_id:02d}_00", float(rule_id)))

    candidates, metrics = build_personalized_candidate_pools(
        pd.DataFrame({"profile_id": ["p1"]}),
        pd.DataFrame(content_rows),
        pd.DataFrame(membership_rows),
        popularity,
        config=RetrievalConfig(candidate_budget=400),
        external_rankings={"p1": sequence_ranking},
        external_source="sasrec_lite",
    )

    assert metrics["sasrec_lite_enabled"] is True
    assert metrics["hybrid_two_tower_enabled"] is False
    assert candidates["retrieval_sources"].str.contains("sasrec_lite").any()


def test_external_replacement_is_quota_bounded_and_preserves_anchors() -> None:
    contents = pd.DataFrame(
        {
            "content_id": [f"c{i:03d}" for i in range(500)],
            "tag_names": ["x"] * 500,
            "type_id": ["1"] * 500,
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
        content_id: float(500 - index)
        for index, content_id in enumerate(contents["content_id"])
    }
    baseline, _ = build_personalized_candidate_pools(
        pd.DataFrame({"profile_id": ["p1"]}),
        contents,
        membership,
        popularity,
        config=RetrievalConfig(
            candidate_budget=200,
            candidate_rules=20,
            reserved_per_rule=5,
        ),
    )
    external = [(f"c{i:03d}", float(i)) for i in range(499, 299, -1)]
    replacement_truth = pd.DataFrame(
        {
            "profile_id": ["p1", "p1", "p1"],
            "content_id": ["c000", "c180", "c499"],
            "completion_gain": [0.25, 0.5, 1.0],
            "watch_time_gain": [0.5, 1.0, 0.75],
        }
    )
    replaced, metrics, blend_audit = build_personalized_candidate_pools(
        pd.DataFrame({"profile_id": ["p1"]}),
        contents,
        membership,
        popularity,
        config=RetrievalConfig(
            candidate_budget=200,
            candidate_rules=20,
            reserved_per_rule=5,
            external_candidate_mode="replace",
            external_candidate_slots=20,
            external_replacement_requires_history=False,
        ),
        external_rankings={"p1": external},
        capture_blend_audit=True,
        evaluation_truth=replacement_truth,
    )
    baseline_anchors = set(
        baseline.loc[baseline["is_safety_anchor"], "content_id"]
    )
    assert baseline_anchors <= set(replaced["content_id"])
    assert metrics["mean_external_promoted_candidates"] <= 20
    assert metrics["minimum_safety_anchor_retention_rate"] == 1.0
    assert metrics["minimum_incumbent_candidate_retention_rate"] >= 0.9
    assert replaced.loc[
        replaced["retrieval_source"].eq("hybrid_two_tower"), "candidate_rank"
    ].min() < 200
    lost_ids = set(blend_audit.lost_incumbents["content_id"])
    assert lost_ids == {f"c{i:03d}" for i in range(180, 200)}
    assert not blend_audit.lost_incumbents["is_safety_anchor"].any()
    retained = replaced[replaced["is_incumbent_candidate"]][
        ["profile_id", "content_id"]
    ]
    reconstructed = pd.concat(
        [
            retained,
            blend_audit.lost_incumbents[["profile_id", "content_id"]],
        ],
        ignore_index=True,
    )
    assert pair_keys_hash(reconstructed) == pair_keys_hash(baseline)
    assert metrics["preblend_incumbent_truth_hits"] == 2
    assert metrics["final_blend_truth_hits"] == 2
    assert metrics["blend_gained_truth_hits"] == 1
    assert metrics["blend_lost_truth_hits"] == 1
    assert metrics["blend_lost_safety_anchor_truth_hits"] == 0
    assert metrics["external_incremental_reference"] == "preblend_incumbent"
    assert metrics["external_incremental_truth_hits"] == 1
    assert metrics["external_oracle_union_truth_hits"] == 3
    assert metrics["external_oracle_union_recall@200+incumbent"] == 1.0
    assert metrics["blend_gained_completion_gain"] == 1.0
    assert metrics["blend_lost_completion_gain"] == 0.5
    profile_audit = blend_audit.profiles.iloc[0]
    assert profile_audit["gained_count"] == 20
    assert profile_audit["lost_count"] == 20
    assert profile_audit["lost_safety_anchor_count"] == 0

    cold_guarded, cold_metrics, cold_audit = build_personalized_candidate_pools(
        pd.DataFrame({"profile_id": ["p1"]}),
        contents,
        membership,
        popularity,
        config=RetrievalConfig(
            candidate_budget=200,
            candidate_rules=20,
            reserved_per_rule=5,
            external_candidate_mode="replace",
            external_candidate_slots=20,
        ),
        external_rankings={"p1": external},
        capture_blend_audit=True,
    )
    assert pair_keys_hash(cold_guarded) == pair_keys_hash(baseline)
    assert cold_metrics["external_replacement_blocked_profiles"] == 1
    assert cold_audit.profiles.iloc[0]["replacement_blocked_no_history"]
    assert cold_audit.profiles.iloc[0]["gained_count"] == 0
    assert cold_audit.profiles.iloc[0]["lost_count"] == 0


def test_external_replacement_witness_gate_restores_incumbent_tail() -> None:
    contents = pd.DataFrame(
        {
            "content_id": [f"c{i:03d}" for i in range(500)],
            "tag_names": ["x"] * 500,
            "type_id": ["1"] * 500,
        }
    )
    membership = pd.DataFrame(
        [
            {"content_id": f"c{i:03d}", "rule_id": str(i % 20 + 1)}
            for i in range(500)
        ]
    )
    popularity = {f"c{i:03d}": float(500 - i) for i in range(500)}
    profiles = pd.DataFrame({"profile_id": ["p1"]})
    common = dict(
        candidate_budget=200,
        candidate_rules=20,
        reserved_per_rule=5,
        source_candidates_per_rule=20,
        two_tower_candidates_per_rule=20,
        external_candidate_mode="replace",
        external_candidate_slots=20,
        external_replacement_requires_history=False,
    )
    external = [
        *[(f"c{i:03d}", float(i)) for i in range(399, 389, -1)],
        *[(f"c{i:03d}", float(i)) for i in range(499, 489, -1)],
    ]

    ungated, _ = build_personalized_candidate_pools(
        profiles,
        contents,
        membership,
        popularity,
        config=RetrievalConfig(**common),
        external_rankings={"p1": external},
    )
    gated, metrics, audit = build_personalized_candidate_pools(
        profiles,
        contents,
        membership,
        popularity,
        config=RetrievalConfig(
            **common,
            external_replacement_requires_witness=True,
        ),
        external_rankings={"p1": external},
        capture_blend_audit=True,
    )

    assert {f"c{i:03d}" for i in range(390, 400)} <= set(
        ungated["content_id"]
    )
    assert {f"c{i:03d}" for i in range(490, 500)} <= set(
        ungated["content_id"]
    )
    assert not ({f"c{i:03d}" for i in range(490, 500)} & set(gated["content_id"]))
    assert {f"c{i:03d}" for i in range(390, 400)} <= set(gated["content_id"])
    assert gated["content_id"].nunique() == 200
    assert not gated["is_safety_anchor"].isna().any()
    assert metrics["external_replacement_requires_witness"] is True
    assert metrics["external_witness_rejected_candidates"] == 10
    assert metrics["external_witness_rejected_profiles"] == 1
    assert audit.profiles.iloc[0]["witness_rejected_count"] == 10
    assert audit.profiles.iloc[0]["lost_safety_anchor_count"] == 0


def test_leave_one_out_history_and_ranking_map_metrics() -> None:
    history = np.asarray([[1, 2, 1], [3, 4, 5]], dtype=np.int64)
    masked = _leave_one_out_history(history, np.asarray([1, 4]))
    np.testing.assert_array_equal(masked, [[0, 2, 0], [3, 0, 5]])
    np.testing.assert_array_equal(history, [[1, 2, 1], [3, 4, 5]])

    truth = pd.DataFrame(
        {
            "profile_id": ["p1", "p1"],
            "content_id": ["a", "b"],
            "completion_gain": [1.0, 0.25],
            "watch_time_gain": [0.5, 1.0],
        }
    )
    metrics = ranking_map_recall_metrics(
        {"p1": [("a", 1.0), ("other", 0.0)]}, truth, 2
    )
    assert metrics["external_candidate_recall@2"] == 0.5
    assert metrics["external_completion_gain_recall@2"] == 0.8
    assert metrics["external_watch_time_gain_recall@2"] == pytest.approx(1 / 3)
    incremental = ranking_map_incremental_metrics(
        {"p1": [("a", 1.0), ("b", 0.5)]},
        pd.DataFrame({"profile_id": ["p1"], "content_id": ["a"]}),
        truth,
        2,
    )
    assert incremental["external_incremental_truth_hits"] == 1
    assert incremental["external_oracle_union_recall@2+incumbent"] == 1.0


def test_model_training_uses_both_graded_targets_without_classification(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "profile_id": [f"p{index % 4}" for index in range(24)],
            "content_id": [f"c{index % 8}" for index in range(24)],
            "type_id": [str(index % 2) for index in range(24)],
            "tag_names": ["movie,kids" if index % 2 else "music" for index in range(24)],
            "historical_interaction_count": list(range(1, 25)),
            "label": [1 if index % 5 == 0 else 0 for index in range(24)],
            "completion_gain": [1.0 if index % 5 == 0 else 0.0 for index in range(24)],
            "watch_time_gain": [0.7 if index % 5 == 0 else 0.0 for index in range(24)],
            "target_semantics": ["engagement_v1"] * 24,
            "sample_weight": [1.0 if index % 5 == 0 else 0.25 for index in range(24)],
            "sample_type": [
                "observed_engagement" if index % 5 == 0 else "unobserved_sample"
                for index in range(24)
            ],
        }
    )
    config = TrainingConfig(
        model_family="embedding_mlp_baseline",
        epochs=1,
        patience=1,
        batch_size=4,
        hidden_dims=(8,),
        embedding_dim=4,
        numeric_projection_dim=4,
    )

    _, _, metrics = train_model(
        frame,
        frame.iloc[:8].copy(),
        "all",
        tmp_path,
        config,
    )

    assert metrics["target_columns"] == ["completion_gain", "watch_time_gain"]
    assert metrics["target_semantics"] == "engagement_v1"
    assert metrics["classification_metrics_valid"] is False
    assert set(metrics["validation_objective"]) == {
        "completion_gain",
        "watch_time_gain",
    }
    assert "validation" not in metrics
