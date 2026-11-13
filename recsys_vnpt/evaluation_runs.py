"""Offline evaluation runs.

Builds the serving candidate set for a fixed temporal truth, then scores it with
either the popularity baseline or a trained neural checkpoint. Both entry points
share `_serving_candidates_for_evaluation` and the same manifest config, so the
two are comparable by construction.
"""


import hashlib
import json
import pandas as pd
import time

from dataclasses import asdict
from pathlib import Path
from recsys_vnpt.canonical import select_content_metadata_as_of
from recsys_vnpt.content_scorer import ContentScorerModel, train_content_scorer
from recsys_vnpt.engagement import (
    GAIN_COLUMNS,
    historical_sequence_features,
)
from recsys_vnpt.evaluation import evaluate_scored_candidates
from recsys_vnpt.hybrid_retrieval import (
    HybridRetrievalConfig,
    train_and_rank_hybrid_retriever,
)
from recsys_vnpt.implicit_baselines import (
    IMPLICIT_BASELINE_FAMILIES,
    ImplicitBaselineConfig,
    score_implicit_baseline,
)
from recsys_vnpt.metrics_schema import MetricBundle
from recsys_vnpt.metrics_store import (
    MetricsStore,
    create_manifest,
    render_markdown,
)
from recsys_vnpt.model_runner import (
    load_v2_checkpoint,
    predict,
    prepare_inference_input,
)
from recsys_vnpt.paths import (
    data_artifact_path,
    model_artifact_path,
)
from recsys_vnpt.prepare import _read_canonical_interactions
from recsys_vnpt.retrieval import (
    CandidateBlendAudit,
    RetrievalConfig,
    build_personalized_candidate_pools,
    candidate_gain_recall_by_profile,
    candidate_gain_recall_curve,
    candidate_recall_by_profile,
    candidate_recall_curve,
    fuse_content_scores_with_external_rank,
    historical_content_popularity,
    pair_keys_hash,
    prepare_retrieval_truth,
    profile_rule_affinity_scores,
    ranking_map_recall_metrics,
    retrievable_content_ids,
    retrieval_truth_attribution,
)
from recsys_vnpt.rule_assembly import (
    RuleAssemblyConfig,
    aggregate_model_rule_scores,
)
from recsys_vnpt.orderings import (
    FcnModel,
    OrderingConfig,
    apply_orderings,
    load_fcn_models,
    ordering_parts,
)
from recsys_vnpt.rule_scorer import RuleScorerModel, train_rule_scorer
from recsys_vnpt.runtime import _partition_serving_interaction_scopes
from recsys_vnpt.sequence_retrieval import (
    SASRecLiteConfig,
    train_and_rank_sasrec_lite,
)
from recsys_vnpt.source_manifest import (
    sha256_file,
    stable_hash,
)
from recsys_vnpt.temporal import (
    build_pair_features,
    historical_profile_features,
    load_temporal_windows,
    temporal_artifact_root,
)
from typing import Any


def _evaluation_seen_entities(
    temporal: Path,
    evaluation_split: str,
) -> tuple[set[str], set[str], dict[str, Any]]:
    if evaluation_split not in {"validation", "test"}:
        raise ValueError("evaluation_split must be validation or test")
    window_names = ["train_1", "train_2", "train_3"]
    if evaluation_split == "test":
        window_names.append("validation")
    observed_frames: list[pd.DataFrame] = []
    loaded_windows: list[str] = []
    for window_name in window_names:
        path = temporal / f"{window_name}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(
            path,
            columns=["profile_id", "content_id", "sample_type"],
        )
        frame = frame[frame["sample_type"].astype(str).eq("observed_engagement")]
        if frame.empty:
            continue
        observed_frames.append(frame[["profile_id", "content_id"]])
        loaded_windows.append(window_name)
    if not observed_frames:
        return set(), set(), {
            "seen_history_windows": loaded_windows,
            "seen_history_observed_pairs": 0,
            "seen_history_profiles": 0,
            "seen_history_contents": 0,
        }
    observed = pd.concat(observed_frames, ignore_index=True).drop_duplicates()
    seen_profiles = set(observed["profile_id"].dropna().astype(str))
    seen_contents = set(observed["content_id"].dropna().astype(str))
    return seen_profiles, seen_contents, {
        "seen_history_windows": loaded_windows,
        "seen_history_observed_pairs": int(len(observed)),
        "seen_history_profiles": int(len(seen_profiles)),
        "seen_history_contents": int(len(seen_contents)),
        "seen_history_semantics": "observed_engagement_only",
    }


def _serving_candidates_for_evaluation(
    root: Path,
    domain: str,
    truth_candidates: pd.DataFrame,
    candidate_budget: int,
    cutoff: str | pd.Timestamp,
    exact_cowatch: bool = False,
    hybrid_retrieval: bool = False,
    hybrid_config: HybridRetrievalConfig | None = None,
    sequence_mode: str = "none",
    sequence_config: SASRecLiteConfig | None = None,
    external_candidate_mode: str = "rerank",
    external_candidate_slots: int = 100,
    external_rrf_weight: float = 0.25,
    external_replacement_requires_history: bool = True,
    external_replacement_requires_witness: bool = False,
    materialize_model_features: bool = True,
    evaluation_split: str = "test",
    recency_candidates_per_rule: int = 0,
    recency_reserved_per_rule: int = 0,
    recency_max_age_days: int = 0,
    cowatch_candidates_per_rule: int = 0,
    tag_candidates_per_rule: int = 0,
    candidate_rules: int | None = None,
    reserved_per_rule: int | None = None,
    source_candidates_per_rule: int | None = None,
    rule_affinity: str = "count",
    rule_affinity_half_life_days: float | None = None,
    rule_ordering: str = "count",
    rule_scorer_fusion_weight: float = 1.25,
    content_ordering: str = "hand_sum",
    content_scorer_fusion_weight: float = 5.0,
    return_blend_audit: bool = False,
    rule_scorer_model: RuleScorerModel | None = None,
    content_scorer_model: ContentScorerModel | None = None,
    gbdt_backend: str = "sklearn",
    gbdt_device: str = "auto",
    content_rank_scope: str = "rule",
    content_fcn_family: str = "fcn_official",
    routed_fallback: str = "learned_rrf",
    fcn_model_dir: str | Path | None = None,
    fcn_models: dict[str, FcnModel] | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
    pd.Series,
] | tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
    pd.Series,
    CandidateBlendAudit,
]:
    serving_started = time.perf_counter()
    catalog_load_started = time.perf_counter()
    canonical = data_artifact_path("canonical", domain, project_root=root)
    contents = pd.read_parquet(canonical / "contents.parquet")
    membership = pd.read_parquet(canonical / "content_rule_membership.parquet")
    mapping = pd.read_parquet(canonical / "profile_mapping.parquet")
    users = pd.read_parquet(canonical / "users.parquet")
    cutoff = pd.Timestamp(cutoff)
    versions_path = canonical / "content_versions.parquet"
    metadata_backfill_contents = 0
    if versions_path.exists():
        feature_items = select_content_metadata_as_of(
            pd.read_parquet(versions_path), cutoff
        )
        metadata_backfill_contents = int(
            feature_items.get(
                "metadata_backfill", pd.Series(False, index=feature_items.index)
            ).fillna(False).astype(bool).sum()
        )
        eligible_ids = set(feature_items["content_id"].astype(str))
        # Retrieval keeps tag_names so the profile_tag_affinity channel works;
        # tags share the documented backfill provenance of rule membership and
        # publish_date. Model FEATURES still exclude tag text below.
        retrieval_contents = contents[
            contents["content_id"].astype(str).isin(eligible_ids)
        ].copy()
        feature_items = feature_items.drop(columns=["tag_names"], errors="ignore")
    else:
        feature_items = contents
        retrieval_contents = contents
    catalog_load_seconds = time.perf_counter() - catalog_load_started
    interaction_load_started = time.perf_counter()
    retrieval_content_ids = set(retrieval_contents["content_id"].astype(str))
    positive_mask = (
        pd.to_numeric(truth_candidates["label"], errors="coerce")
        .fillna(0)
        .eq(1)
        if "label" in truth_candidates
        else pd.Series(True, index=truth_candidates.index)
    )
    truth_content_ids = set(
        truth_candidates.loc[positive_mask, "content_id"].dropna().astype(str)
    )
    # Repeat-positive auditing must also see history for positives that are no
    # longer retrievable. The bounded union avoids loading the whole source.
    loaded_interactions = _read_canonical_interactions(
        canonical / "interactions.parquet",
        content_ids=retrieval_content_ids | truth_content_ids,
    )
    interactions, truth_audit_interactions, interaction_scope_metrics = (
        _partition_serving_interaction_scopes(
            loaded_interactions, retrieval_content_ids
        )
    )
    interaction_load_seconds = time.perf_counter() - interaction_load_started
    profiles = (
        truth_candidates[["profile_id"]]
        .drop_duplicates()
        .astype({"profile_id": str})
    )
    candidate_generation_started = time.perf_counter()
    popularity = historical_content_popularity(interactions, cutoff=cutoff)
    retrieval_config = RetrievalConfig(
        global_candidates=candidate_budget,
        candidate_budget=candidate_budget,
        exact_cowatch_enabled=exact_cowatch,
        external_candidate_mode=external_candidate_mode,
        external_candidate_slots=min(external_candidate_slots, candidate_budget),
        external_rrf_weight=external_rrf_weight,
        external_replacement_requires_history=(
            external_replacement_requires_history
        ),
        external_replacement_requires_witness=(
            external_replacement_requires_witness
        ),
        recency_candidates_per_rule=recency_candidates_per_rule,
        recency_reserved_per_rule=recency_reserved_per_rule,
        recency_max_age_days=recency_max_age_days,
        cowatch_candidates_per_rule=cowatch_candidates_per_rule,
        tag_candidates_per_rule=tag_candidates_per_rule,
        # Left at the dataclass default unless a caller asks: raising them trips
        # the guard candidate_rules * reserved_per_rule <= candidate_budget.
        **{
            name: value
            for name, value in (
                ("candidate_rules", candidate_rules),
                ("reserved_per_rule", reserved_per_rule),
                ("source_candidates_per_rule", source_candidates_per_rule),
            )
            if value is not None
        },
    )
    retrievable_ids = retrievable_content_ids(
        retrieval_contents, membership, popularity, retrieval_config
    )
    if sequence_mode not in {"none", "historical"}:
        raise ValueError("sequence_mode must be none or historical")
    external_rankings = None
    external_source = "hybrid_two_tower"
    hybrid_metrics: dict[str, Any] = {}
    sequence_metrics: dict[str, Any] = {}
    playback_data: pd.DataFrame | None = None
    if hybrid_retrieval:
        playback_path = canonical / "playback_events.parquet"
        if not playback_path.exists():
            raise FileNotFoundError(
                "hybrid retrieval requires canonical playback_events.parquet"
            )
        profile_metadata = mapping.merge(users, on="username", how="left")
        rule_tokens = (
            membership[["content_id", "rule_id"]]
            .drop_duplicates()
            .groupby("content_id")["rule_id"]
            .agg(lambda values: ",".join(sorted(set(map(str, values)))))
            .rename("rule_ids")
        )
        hybrid_contents = retrieval_contents.merge(
            rule_tokens, left_on="content_id", right_index=True, how="left"
        )
        playback_data = pd.read_parquet(playback_path)
        hybrid_result = train_and_rank_hybrid_retriever(
            playback_data,
            hybrid_contents,
            profiles,
            cutoff=cutoff,
            candidate_budget=candidate_budget,
            profile_metadata=profile_metadata,
            config=hybrid_config or HybridRetrievalConfig(),
            output_dir=model_artifact_path(
                domain,
                "retrieval",
                f"hybrid_{cutoff.strftime('%Y%m%d')}",
                project_root=root,
            ),
            eligible_content_ids=retrievable_ids,
        )
        external_rankings = hybrid_result.rankings
        hybrid_metrics = hybrid_result.metrics
    if sequence_mode == "historical":
        playback_path = canonical / "playback_events.parquet"
        if playback_data is None:
            playback_data = pd.read_parquet(playback_path)
        sequence_result = train_and_rank_sasrec_lite(
            playback_data,
            retrieval_contents,
            profiles,
            cutoff=cutoff,
            candidate_budget=candidate_budget,
            config=sequence_config or SASRecLiteConfig(),
            output_dir=model_artifact_path(
                domain,
                "retrieval",
                f"sasrec_{cutoff.strftime('%Y%m%d')}",
                project_root=root,
            ),
        )
        sequence_metrics = sequence_result.metrics
        if external_rankings is None:
            external_rankings = sequence_result.rankings
            external_source = "sasrec_lite"
        else:
            blended: dict[str, list[tuple[str, float]]] = {}
            for profile_id in sorted(
                set(external_rankings) | set(sequence_result.rankings)
            ):
                sources = (
                    (external_rankings.get(profile_id, []), 0.7),
                    (sequence_result.rankings.get(profile_id, []), 0.3),
                )
                combined: dict[str, float] = {}
                for ranking, weight in sources:
                    values = [float(score) for _, score in ranking]
                    minimum = min(values, default=0.0)
                    maximum = max(values, default=0.0)
                    for content_id, score in ranking:
                        normalized = (
                            (float(score) - minimum) / (maximum - minimum)
                            if maximum > minimum
                            else 1.0
                        )
                        combined[str(content_id)] = (
                            combined.get(str(content_id), 0.0)
                            + weight * normalized
                        )
                blended[profile_id] = sorted(
                    combined.items(), key=lambda pair: (-pair[1], pair[0])
                )[:candidate_budget]
            external_rankings = blended
            external_source = "hybrid_two_tower+sasrec_lite"
    truth_policy = prepare_retrieval_truth(
        truth_candidates,
        truth_audit_interactions,
        retrievable_ids,
        cutoff=cutoff,
    )
    novel_content = truth_policy.novel["content_id"].astype(str)
    in_as_of_catalog = novel_content.isin(retrieval_content_ids)
    in_retrievable_policy = novel_content.isin(retrievable_ids)
    truth_policy.metrics.update(
        {
            "unavailable_absent_from_as_of_catalog_pairs": int(
                (~in_as_of_catalog).sum()
            ),
            "unavailable_without_eligible_rule_pairs": int(
                (in_as_of_catalog & ~in_retrievable_policy).sum()
            ),
        }
    )
    built_candidates = build_personalized_candidate_pools(
        profiles,
        retrieval_contents,
        membership,
        popularity,
        interactions=interactions,
        cutoff=cutoff,
        config=retrieval_config,
        external_rankings=external_rankings,
        external_source=external_source,
        capture_blend_audit=return_blend_audit,
        evaluation_truth=truth_policy.eligible,
    )
    if return_blend_audit:
        candidate_pairs, retrieval_metrics, blend_audit = built_candidates
    else:
        candidate_pairs, retrieval_metrics = built_candidates
        blend_audit = None
    if hybrid_metrics:
        retrieval_metrics["hybrid_retrieval"] = hybrid_metrics
    if sequence_metrics:
        retrieval_metrics["sequence_retrieval"] = sequence_metrics
    retrieval_metrics.update(
        {
            "metadata_as_of_contents": int(len(feature_items)),
            "metadata_backfill_contents": metadata_backfill_contents,
            "metadata_backfill_rate": metadata_backfill_contents
            / max(len(feature_items), 1),
        }
    )
    def interactions_loader(content_ids, columns):
        return _read_canonical_interactions(
            canonical / "interactions.parquet", content_ids=content_ids, columns=columns
        )

    rule_affinity_started = time.perf_counter()
    ordering = OrderingConfig(
        rule_ordering=rule_ordering,
        content_ordering=content_ordering,
        content_rank_scope=content_rank_scope,
        content_fcn_family=content_fcn_family,
        routed_fallback=routed_fallback,
        rule_scorer_fusion_weight=rule_scorer_fusion_weight,
        content_scorer_fusion_weight=content_scorer_fusion_weight,
    )
    # Rule ordering only; candidate pool construction keeps the incumbent
    # affinity so candidate_pairs_hash stays comparable across variants.
    profile_rule_scores = profile_rule_affinity_scores(
        profiles,
        membership,
        popularity,
        interactions=interactions,
        cutoff=cutoff,
        affinity=rule_affinity,
        half_life_days=rule_affinity_half_life_days,
    )
    # A persisted model is applied as is; otherwise it is fitted on the
    # split's pre-cutoff windows.
    if ordering.rule_gbdt and rule_scorer_model is None:
        rule_scorer_model = train_rule_scorer(
            canonical, interactions_loader, evaluation_split=evaluation_split,
            backend=gbdt_backend, device=gbdt_device,
        )
    if ordering.content_gbdt and content_scorer_model is None:
        content_scorer_model = train_content_scorer(
            canonical, cutoff, interactions_loader, evaluation_split=evaluation_split,
            backend=gbdt_backend, device=gbdt_device,
        )
    if fcn_models is None:
        fcn_models = load_fcn_models(
            fcn_model_dir or model_artifact_path(domain, "v2", project_root=root),
            ordering.fcn_families,
        )
    candidate_pairs, profile_rule_scores, ordering_metrics = apply_orderings(
        ordering,
        ordering_parts(
            ordering, canonical, cutoff, interactions_loader, interactions, popularity,
            rule_model=rule_scorer_model, content_model=content_scorer_model, fcn=fcn_models,
        ),
        profiles["profile_id"].astype(str),
        candidate_pairs,
        profile_rule_scores,
        membership,
    )
    retrieval_metrics.update(ordering_metrics)
    rule_affinity_seconds = time.perf_counter() - rule_affinity_started
    candidate_generation_seconds = (
        time.perf_counter() - candidate_generation_started
    )
    feature_materialization_started = time.perf_counter()
    if materialize_model_features:
        candidate_pairs = candidate_pairs.merge(
            mapping[["profile_id", "username"]].astype({"profile_id": str}),
            on="profile_id",
            how="left",
        )
        feature_frame = build_pair_features(
            candidate_pairs,
            interactions,
            feature_items,
            users,
            cutoff,
        )
        playback_path = canonical / "playback_events.parquet"
        if playback_path.exists():
            sequence_features = historical_sequence_features(
                pd.read_parquet(playback_path), cutoff
            )
            feature_frame = feature_frame.merge(
                sequence_features, on="profile_id", how="left"
            )
            for column in (
                "historical_session_count",
                "historical_playback_count",
                "historical_sequence_depth",
            ):
                feature_frame[column] = pd.to_numeric(
                    feature_frame[column], errors="coerce"
                ).fillna(0.0)
    else:
        minimal_item_columns = [
            column
            for column in ("content_id", "type_id", "content_family")
            if column in feature_items
        ]
        feature_frame = candidate_pairs.merge(
            feature_items[minimal_item_columns].drop_duplicates(
                "content_id", keep="last"
            ),
            on="content_id",
            how="left",
        )
        candidate_pairs = feature_frame
    feature_frame["rule_id"] = feature_frame["candidate_rule_id"].astype("string")
    retrieval_metrics["model_features_materialized"] = bool(
        materialize_model_features
    )
    retrieval_metrics["serving_candidate_columns"] = int(
        len(feature_frame.columns)
    )
    if external_rankings:
        retrieval_metrics.update(
            ranking_map_recall_metrics(
                external_rankings,
                truth_policy.eligible,
                candidate_budget,
            )
        )
    positive_keys = set(
        truth_policy.eligible[["profile_id", "content_id"]].itertuples(
            index=False, name=None
        )
    )
    feature_frame["label"] = [
        int((str(profile_id), str(content_id)) in positive_keys)
        for profile_id, content_id in feature_frame[
            ["profile_id", "content_id"]
        ].itertuples(index=False, name=None)
    ]
    feature_frame["sample_type"] = "serving_candidate"
    feature_frame["fallback"] = False

    seen_profiles, seen_contents, seen_metrics = _evaluation_seen_entities(
        temporal_artifact_root(canonical), evaluation_split
    )
    if seen_profiles or seen_contents:
        feature_frame["observed_history_profile_oov"] = ~feature_frame[
            "profile_id"
        ].astype(str).isin(
            seen_profiles
        )
        feature_frame["observed_history_content_oov"] = ~feature_frame[
            "content_id"
        ].astype(str).isin(
            seen_contents
        )
        feature_frame["profile_oov"] = feature_frame[
            "observed_history_profile_oov"
        ]
        feature_frame["content_oov"] = feature_frame[
            "observed_history_content_oov"
        ]
    historical_counts = historical_profile_features(interactions, cutoff)
    count_map = historical_counts.set_index(
        historical_counts["profile_id"].astype(str)
    )["historical_interaction_count"]
    activity = feature_frame["profile_id"].astype(str).map(count_map).fillna(0)
    feature_frame["activity_bucket"] = pd.cut(
        activity,
        bins=[-1, 0, 5, 20, float("inf")],
        labels=["cold", "low", "medium", "high"],
    ).astype("string")
    truth_attributes = truth_policy.eligible[["profile_id", "content_id"]].copy()
    truth_attributes["profile_id"] = truth_attributes["profile_id"].astype(str)
    truth_attributes["content_id"] = truth_attributes["content_id"].astype(str)
    truth_attributes["observed_history_profile_oov"] = ~truth_attributes[
        "profile_id"
    ].isin(seen_profiles)
    truth_attributes["observed_history_content_oov"] = ~truth_attributes[
        "content_id"
    ].isin(seen_contents)
    truth_activity = truth_attributes["profile_id"].map(count_map).fillna(0)
    truth_attributes["activity_bucket"] = pd.cut(
        truth_activity,
        bins=[-1, 0, 5, 20, float("inf")],
        labels=["cold", "low", "medium", "high"],
    ).astype("string")
    item_attribute_columns = [
        column
        for column in ("content_id", "content_family", "type_id")
        if column in feature_items
    ]
    if len(item_attribute_columns) > 1:
        item_attributes = feature_items[item_attribute_columns].copy()
        item_attributes["content_id"] = item_attributes["content_id"].astype(str)
        truth_attributes = truth_attributes.merge(
            item_attributes.drop_duplicates("content_id", keep="last"),
            on="content_id",
            how="left",
        )
    truth_attribution = retrieval_truth_attribution(
        candidate_pairs,
        truth_policy.eligible,
        truth_attributes,
    )
    feature_materialization_seconds = (
        time.perf_counter() - feature_materialization_started
    )

    truth_metrics_started = time.perf_counter()
    all_positive_recall, _ = candidate_recall_by_profile(
        candidate_pairs, truth_policy.raw
    )
    novel_positive_recall, _ = candidate_recall_by_profile(
        candidate_pairs, truth_policy.novel
    )
    micro_recall, per_profile_recall = candidate_recall_by_profile(
        candidate_pairs, truth_policy.eligible
    )
    retrieval_metrics.update(
        {
            **interaction_scope_metrics,
            **truth_policy.metrics,
            **candidate_recall_curve(
                candidate_pairs,
                truth_policy.eligible,
                k_values=(100, 200, 500, candidate_budget),
            ),
            f"candidate_recall@{candidate_budget}": micro_recall,
            "candidate_recall@actual_k": micro_recall,
            "candidate_recall_all_positives": all_positive_recall,
            "candidate_recall_novel_positives": novel_positive_recall,
            "evaluation_truth_pairs_hash": pair_keys_hash(
                truth_policy.eligible
            ),
            "evaluation_truth_policy": "novel_retrievable_v1",
            "positive_hits_by_source": {
                source: int(values["truth_hits"])
                for source, values in truth_attribution[
                    "primary_source"
                ].items()
            },
            "retrieval_truth_attribution": truth_attribution,
            **seen_metrics,
            "unavailable_positive_pairs": truth_policy.metrics[
                "unavailable_novel_positive_pairs"
            ],
            "profiles_with_zero_positive_candidate": int(
                sum(value == 0 for value in per_profile_recall.values())
            ),
            "rule_membership_backfill": True,
        }
    )
    for gain_column in GAIN_COLUMNS:
        if gain_column in truth_policy.eligible:
            retrieval_metrics.update(
                candidate_gain_recall_curve(
                    candidate_pairs,
                    truth_policy.eligible,
                    gain_column,
                    k_values=(100, 200, 500, candidate_budget),
                )
            )
    gain_hash_columns = [
        column
        for column in ("profile_id", "content_id", *GAIN_COLUMNS)
        if column in truth_policy.eligible
    ]
    gain_hash_frame = truth_policy.eligible[gain_hash_columns].sort_values(
        ["profile_id", "content_id"], kind="stable"
    )
    retrieval_metrics["evaluation_truth_gain_hash"] = stable_hash(
        gain_hash_frame.to_dict("records")
    )
    truth_metrics_seconds = time.perf_counter() - truth_metrics_started
    retrieval_metrics["serving_candidate_pipeline_time_seconds"] = (
        time.perf_counter() - serving_started
    )
    retrieval_metrics["serving_stage_timings"] = {
        "catalog_load_seconds": catalog_load_seconds,
        "interaction_load_seconds": interaction_load_seconds,
        "candidate_generation_seconds": candidate_generation_seconds,
        "feature_materialization_seconds": feature_materialization_seconds,
        "truth_metrics_seconds": truth_metrics_seconds,
        "rule_affinity_seconds": rule_affinity_seconds,
    }
    result = (
        feature_frame,
        profile_rule_scores,
        retrieval_metrics,
        truth_policy.eligible,
        popularity,
    )
    if return_blend_audit:
        if blend_audit is None:
            raise RuntimeError("blend audit was requested but not captured")
        return (*result, blend_audit)
    return result


def _materialize_fixed_candidate_features(
    root: Path,
    domain: str,
    candidates: pd.DataFrame,
    cutoff: str | pd.Timestamp,
) -> pd.DataFrame:
    """Materialize features only for the explicitly requested fixed-pool path."""
    canonical = data_artifact_path("canonical", domain, project_root=root)
    cutoff_value = pd.Timestamp(cutoff)
    interactions = _read_canonical_interactions(
        canonical / "interactions.parquet",
        content_ids=set(candidates["content_id"].astype(str)),
    )
    versions_path = canonical / "content_versions.parquet"
    items = (
        select_content_metadata_as_of(
            pd.read_parquet(versions_path), cutoff_value
        )
        if versions_path.exists()
        else pd.read_parquet(canonical / "contents.parquet")
    )
    users = pd.read_parquet(canonical / "users.parquet")
    work = candidates.copy()
    if "username" not in work:
        mapping = pd.read_parquet(canonical / "profile_mapping.parquet")
        work = work.merge(
            mapping[["profile_id", "username"]], on="profile_id", how="left"
        )
    output = build_pair_features(
        work, interactions, items, users, cutoff_value
    )
    playback_path = canonical / "playback_events.parquet"
    if playback_path.exists():
        sequence_features = historical_sequence_features(
            pd.read_parquet(playback_path), cutoff_value
        )
        output = output.merge(sequence_features, on="profile_id", how="left")
        for column in (
            "historical_session_count",
            "historical_playback_count",
            "historical_sequence_depth",
        ):
            output[column] = pd.to_numeric(
                output[column], errors="coerce"
            ).fillna(0.0)
    return output


def scoring_manifest_config(
    *,
    evaluation_split,
    maximum_evaluation_profiles,
    evaluation_profile_seed,
    rule_affinity,
    rule_affinity_half_life_days,
    rule_ordering,
    rule_scorer_fusion_weight,
    content_ordering,
    content_scorer_fusion_weight,
    contents_per_rule_max,
    page_content_budget,
    page_dedup,
    candidate_rules,
    reserved_per_rule,
    source_candidates_per_rule,
    gbdt_backend,
    gbdt_device,
    content_rank_scope,
    content_fcn_family,
    routed_fallback,
    fcn_model_dir,
) -> dict[str, Any]:
    """The config that decides which page a run was scored on.

    Two evaluation paths write manifests and both have to record this, or a
    number cannot be traced back to the page it came from. The neural path used
    to record nine unrelated fields and drop all of these.
    """

    return {
        "evaluation_split": evaluation_split,
        "maximum_evaluation_profiles": maximum_evaluation_profiles,
        "evaluation_profile_seed": evaluation_profile_seed,
        "rule_affinity": rule_affinity,
        "rule_affinity_half_life_days": rule_affinity_half_life_days,
        "rule_ordering": rule_ordering,
        "rule_scorer_fusion_weight": rule_scorer_fusion_weight,
        "content_ordering": content_ordering,
        "content_scorer_fusion_weight": content_scorer_fusion_weight,
        "contents_per_rule_max": contents_per_rule_max,
        "page_content_budget": page_content_budget,
        "page_dedup": page_dedup,
        "candidate_rules": candidate_rules,
        "reserved_per_rule": reserved_per_rule,
        "source_candidates_per_rule": source_candidates_per_rule,
        "gbdt_backend": gbdt_backend,
        "gbdt_device": gbdt_device,
        "content_rank_scope": content_rank_scope,
        "content_fcn_family": content_fcn_family,
        "routed_fallback": routed_fallback,
        "fcn_model_dir": None if fcn_model_dir is None else str(fcn_model_dir),
    }


def evaluate_temporal_model(
    project_root: str | Path = ".",
    domain: str = "all",
    model_family: str = "fcn_official",
    run_id: str | None = None,
    batch_size: int = 4096,
    amp: bool = False,
    eval_serving_candidates: bool = True,
    candidate_budget: int = 1000,
    exact_cowatch: bool = False,
    hybrid_retrieval: bool = False,
    hybrid_epochs: int = 5,
    hybrid_embedding_dim: int = 64,
    seed: int = 42,
    sequence_mode: str = "none",
    sequence_epochs: int = 3,
    evaluation_split: str = "test",
    maximum_evaluation_profiles: int | None = None,
    evaluation_profile_seed: int = 42,
    rule_affinity: str = "count",
    rule_affinity_half_life_days: float | None = None,
    rule_ordering: str = "count",
    rule_scorer_fusion_weight: float = 1.25,
    content_ordering: str = "hand_sum",
    content_scorer_fusion_weight: float = 5.0,
    contents_per_rule_max: int | None = None,
    page_content_budget: int | None = None,
    page_dedup: str = "none",
    candidate_rules: int | None = None,
    reserved_per_rule: int | None = None,
    source_candidates_per_rule: int | None = None,
    recency_candidates_per_rule: int = 0,
    recency_reserved_per_rule: int = 0,
    recency_max_age_days: int = 0,
    gbdt_backend: str = "sklearn",
    gbdt_device: str = "auto",
    content_rank_scope: str = "rule",
    content_fcn_family: str = "fcn_official",
    routed_fallback: str = "learned_rrf",
    fcn_model_dir: str | Path | None = None,
) -> MetricBundle:
    root = Path(project_root).resolve()
    model_output = model_artifact_path(domain, "v2", model_family, project_root=root)
    checkpoint_path = model_output / "best_model_v2.pth"
    contract_path = model_output / "feature_contract_v2.json"
    if not checkpoint_path.exists() or not contract_path.exists():
        raise FileNotFoundError(
            f"Missing {model_family} checkpoint/contract for {domain}. Run pipeline_cli train first."
        )
    temporal = data_artifact_path("canonical", domain, "temporal", project_root=root)
    # Shares the popularity path's selection so both evaluate the same split
    # on the same profiles against that split's own truth.
    truth_candidates, evaluation_cutoff, selection_metrics = (
        _evaluation_truth_selection(
            temporal,
            evaluation_split,
            maximum_evaluation_profiles,
            evaluation_profile_seed,
        )
    )
    retrieval_started = time.perf_counter()
    if eval_serving_candidates:
        (
            candidates,
            profile_rule_scores,
            retrieval_metrics,
            evaluation_truth,
            popularity,
        ) = (
            _serving_candidates_for_evaluation(
                root,
                domain,
                truth_candidates,
                candidate_budget=candidate_budget,
                cutoff=evaluation_cutoff,
                exact_cowatch=exact_cowatch,
                hybrid_retrieval=hybrid_retrieval,
                hybrid_config=HybridRetrievalConfig(
                    epochs=hybrid_epochs,
                    embedding_dim=hybrid_embedding_dim,
                    seed=seed,
                ),
                sequence_mode=sequence_mode,
                sequence_config=SASRecLiteConfig(
                    epochs=sequence_epochs,
                    seed=seed,
                ),
                evaluation_split=evaluation_split,
                recency_candidates_per_rule=recency_candidates_per_rule,
                recency_reserved_per_rule=recency_reserved_per_rule,
                recency_max_age_days=recency_max_age_days,
                gbdt_backend=gbdt_backend,
                gbdt_device=gbdt_device,
                candidate_rules=candidate_rules,
                reserved_per_rule=reserved_per_rule,
                source_candidates_per_rule=source_candidates_per_rule,
                rule_affinity=rule_affinity,
                rule_affinity_half_life_days=rule_affinity_half_life_days,
                rule_ordering=rule_ordering,
                rule_scorer_fusion_weight=rule_scorer_fusion_weight,
                content_ordering=content_ordering,
                content_scorer_fusion_weight=content_scorer_fusion_weight,
                content_rank_scope=content_rank_scope,
                content_fcn_family=content_fcn_family,
                routed_fallback=routed_fallback,
                fcn_model_dir=fcn_model_dir,
            )
        )
    else:
        candidates = _materialize_fixed_candidate_features(
            root,
            domain,
            truth_candidates,
            load_temporal_windows(
                data_artifact_path(
                    "canonical", domain, "temporal", project_root=root
                )
            )[4].history_end,
        )
        evaluation_truth = truth_candidates
        profile_rule_scores = pd.DataFrame()
        counts = candidates.groupby("profile_id")["content_id"].nunique()
        retrieval_metrics = {
            "candidate_recall@actual_k": 1.0,
            "candidate_recall@1000": 1.0,
            "candidate_pairs": int(len(candidates)),
            "mean_candidates_per_profile": float(counts.mean()),
            "max_candidates_per_profile": int(counts.max()),
        }
        interaction_ids = set(candidates["content_id"].astype(str))
        fallback_interactions = _read_canonical_interactions(
            data_artifact_path(
                "canonical", domain, "interactions.parquet", project_root=root
            ),
            content_ids=interaction_ids,
        )
        popularity = historical_content_popularity(
            fallback_interactions,
            cutoff=load_temporal_windows(
                data_artifact_path(
                    "canonical", domain, "temporal", project_root=root
                )
            )[4].history_end,
        )
    retrieval_seconds = time.perf_counter() - retrieval_started
    membership = pd.read_parquet(
        data_artifact_path("canonical", domain, "content_rule_membership.parquet", project_root=root)
    )
    model, contract, checkpoint = load_v2_checkpoint(checkpoint_path, contract_path)
    prepared_candidates = prepare_inference_input(
        contract, candidates, model_family
    )
    started = time.perf_counter()
    scores = predict(
        model,
        contract,
        candidates,
        model_family,
        batch_size=batch_size,
        amp=amp,
        prepared=prepared_candidates,
    )
    inference_seconds = time.perf_counter() - started
    scored = candidates.copy()
    scored["score"] = scores
    if model_family == "hierarchical_fcn":
        scored["model_rule_score"] = predict(
            model,
            contract,
            candidates,
            model_family,
            batch_size=batch_size,
            amp=amp,
            output_key="rule_score",
            prepared=prepared_candidates,
        )
        model_rule_scores = aggregate_model_rule_scores(
            scored,
            membership,
        )
        profile_rule_scores = pd.concat(
            [profile_rule_scores, model_rule_scores], ignore_index=True
        ).drop_duplicates(["profile_id", "rule_id"], keep="last")
    ranking_started = time.perf_counter()
    assembly_config = RuleAssemblyConfig(
        contents_per_rule_max=contents_per_rule_max,
        page_content_budget=page_content_budget,
        page_dedup=page_dedup,
    )
    evaluation_metrics, _ = evaluate_scored_candidates(
        scored,
        membership,
        popularity,
        ground_truth_frame=evaluation_truth,
        profile_rule_scores=profile_rule_scores,
        assembly_config=assembly_config,
    )
    ranking_seconds = time.perf_counter() - ranking_started
    evaluation_metrics.update(retrieval_metrics)
    profile_retrieval_metrics = _profile_retrieval_metric_frame(
        candidates, evaluation_truth, candidate_budget
    )
    evaluation_metrics["retrieval_profile_metrics_rows"] = int(
        len(profile_retrieval_metrics)
    )

    store = MetricsStore(root)
    if run_id is not None:
        bundle = store.read_bundle(run_id, domain)
        if bundle.manifest.model_family != model_family:
            raise ValueError("metrics run model family does not match evaluated checkpoint")
    else:
        manifest = create_manifest(
            root,
            domain,
            "sprint-2" if model_family == "dense_repaired_baseline" else "sprint-3",
            model_family,
            hashes={"feature_contract": contract.artifact_hash},
            config={
                "batch_size": batch_size,
                "amp": amp,
                "eval_serving_candidates": eval_serving_candidates,
                "candidate_budget": candidate_budget,
                "exact_cowatch": exact_cowatch,
                "hybrid_retrieval": hybrid_retrieval,
                "hybrid_epochs": hybrid_epochs,
                "hybrid_embedding_dim": hybrid_embedding_dim,
                "sequence_mode": sequence_mode,
                "sequence_epochs": sequence_epochs,
                "recency_candidates_per_rule": recency_candidates_per_rule,
                "recency_reserved_per_rule": recency_reserved_per_rule,
                "recency_max_age_days": recency_max_age_days,
                **scoring_manifest_config(
                    evaluation_split=evaluation_split,
                    maximum_evaluation_profiles=maximum_evaluation_profiles,
                    evaluation_profile_seed=evaluation_profile_seed,
                    rule_affinity=rule_affinity,
                    rule_affinity_half_life_days=rule_affinity_half_life_days,
                    rule_ordering=rule_ordering,
                    rule_scorer_fusion_weight=rule_scorer_fusion_weight,
                    content_ordering=content_ordering,
                    content_scorer_fusion_weight=content_scorer_fusion_weight,
                    contents_per_rule_max=contents_per_rule_max,
                    page_content_budget=page_content_budget,
                    page_dedup=page_dedup,
                    candidate_rules=candidate_rules,
                    reserved_per_rule=reserved_per_rule,
                    source_candidates_per_rule=source_candidates_per_rule,
                    gbdt_backend=gbdt_backend,
                    gbdt_device=gbdt_device,
                    content_rank_scope=content_rank_scope,
                    content_fcn_family=content_fcn_family,
                    routed_fallback=routed_fallback,
                    fcn_model_dir=fcn_model_dir,
                ),
            },
        )
        bundle = MetricBundle(manifest=manifest)
    bundle.evaluation = evaluation_metrics
    bundle.inference.update(
        {
            "evaluation_inference_time_seconds": inference_seconds,
            "evaluation_retrieval_time_seconds": retrieval_seconds,
            "evaluation_ranking_time_seconds": ranking_seconds,
            "evaluation_model_path_time_seconds": (
                retrieval_seconds + inference_seconds
            ),
            "evaluation_end_to_end_time_seconds": (
                retrieval_seconds + inference_seconds + ranking_seconds
            ),
            "inference_pairs": int(len(scored)),
            "inference_seconds_per_pair": inference_seconds / max(len(scored), 1),
            "pairs_per_second": len(scored) / max(inference_seconds, 1e-12),
            "batch_size": batch_size,
            "amp": amp,
            "checkpoint_epoch": int(checkpoint["epoch"]),
        }
    )
    directory = store.write_bundle(bundle)
    report = render_markdown(bundle)
    (directory / "report.md").write_text(report, encoding="utf-8")
    return bundle


def _profile_retrieval_metric_frame(
    candidates: pd.DataFrame,
    evaluation_truth: pd.DataFrame,
    candidate_budget: int,
) -> pd.DataFrame:
    _, binary_recall_by_profile = candidate_recall_by_profile(
        candidates, evaluation_truth
    )
    output = pd.DataFrame({"profile_id": sorted(binary_recall_by_profile)})
    output["candidate_recall"] = output["profile_id"].map(
        binary_recall_by_profile
    )
    for gain_column in GAIN_COLUMNS:
        if gain_column not in evaluation_truth:
            continue
        _, gain_by_profile = candidate_gain_recall_by_profile(
            candidates,
            evaluation_truth,
            gain_column,
            k=candidate_budget,
        )
        output[f"candidate_{gain_column}_recall"] = output["profile_id"].map(
            gain_by_profile
        )
    cohort_columns = [
        column
        for column in ("profile_id", "activity_bucket", "profile_oov")
        if column in candidates
    ]
    if len(cohort_columns) > 1:
        output = output.merge(
            candidates[cohort_columns].drop_duplicates("profile_id"),
            on="profile_id",
            how="left",
        )
    return output


def _evaluation_truth_selection(
    temporal: Path,
    split: str,
    maximum_profiles: int | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.Timestamp, dict[str, Any]]:
    if split not in {"validation", "test"}:
        raise ValueError("evaluation split must be validation or test")
    if maximum_profiles is not None and maximum_profiles < 1:
        raise ValueError("maximum evaluation profiles must be positive")
    window = next(
        window
        for window in load_temporal_windows(temporal)
        if window.name == split
    )
    truth = pd.read_parquet(temporal / f"{split}_candidates.parquet")
    positive_mask = pd.to_numeric(
        truth.get("label", pd.Series(1, index=truth.index)), errors="coerce"
    ).fillna(0).eq(1)
    profile_ids = sorted(truth.loc[positive_mask, "profile_id"].dropna().astype(str).unique())
    if maximum_profiles is not None and len(profile_ids) > maximum_profiles:
        profile_ids = sorted(
            profile_ids,
            key=lambda profile_id: hashlib.sha256(
                f"{seed}|{profile_id}".encode("utf-8")
            ).hexdigest(),
        )[:maximum_profiles]
        selected = set(profile_ids)
        truth = truth[truth["profile_id"].astype(str).isin(selected)].copy()
    selection_hash = hashlib.sha256(
        "\n".join(sorted(profile_ids)).encode("utf-8")
    ).hexdigest()
    return truth, window.history_end, {
        "evaluation_split": split,
        "evaluation_profile_cap": maximum_profiles,
        "evaluation_profile_seed": seed,
        "evaluation_profiles_selected": int(len(profile_ids)),
        "evaluation_profile_selection_hash": selection_hash,
        "evaluation_history_cutoff": str(window.history_end),
    }


_EVALUATION_DATA_METRIC_KEYS = {
    "candidate_truth_catalog_coverage",
    "candidate_truth_completion_gain_coverage",
    "candidate_truth_policy_coverage",
    "candidate_truth_watch_time_gain_coverage",
    "evaluation_history_cutoff",
    "evaluation_profile_cap",
    "evaluation_profile_seed",
    "evaluation_profile_selection_hash",
    "evaluation_profiles_selected",
    "evaluation_split",
    "evaluation_truth_gain_hash",
    "evaluation_truth_pairs_hash",
    "evaluation_truth_policy",
    "metadata_as_of_contents",
    "metadata_backfill_contents",
    "metadata_backfill_rate",
    "novel_completion_gain_sum",
    "novel_positive_pairs",
    "novel_watch_time_gain_sum",
    "raw_completion_gain_sum",
    "raw_positive_pairs",
    "raw_truth_profiles",
    "raw_watch_time_gain_sum",
    "repeat_positive_pairs_excluded",
    "retrievable_completion_gain_sum",
    "retrievable_positive_pairs",
    "retrievable_truth_profiles",
    "retrievable_watch_time_gain_sum",
    "rule_membership_backfill",
    "truth_profiles_excluded_by_policy",
    "unavailable_absent_from_as_of_catalog_pairs",
    "unavailable_novel_positive_pairs",
    "unavailable_positive_pairs",
    "unavailable_without_eligible_rule_pairs",
}


def _evaluation_run_data_metrics(
    temporal: Path,
    retrieval_metrics: dict[str, Any],
) -> dict[str, Any]:
    temporal_metrics_path = temporal / "data_metrics.json"
    if temporal_metrics_path.exists():
        payload = json.loads(temporal_metrics_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("temporal data_metrics.json must contain an object")
        data_metrics = dict(payload)
    else:
        data_metrics = {}
    data_metrics.update(
        {
            key: retrieval_metrics[key]
            for key in _EVALUATION_DATA_METRIC_KEYS
            if key in retrieval_metrics
        }
    )
    cache_metadata_path = temporal / "cache_metadata.json"
    if cache_metadata_path.exists():
        cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
        fingerprint = cache_metadata.get("fingerprint")
        if fingerprint:
            data_metrics["temporal_data_fingerprint"] = str(fingerprint)
    return data_metrics


def evaluate_popularity_baseline(
    project_root: str | Path = ".",
    domain: str = "all",
    eval_serving_candidates: bool = True,
    candidate_budget: int = 1000,
    exact_cowatch: bool = False,
    baseline_family: str = "graded_per_rule_popularity",
    epochs: int = 10,
    embedding_dim: int = 32,
    batch_size: int = 8192,
    fusion_bpr_weight: float = 1.0,
    seed: int = 42,
    device: str | None = None,
    hybrid_retrieval: bool = False,
    hybrid_epochs: int = 5,
    hybrid_embedding_dim: int = 64,
    hybrid_holdout_days: int | None = None,
    hybrid_item_identity_policy: str = "target_only",
    hybrid_scoring_catalog_policy: str = "all_input",
    sequence_mode: str = "none",
    sequence_epochs: int = 3,
    evaluation_split: str = "test",
    maximum_evaluation_profiles: int | None = None,
    evaluation_profile_seed: int = 42,
    external_candidate_mode: str = "rerank",
    external_candidate_slots: int = 100,
    external_rrf_weight: float = 0.25,
    external_replacement_requires_history: bool = True,
    external_replacement_requires_witness: bool = False,
    content_external_rrf_weight: float = 0.0,
    content_rrf_constant: int = 60,
    recency_candidates_per_rule: int = 0,
    recency_reserved_per_rule: int = 0,
    recency_max_age_days: int = 0,
    cowatch_candidates_per_rule: int = 0,
    tag_candidates_per_rule: int = 0,
    candidate_rules: int | None = None,
    reserved_per_rule: int | None = None,
    source_candidates_per_rule: int | None = None,
    rule_affinity: str = "count",
    rule_affinity_half_life_days: float | None = None,
    rule_ordering: str = "count",
    rule_scorer_fusion_weight: float = 1.25,
    content_ordering: str = "hand_sum",
    content_scorer_fusion_weight: float = 5.0,
    contents_per_rule_max: int | None = None,
    page_content_budget: int | None = None,
    page_dedup: str = "none",
    rule_scorer_model: RuleScorerModel | None = None,
    content_scorer_model: ContentScorerModel | None = None,
    gbdt_backend: str = "sklearn",
    gbdt_device: str = "auto",
    content_rank_scope: str = "rule",
    content_fcn_family: str = "fcn_official",
    routed_fallback: str = "learned_rrf",
    fcn_model_dir: str | Path | None = None,
    fcn_models: dict[str, FcnModel] | None = None,
) -> MetricBundle:
    """Offline evaluation of the popularity path on fixed temporal truth.

    Learned orderings are fitted on the split's pre-cutoff windows unless a
    persisted ``*_scorer_model`` is given, in which case it is applied as is.
    """
    if baseline_family == "per_rule_popularity":
        baseline_family = "graded_per_rule_popularity"
    if baseline_family not in IMPLICIT_BASELINE_FAMILIES:
        raise ValueError(f"unsupported baseline family: {baseline_family}")
    root = Path(project_root).resolve()
    canonical = data_artifact_path("canonical", domain, project_root=root)
    temporal = temporal_artifact_root(canonical)
    truth_candidates, evaluation_cutoff, selection_metrics = _evaluation_truth_selection(
        temporal,
        evaluation_split,
        maximum_evaluation_profiles,
        evaluation_profile_seed,
    )
    effective_hybrid_holdout_days = (
        hybrid_holdout_days
        if hybrid_holdout_days is not None
        else (11 if evaluation_split == "validation" else 10)
    )
    retrieval_started = time.perf_counter()
    if eval_serving_candidates:
        (
            candidates,
            profile_rule_scores,
            retrieval_metrics,
            evaluation_truth,
            popularity,
            blend_audit,
        ) = (
            _serving_candidates_for_evaluation(
                root,
                domain,
                truth_candidates,
                candidate_budget=candidate_budget,
                cutoff=evaluation_cutoff,
                exact_cowatch=exact_cowatch,
                hybrid_retrieval=hybrid_retrieval,
                hybrid_config=HybridRetrievalConfig(
                    epochs=hybrid_epochs,
                    embedding_dim=hybrid_embedding_dim,
                    temporal_holdout_days=effective_hybrid_holdout_days,
                    item_identity_policy=hybrid_item_identity_policy,
                    scoring_catalog_policy=hybrid_scoring_catalog_policy,
                    seed=seed,
                ),
                sequence_mode=sequence_mode,
                sequence_config=SASRecLiteConfig(
                    epochs=sequence_epochs,
                    seed=seed,
                ),
                external_candidate_mode=external_candidate_mode,
                external_candidate_slots=external_candidate_slots,
                external_rrf_weight=external_rrf_weight,
                external_replacement_requires_history=(
                    external_replacement_requires_history
                ),
                external_replacement_requires_witness=(
                    external_replacement_requires_witness
                ),
                materialize_model_features=False,
                evaluation_split=evaluation_split,
                recency_candidates_per_rule=recency_candidates_per_rule,
                recency_reserved_per_rule=recency_reserved_per_rule,
                recency_max_age_days=recency_max_age_days,
                cowatch_candidates_per_rule=cowatch_candidates_per_rule,
                tag_candidates_per_rule=tag_candidates_per_rule,
                candidate_rules=candidate_rules,
                reserved_per_rule=reserved_per_rule,
                source_candidates_per_rule=source_candidates_per_rule,
                rule_affinity=rule_affinity,
                rule_affinity_half_life_days=rule_affinity_half_life_days,
                rule_ordering=rule_ordering,
                rule_scorer_fusion_weight=rule_scorer_fusion_weight,
                content_ordering=content_ordering,
                content_scorer_fusion_weight=content_scorer_fusion_weight,
                return_blend_audit=True,
                rule_scorer_model=rule_scorer_model,
                content_scorer_model=content_scorer_model,
                gbdt_backend=gbdt_backend,
                gbdt_device=gbdt_device,
                content_rank_scope=content_rank_scope,
                content_fcn_family=content_fcn_family,
                routed_fallback=routed_fallback,
                fcn_model_dir=fcn_model_dir,
                fcn_models=fcn_models,
            )
        )
    else:
        candidates = truth_candidates
        evaluation_truth = truth_candidates
        profile_rule_scores = pd.DataFrame()
        retrieval_metrics = {
            "candidate_recall@actual_k": 1.0,
            "candidate_recall@1000": 1.0,
        }
        fallback_interactions = _read_canonical_interactions(
            canonical / "interactions.parquet",
            content_ids=set(candidates["content_id"].astype(str)),
        )
        popularity = historical_content_popularity(
            fallback_interactions, cutoff=evaluation_cutoff
        )
        blend_audit = CandidateBlendAudit(
            lost_incumbents=pd.DataFrame(
                columns=["profile_id", "content_id"]
            ),
            profiles=pd.DataFrame(
                {"profile_id": candidates["profile_id"].astype(str).unique()}
            ),
        )
    retrieval_metrics.update(selection_metrics)
    retrieval_seconds = time.perf_counter() - retrieval_started
    membership = pd.read_parquet(canonical / "content_rule_membership.parquet")
    scored = candidates.copy()
    baseline_config = ImplicitBaselineConfig(
        family=baseline_family,
        epochs=epochs,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        fusion_bpr_weight=fusion_bpr_weight,
        seed=seed,
    )
    baseline_result = score_implicit_baseline(
        pd.read_parquet(canonical / "playback_events.parquet"),
        scored,
        cutoff=evaluation_cutoff,
        config=baseline_config,
        output_dir=model_artifact_path(
            domain, "implicit", baseline_family, project_root=root
        ),
        device=device,
    )
    scored["base_score"] = baseline_result.scores
    fusion_started = time.perf_counter()
    fused_scores, score_fusion_metrics = fuse_content_scores_with_external_rank(
        scored,
        scored["base_score"].to_numpy(dtype=float),
        external_weight=content_external_rrf_weight,
        rrf_constant=content_rrf_constant,
    )
    score_fusion_seconds = time.perf_counter() - fusion_started
    if (
        content_external_rrf_weight > 0
        and score_fusion_metrics["content_external_rank_coverage"] <= 0
    ):
        raise ValueError(
            "content external RRF fusion requires external retrieval rankings"
        )
    scored["score"] = fused_scores
    inference_seconds = (
        float(baseline_result.metrics["scoring_time_seconds"])
        + score_fusion_seconds
    )
    ranking_started = time.perf_counter()
    assembly_config = RuleAssemblyConfig(
        contents_per_rule_max=contents_per_rule_max,
        page_content_budget=page_content_budget,
        page_dedup=page_dedup,
    )
    evaluation_metrics, _ = evaluate_scored_candidates(
        scored,
        membership,
        popularity,
        ground_truth_frame=evaluation_truth,
        profile_rule_scores=profile_rule_scores,
        assembly_config=assembly_config,
    )
    ranking_seconds = time.perf_counter() - ranking_started
    evaluation_metrics.update(retrieval_metrics)
    evaluation_metrics.update(score_fusion_metrics)
    profile_retrieval_metrics = _profile_retrieval_metric_frame(
        candidates, evaluation_truth, candidate_budget
    )
    evaluation_metrics["retrieval_profile_metrics_rows"] = int(
        len(profile_retrieval_metrics)
    )
    data_metrics = _evaluation_run_data_metrics(temporal, retrieval_metrics)
    manifest_hashes = {
        name: str(value)
        for name, value in {
            "candidate_pairs": evaluation_metrics.get("candidate_pairs_hash"),
            "evaluation_truth_pairs": evaluation_metrics.get(
                "evaluation_truth_pairs_hash"
            ),
            "evaluation_truth_gain": evaluation_metrics.get(
                "evaluation_truth_gain_hash"
            ),
            "evaluation_profile_selection": evaluation_metrics.get(
                "evaluation_profile_selection_hash"
            ),
            "temporal_data": data_metrics.get("temporal_data_fingerprint"),
            "label_contract": data_metrics.get("label_contract_hash"),
            "preblend_incumbent_pair_set": evaluation_metrics.get(
                "preblend_incumbent_pair_set_hash"
            ),
            "blend_gained_pair_set": evaluation_metrics.get(
                "blend_gained_pair_set_hash"
            ),
            "blend_lost_pair_set": evaluation_metrics.get(
                "blend_lost_pair_set_hash"
            ),
            "blend_union_pair_set": evaluation_metrics.get(
                "blend_union_pair_set_hash"
            ),
        }.items()
        if value
    }
    manifest = create_manifest(
        root,
        domain,
        "label-l2",
        baseline_family,
        seed=seed,
        hashes=manifest_hashes,
        split={
            evaluation_split: {
                "history_end": str(evaluation_cutoff),
            }
        },
        config={
            "candidate_budget": candidate_budget,
            "eval_serving_candidates": eval_serving_candidates,
            "exact_cowatch": exact_cowatch,
            "hybrid_retrieval": hybrid_retrieval,
            "hybrid_epochs": hybrid_epochs,
            "hybrid_embedding_dim": hybrid_embedding_dim,
            "hybrid_holdout_days": effective_hybrid_holdout_days,
            "hybrid_item_identity_policy": hybrid_item_identity_policy,
            "hybrid_scoring_catalog_policy": hybrid_scoring_catalog_policy,
            "sequence_mode": sequence_mode,
            "sequence_epochs": sequence_epochs,
            "evaluation_split": evaluation_split,
            "maximum_evaluation_profiles": maximum_evaluation_profiles,
            "evaluation_profile_seed": evaluation_profile_seed,
            "external_candidate_mode": external_candidate_mode,
            "external_candidate_slots": external_candidate_slots,
            "external_rrf_weight": external_rrf_weight,
            "external_replacement_requires_history": (
                external_replacement_requires_history
            ),
            "external_replacement_requires_witness": (
                external_replacement_requires_witness
            ),
            "content_external_rrf_weight": content_external_rrf_weight,
            "content_rrf_constant": content_rrf_constant,
            "recency_candidates_per_rule": recency_candidates_per_rule,
            "recency_reserved_per_rule": recency_reserved_per_rule,
            "recency_max_age_days": recency_max_age_days,
            "cowatch_candidates_per_rule": cowatch_candidates_per_rule,
            "tag_candidates_per_rule": tag_candidates_per_rule,
            "candidate_rules": candidate_rules,
            "reserved_per_rule": reserved_per_rule,
            "source_candidates_per_rule": source_candidates_per_rule,
            "rule_affinity": rule_affinity,
            "rule_affinity_half_life_days": rule_affinity_half_life_days,
            "rule_ordering": rule_ordering,
            "rule_scorer_fusion_weight": rule_scorer_fusion_weight,
            "content_ordering": content_ordering,
            "content_scorer_fusion_weight": content_scorer_fusion_weight,
            "contents_per_rule_max": contents_per_rule_max,
            "page_content_budget": page_content_budget,
            "page_dedup": page_dedup,
            "gbdt_backend": gbdt_backend,
            "gbdt_device": gbdt_device,
            "content_rank_scope": content_rank_scope,
            "content_fcn_family": content_fcn_family,
            "routed_fallback": routed_fallback,
            "fcn_model_dir": None if fcn_model_dir is None else str(fcn_model_dir),
            **asdict(baseline_config),
        },
    )
    bundle = MetricBundle(
        manifest=manifest,
        data=data_metrics,
        training=baseline_result.metrics,
        evaluation=evaluation_metrics,
        inference={
            "evaluation_inference_time_seconds": inference_seconds,
            "evaluation_score_fusion_time_seconds": score_fusion_seconds,
            "evaluation_retrieval_time_seconds": retrieval_seconds,
            "evaluation_ranking_time_seconds": ranking_seconds,
            "evaluation_model_path_time_seconds": (
                retrieval_seconds + inference_seconds
            ),
            "evaluation_end_to_end_time_seconds": (
                retrieval_seconds + inference_seconds + ranking_seconds
            ),
            "inference_pairs": int(len(scored)),
            "inference_seconds_per_pair": inference_seconds / max(len(scored), 1),
            "pairs_per_second": len(scored) / max(inference_seconds, 1e-12),
        },
    )
    store = MetricsStore(root)
    directory = store.write_bundle(bundle)
    candidate_artifact_columns = [
        column
        for column in (
            "profile_id",
            "content_id",
            "candidate_rank",
            "retrieval_source",
            "retrieval_sources",
            "candidate_rule_id",
            "retrieval_score",
            "rule_score",
            "incumbent_rank",
            "external_rank",
            "external_fusion_score",
            "is_incumbent_candidate",
            "is_external_candidate",
            "is_safety_anchor",
            "activity_bucket",
            "profile_oov",
            "content_oov",
            "observed_history_profile_oov",
            "observed_history_content_oov",
            "type_id",
            "base_score",
            "score",
        )
        if column in scored
    ]
    candidate_artifact_path = directory / "retrieval_candidates.parquet"
    truth_artifact_path = directory / "evaluation_truth.parquet"
    profile_metrics_path = directory / "retrieval_profile_metrics.parquet"
    blend_lost_path = directory / "retrieval_preblend_lost.parquet"
    blend_profiles_path = directory / "retrieval_blend_profiles.parquet"
    scored[candidate_artifact_columns].to_parquet(candidate_artifact_path, index=False)
    evaluation_truth.to_parquet(truth_artifact_path, index=False)
    profile_retrieval_metrics.to_parquet(profile_metrics_path, index=False)
    blend_audit.lost_incumbents.to_parquet(blend_lost_path, index=False)
    blend_audit.profiles.to_parquet(blend_profiles_path, index=False)
    artifact_hashes = {
        "retrieval_candidates_artifact": sha256_file(candidate_artifact_path),
        "evaluation_truth_artifact": sha256_file(truth_artifact_path),
        "retrieval_profile_metrics_artifact": sha256_file(profile_metrics_path),
        "retrieval_preblend_lost_artifact": sha256_file(blend_lost_path),
        "retrieval_blend_profiles_artifact": sha256_file(blend_profiles_path),
    }
    bundle.manifest.hashes.update(artifact_hashes)
    bundle.evaluation.update(
        {
            "retrieval_candidates_artifact_sha256": artifact_hashes[
                "retrieval_candidates_artifact"
            ],
            "evaluation_truth_artifact_sha256": artifact_hashes[
                "evaluation_truth_artifact"
            ],
            "retrieval_profile_metrics_artifact_sha256": artifact_hashes[
                "retrieval_profile_metrics_artifact"
            ],
            "retrieval_preblend_lost_artifact_sha256": artifact_hashes[
                "retrieval_preblend_lost_artifact"
            ],
            "retrieval_blend_profiles_artifact_sha256": artifact_hashes[
                "retrieval_blend_profiles_artifact"
            ],
        }
    )
    store.write_bundle(bundle)
    (directory / "report.md").write_text(render_markdown(bundle), encoding="utf-8")
    return bundle
