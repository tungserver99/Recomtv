"""The serving path: build a real page and write the client files.

This is the only stage that publishes. It runs the same retrieval and the exact
assembly the evaluation stage measures, then validates and atomically writes the
three client files.
"""



def _train_temporal_model():
    """Deferred import: `pipeline_v2` re-exports this module, so importing the
    trainer at module scope would close a cycle."""

    from recsys_vnpt.pipeline_v2 import train_temporal_model

    return train_temporal_model


import hashlib
import json
import numpy as np
import pandas as pd
import time

from dataclasses import dataclass, replace
from pathlib import Path
from recsys_vnpt.bootstrap import ensure_checkpoint
from recsys_vnpt.content_scorer import (
    ContentScorerModel,
    ContentServingScope,
    apply_content_scorer,
    content_serving_scope,
    train_content_scorer,
)
from recsys_vnpt.fcn_scorer import FCNScorerModel, score_fcn_candidates
from recsys_vnpt.scoring_config import ScoringConfig, resolve_scoring_config, scoring_config_payload
from recsys_vnpt.evaluation import evaluate_scored_candidates
from recsys_vnpt.evaluation_runs import _serving_candidates_for_evaluation
from recsys_vnpt.export import (
    atomic_write_client_outputs,
    serialize_assembled_output,
    write_production_named_outputs,
)
from recsys_vnpt.hardware import NvidiaSmiSampler
from recsys_vnpt.metrics_schema import MetricBundle
from recsys_vnpt.metrics_store import (
    MetricsStore,
    create_manifest,
    evaluate_gate,
    render_markdown,
)
from recsys_vnpt.model_runner import (
    load_v2_checkpoint,
    predict,
    prepare_inference_input,
)
from recsys_vnpt.output_validation import validate_client_output
from recsys_vnpt.paths import (
    data_artifact_path,
    model_artifact_path,
)
from recsys_vnpt.prepare import _read_canonical_interactions
from recsys_vnpt.retrieval import (
    RetrievalConfig,
    balanced_catalog_cap,
    RetrievalIndex,
    build_personalized_candidate_pools,
    historical_content_popularity,
    prepare_retrieval_index,
    profile_rule_affinity_scores,
)
from recsys_vnpt.rule_assembly import (
    RuleAssemblyConfig,
    aggregate_model_rule_scores,
    assemble_recommendations,
    bayesian_rule_popularity,
    explode_content_rule_membership,
)
from recsys_vnpt.rule_scorer import (
    RuleScorerModel,
    RuleServingScope,
    apply_rule_scorer,
    rule_serving_scope,
    train_rule_scorer,
)
from recsys_vnpt.runtime import _rss_bytes
from recsys_vnpt.temporal import (
    build_pair_features,
    historical_profile_features,
    load_temporal_windows,
)
from typing import Any


def _member_id_map(profiles: pd.DataFrame) -> dict[str, str]:
    """Profile -> owning member, from the canonical mapping.

    A member owns one or more profiles and only the default profile carries the
    member id as its own id, so the grouping file must join rather than copy
    `pid`. Blank, zero, and self-equal rows are dropped: the writer already
    falls back to `pid`, so carrying them would only inflate the map.
    """

    if "member_id" not in profiles.columns:
        return {}
    frame = profiles[["profile_id", "member_id"]].dropna()
    pid = frame["profile_id"].astype("string").str.strip()
    mid = frame["member_id"].astype("string").str.strip()
    usable = mid.ne("") & mid.ne("0") & mid.ne(pid)
    return dict(zip(pid[usable], mid[usable]))


def _fallback_canonical_paths(root: Path, domain: str) -> tuple[Path, Path, Path]:
    canonical = data_artifact_path("canonical", domain, project_root=root)
    return canonical / "contents.parquet", canonical / "content_rule_membership.parquet", canonical / "profile_mapping.parquet"


def _interactions_loader(canonical: Path):
    return lambda content_ids, columns: _read_canonical_interactions(
        canonical / "interactions.parquet",
        content_ids=content_ids,
        columns=columns,
    )


def serving_retrieval_config(
    candidate_budget: int = 1000,
    exact_cowatch: bool = False,
    recency_candidates_per_rule: int = 0,
    recency_reserved_per_rule: int = 0,
    recency_max_age_days: int = 0,
    candidate_rules: int = 40,
    reserved_per_rule: int = 10,
    source_candidates_per_rule: int = 20,
) -> RetrievalConfig:
    return RetrievalConfig(
        global_candidates=candidate_budget,
        candidate_budget=candidate_budget,
        exact_cowatch_enabled=exact_cowatch,
        recency_candidates_per_rule=recency_candidates_per_rule,
        recency_reserved_per_rule=recency_reserved_per_rule,
        recency_max_age_days=recency_max_age_days,
        candidate_rules=candidate_rules,
        reserved_per_rule=reserved_per_rule,
        source_candidates_per_rule=source_candidates_per_rule,
    )


@dataclass(frozen=True)
class ServingContext:
    """Everything a page depends on except which profiles are served.

    Loaded once per process and shared by every profile shard, so per-shard
    work is proportional to the shard, not to the whole history.
    """

    domain: str
    canonical: Path
    contents: pd.DataFrame
    membership: pd.DataFrame
    full_membership: pd.DataFrame
    profiles: pd.DataFrame
    users: pd.DataFrame
    interactions: pd.DataFrame
    cutoff: pd.Timestamp
    popularity: pd.Series
    retrieval_config: RetrievalConfig
    retrieval_index: RetrievalIndex
    rule_popularity: dict[str, float]
    assembly_config: RuleAssemblyConfig
    content_types: pd.DataFrame
    catalog: dict[str, dict[str, Any]]
    valid_membership: set[tuple[str, str]]
    rule_model: RuleScorerModel | None = None
    rule_scope: RuleServingScope | None = None
    rule_scorer_fusion_weight: float = 1.25
    content_model: ContentScorerModel | None = None
    content_scope: ContentServingScope | None = None
    content_scorer_fusion_weight: float = 5.0
    scoring_config: ScoringConfig | None = None
    fcn_scorer: FCNScorerModel | None = None
    fcn_batch_size: int = 4096


def load_serving_context(
    root: Path,
    domain: str,
    retrieval_config: RetrievalConfig,
    assembly_config: RuleAssemblyConfig,
    num_contents: int | None = None,
) -> ServingContext:
    content_path, membership_path, profile_path = _fallback_canonical_paths(root, domain)
    if not all(path.exists() for path in (content_path, membership_path, profile_path)):
        raise FileNotFoundError("Canonical artifacts are missing. Run pipeline_cli prepare first.")
    canonical = data_artifact_path("canonical", domain, project_root=root)
    contents = pd.read_parquet(content_path)
    membership = pd.read_parquet(membership_path)
    full_membership = membership.copy()
    profiles = pd.read_parquet(profile_path).drop_duplicates("profile_id")
    interactions = _read_canonical_interactions(
        canonical / "interactions.parquet",
        content_ids=set(contents["content_id"].astype(str)),
    )
    users_path = canonical / "users.parquet"
    if not users_path.exists():
        raise FileNotFoundError(f"Missing {users_path}. Run pipeline_cli prepare first.")
    users = pd.read_parquet(users_path)
    cutoff = pd.to_datetime(interactions["create_date"], errors="coerce").max()
    if pd.isna(cutoff):
        raise ValueError("Canonical interactions contain no valid create_date")
    popularity = historical_content_popularity(interactions, cutoff=cutoff)
    contents, membership = balanced_catalog_cap(
        contents, membership, popularity, maximum_contents=num_contents
    )
    return ServingContext(
        domain=domain,
        canonical=canonical,
        contents=contents,
        membership=membership,
        full_membership=full_membership,
        profiles=profiles,
        users=users,
        interactions=interactions,
        cutoff=pd.Timestamp(cutoff),
        popularity=popularity,
        retrieval_config=retrieval_config,
        retrieval_index=prepare_retrieval_index(
            contents, membership, popularity, interactions, cutoff, retrieval_config
        ),
        rule_popularity=bayesian_rule_popularity(popularity, membership),
        assembly_config=assembly_config,
        content_types=contents[["content_id", "type_id"]].drop_duplicates(
            "content_id", keep="last"
        ),
        catalog={
            str(row.content_id): {
                "type_id": str(row.type_id),
                "content_status": str(getattr(row, "content_status", "1")),
                "recommendable": bool(getattr(row, "recommendable", True)),
            }
            for row in contents.itertuples(index=False)
        },
        valid_membership=set(
            membership[["content_id", "rule_id"]]
            .astype("string")
            .itertuples(index=False, name=None)
        ),
    )


def with_scorers(
    context: ServingContext,
    rule_model: RuleScorerModel | None,
    content_model: ContentScorerModel | None,
    rule_scorer_fusion_weight: float = 1.25,
    content_scorer_fusion_weight: float = 5.0,
    scoring_config: ScoringConfig | None = None,
    fcn_scorer: FCNScorerModel | None = None,
    fcn_batch_size: int = 4096,
) -> ServingContext:
    """Attach fitted scorers and their serving scopes at the context cutoff."""
    loader = _interactions_loader(context.canonical)
    return replace(
        context,
        rule_model=rule_model,
        rule_scope=rule_serving_scope(context.canonical, context.cutoff, rule_model, loader)
        if rule_model is not None
        else None,
        rule_scorer_fusion_weight=rule_scorer_fusion_weight,
        content_model=content_model,
        content_scope=content_serving_scope(
            context.canonical, context.cutoff, content_model, loader
        )
        if content_model is not None
        else None,
        content_scorer_fusion_weight=content_scorer_fusion_weight,
        scoring_config=scoring_config
        or resolve_scoring_config(legacy_fusion_weight=content_scorer_fusion_weight),
        fcn_scorer=fcn_scorer,
        fcn_batch_size=fcn_batch_size,
    )


def _score_fcn_candidates(
    context: ServingContext,
    candidates: pd.DataFrame,
    scorer: FCNScorerModel,
    chunk_size: int = 50_000,
    batch_size: int = 4096,
) -> np.ndarray:
    """Build the existing cutoff-safe pair features and score in bounded chunks."""
    from recsys_vnpt.canonical import select_content_metadata_as_of
    from recsys_vnpt.engagement import historical_sequence_features

    profiles = context.profiles
    profile_columns = [
        column for column in ("profile_id", "username") if column in profiles
    ]
    items = context.contents
    versions = context.canonical / "content_versions.parquet"
    if versions.is_file():
        items = select_content_metadata_as_of(
            pd.read_parquet(versions), context.cutoff
        ).drop(columns=["tag_names"], errors="ignore")
    sequence_enabled = "historical_content_sequence" in scorer.contract.multivalue_features
    playback_path = context.canonical / "playback_events.parquet"
    sequence = None
    if sequence_enabled and playback_path.is_file():
        sequence = historical_sequence_features(
            pd.read_parquet(playback_path), context.cutoff
        )

    scores: list[np.ndarray] = []
    for start in range(0, len(candidates), chunk_size):
        part = candidates.iloc[start : start + chunk_size].copy()
        pair_features = part.merge(
            profiles[profile_columns].drop_duplicates("profile_id"),
            on="profile_id",
            how="left",
        )
        frame = build_pair_features(
            pair_features,
            context.interactions,
            items,
            context.users,
            context.cutoff,
        )
        if sequence is not None:
            frame = frame.merge(sequence, on="profile_id", how="left")
            for column in (
                "historical_session_count",
                "historical_playback_count",
                "historical_sequence_depth",
            ):
                if column in frame:
                    frame[column] = pd.to_numeric(
                        frame[column], errors="coerce"
                    ).fillna(0.0)
        frame["rule_id"] = frame["candidate_rule_id"].astype("string")
        scores.append(score_fcn_candidates(scorer, frame, batch_size=batch_size))
    return np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)


def order_candidates(
    context: ServingContext, profiles: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Retrieve candidates for ``profiles`` and order rules and contents."""
    candidate_pairs, metrics = build_personalized_candidate_pools(
        profiles,
        context.contents,
        context.membership,
        context.popularity,
        interactions=context.interactions,
        cutoff=context.cutoff,
        config=context.retrieval_config,
        index=context.retrieval_index,
    )
    history = context.retrieval_index.history
    rule_scores = profile_rule_affinity_scores(
        profiles,
        context.membership,
        context.popularity,
        interactions=history[
            history["profile_id"].isin(set(profiles["profile_id"].astype(str)))
        ],
        cutoff=context.cutoff,
    )
    if context.rule_model is not None:
        rule_scores, rule_metrics = apply_rule_scorer(
            context.rule_model,
            context.rule_scope,
            profiles["profile_id"].astype(str),
            rule_scores,
            context.rule_scorer_fusion_weight,
        )
        metrics.update(rule_metrics)
    scoring_config = context.scoring_config or resolve_scoring_config()
    fcn_scores = (
        _score_fcn_candidates(
            context,
            candidate_pairs,
            context.fcn_scorer,
            batch_size=context.fcn_batch_size,
        )
        if context.fcn_scorer is not None
        else None
    )
    if (
        context.content_model is not None
        or context.fcn_scorer is not None
        or scoring_config.preset != "legacy"
    ):
        candidate_pairs, content_metrics = apply_content_scorer(
            context.content_model,
            context.content_scope,
            candidate_pairs,
            context.content_scorer_fusion_weight,
            scoring_config=scoring_config,
            fcn_scores=fcn_scores,
        )
        metrics.update(content_metrics)
    return candidate_pairs, rule_scores, metrics


def _assemble_chunk(
    pairs: pd.DataFrame,
    profile_rule_scores: pd.DataFrame,
    membership: pd.DataFrame,
    rule_popularity: dict[str, float],
    config: RuleAssemblyConfig,
) -> dict[str, dict[str, Any]]:
    expanded = explode_content_rule_membership(
        pairs[["profile_id", "content_id", "type_id", "score"]],
        membership,
    )
    expanded = expanded.merge(
        profile_rule_scores,
        on=["profile_id", "rule_id"],
        how="left",
    )
    return assemble_recommendations(expanded, rule_popularity, config=config)


def recommend_profiles(
    context: ServingContext,
    profile_ids: list[str],
    chunk_size: int = 1000,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Pages for exactly ``profile_ids``; scored by the ordering the context
    carries and never fitted here. Profiles are independent of one another, so
    any partition of them yields the same pages."""
    requested = sorted(set(map(str, profile_ids)))
    profiles = context.profiles[
        context.profiles["profile_id"].astype(str).isin(requested)
    ]
    missing = set(requested) - set(profiles["profile_id"].astype(str))
    if missing:
        raise ValueError(f"{len(missing)} requested profiles are not in the canonical mapping")
    candidate_pairs, rule_scores, metrics = order_candidates(context, profiles)
    pairs = candidate_pairs.merge(
        context.content_types, on="content_id", how="left", suffixes=("", "_item")
    )
    pairs["score"] = pairs["retrieval_score"].astype(float)
    recommendations: dict[str, dict[str, Any]] = {}
    for start in range(0, len(requested), chunk_size):
        chunk = set(requested[start : start + chunk_size])
        recommendations.update(
            _assemble_chunk(
                pairs[pairs["profile_id"].astype(str).isin(chunk)],
                rule_scores[rule_scores["profile_id"].astype(str).isin(chunk)],
                context.membership,
                context.rule_popularity,
                context.assembly_config,
            )
        )
    return recommendations, metrics


def serving_manifest_config(
    *,
    num_profiles,
    warm_repetitions,
    num_contents,
    profile_chunk_size,
    batch_size,
    amp,
    candidate_budget,
    exact_cowatch,
    rule_ordering,
    rule_scorer_fusion_weight,
    content_ordering,
    content_scorer_fusion_weight,
    contents_per_rule_max,
    page_content_budget,
    page_dedup,
    recency_candidates_per_rule,
    recency_reserved_per_rule,
    recency_max_age_days,
    candidate_rules,
    reserved_per_rule,
    source_candidates_per_rule,
    gbdt_backend,
    gbdt_device,
    scoring_config,
    rule_gbdt_backend,
    content_gbdt_backend,
    fcn_batch_size,
    fcn_checkpoint_hash,
) -> dict[str, Any]:
    """The config a serving run writes into its own manifest.

    Every knob that changes the served page belongs here, so a run's manifest
    says how its page was built; `test_cli_config_parity` compares this
    signature against the CLI.
    """

    return {
        "num_profiles": num_profiles,
        "warm_repetitions": warm_repetitions,
        "num_contents": num_contents,
        "profile_chunk_size": profile_chunk_size,
        "batch_size": batch_size,
        "amp": amp,
        "candidate_budget": candidate_budget,
        "exact_cowatch": exact_cowatch,
        "rule_ordering": rule_ordering,
        "rule_scorer_fusion_weight": rule_scorer_fusion_weight,
        "content_ordering": content_ordering,
        "content_scorer_fusion_weight": content_scorer_fusion_weight,
        "contents_per_rule_max": contents_per_rule_max,
        "page_content_budget": page_content_budget,
        "page_dedup": page_dedup,
        "recency_candidates_per_rule": recency_candidates_per_rule,
        "recency_reserved_per_rule": recency_reserved_per_rule,
        "recency_max_age_days": recency_max_age_days,
        "candidate_rules": candidate_rules,
        "reserved_per_rule": reserved_per_rule,
        "source_candidates_per_rule": source_candidates_per_rule,
        "gbdt_backend": gbdt_backend,
        "gbdt_device": gbdt_device,
        "scoring_config": scoring_config,
        "rule_gbdt_backend": rule_gbdt_backend,
        "content_gbdt_backend": content_gbdt_backend,
        "fcn_batch_size": fcn_batch_size,
        "fcn_checkpoint_hash": fcn_checkpoint_hash,
    }


def popularity_live_test(
    project_root: str | Path = ".",
    domain: str = "all",
    num_profiles: int = 100,
    warm_repetitions: int = 3,
    output_dir: str | Path | None = None,
    model_family: str = "per_rule_popularity",
    num_contents: int | None = None,
    profile_chunk_size: int = 128,
    batch_size: int = 4096,
    dry_run: bool = False,
    resume: bool = False,
    train_if_missing: bool = False,
    run_id: str | None = None,
    amp: bool = False,
    sample_gpu: bool = True,
    candidate_budget: int = 1000,
    exact_cowatch: bool = False,
    rule_ordering: str = "count",
    rule_scorer_fusion_weight: float = 1.25,
    content_ordering: str = "hand_sum",
    content_scorer_fusion_weight: float = 5.0,
    recency_candidates_per_rule: int = 0,
    recency_reserved_per_rule: int = 0,
    recency_max_age_days: int = 0,
    contents_per_rule_max: int | None = None,
    page_content_budget: int | None = None,
    page_dedup: str = "none",
    skip_offline_eval: bool = False,
    candidate_rules: int = 40,
    reserved_per_rule: int = 10,
    source_candidates_per_rule: int = 20,
    gbdt_backend: str = "sklearn",
    gbdt_device: str = "auto",
    scoring_config: ScoringConfig | None = None,
    rule_scorer_model: RuleScorerModel | None = None,
    content_scorer_model: ContentScorerModel | None = None,
    fcn_scorer_model: FCNScorerModel | None = None,
    rule_gbdt_backend: str | None = None,
    content_gbdt_backend: str | None = None,
    fcn_batch_size: int = 4096,
) -> MetricBundle:
    """Serve ``num_profiles`` profiles (0 serves all of them) and record metrics.

    Unlike the batch path this fits the learned scorers on the spot. Legacy
    ordering can fall back to the hand-built path; non-legacy compositions
    require their configured scorers and fail if one cannot be trained. This
    suits rehearsals rather than production.
    """
    if (
        num_profiles < 0
        or warm_repetitions < 0
        or profile_chunk_size < 1
        or candidate_budget < 200
    ):
        raise ValueError("profile count/chunk size must be positive and repetitions nonnegative")
    root = Path(project_root).resolve()
    scoring_config = scoring_config or resolve_scoring_config(
        legacy_content_ordering=content_ordering,
        legacy_fusion_weight=content_scorer_fusion_weight,
    )
    if scoring_config.uses_fcn and fcn_scorer_model is None:
        raise ValueError("FCN composition requires a loaded FCN scorer model")
    observed_rss = [_rss_bytes()]
    stage_started = time.perf_counter()
    assembly_config = RuleAssemblyConfig(
        contents_per_rule_max=contents_per_rule_max,
        page_content_budget=page_content_budget,
        page_dedup=page_dedup,
    )
    context = load_serving_context(
        root,
        domain,
        serving_retrieval_config(
            candidate_budget,
            exact_cowatch,
            recency_candidates_per_rule,
            recency_reserved_per_rule,
            recency_max_age_days,
            candidate_rules,
            reserved_per_rule,
            source_candidates_per_rule,
        ),
        assembly_config,
        num_contents=num_contents,
    )
    contents, membership = context.contents, context.membership
    full_membership, users = context.full_membership, context.users
    interactions, cutoff = context.interactions, context.cutoff
    profiles = context.profiles.head(num_profiles) if num_profiles else context.profiles
    load_seconds = time.perf_counter() - stage_started
    observed_rss.append(_rss_bytes())

    retrieval_started = time.perf_counter()
    # Serving fits on the untouched-test protocol windows (train + validation),
    # the configuration with cross-split promotion evidence. A scorer that
    # cannot train (stale or missing temporal windows) degrades to the
    # hand-built ordering and the reason lands in the run metrics.
    fallback_metrics: dict[str, Any] = {}
    loader = _interactions_loader(context.canonical)
    rule_model = rule_scorer_model
    content_model = content_scorer_model
    if rule_model is None and rule_ordering == "learned_rrf":
        try:
            rule_model = train_rule_scorer(
                context.canonical, loader, backend=rule_gbdt_backend or gbdt_backend, device=gbdt_device
            )
        except Exception as error:  # noqa: BLE001 - enhancer must never kill publish
            fallback_metrics.update(
                {
                    "rule_ordering": "count_fallback",
                    "rule_ordering_fallback_reason": f"{type(error).__name__}: {error}",
                }
            )
    elif rule_ordering != "count":
        raise ValueError(f"unsupported rule ordering: {rule_ordering}")
    if content_model is None and scoring_config.uses_gbdt:
        try:
            content_model = train_content_scorer(
                context.canonical, cutoff, loader, backend=content_gbdt_backend or gbdt_backend, device=gbdt_device
            )
        except Exception as error:  # noqa: BLE001 - enhancer must never kill publish
            fallback_metrics.update(
                {
                    "content_ordering": "hand_sum_fallback",
                    "content_ordering_fallback_reason": f"{type(error).__name__}: {error}",
                }
            )
    elif not scoring_config.uses_gbdt and content_ordering not in {"hand_sum", "learned_rrf"}:
        raise ValueError(f"unsupported content ordering: {content_ordering}")
    context = with_scorers(
        context,
        rule_model,
        content_model,
        rule_scorer_fusion_weight,
        content_scorer_fusion_weight,
        scoring_config,
        fcn_scorer_model,
        fcn_batch_size,
    )
    candidate_pairs, all_profile_rule_scores, retrieval_metrics = order_candidates(
        context, profiles
    )
    retrieval_metrics.update(fallback_metrics)
    candidate_content_ids = set(candidate_pairs["content_id"].astype(str))
    pool = contents[
        contents["content_id"].astype(str).isin(candidate_content_ids)
    ].drop_duplicates("content_id", keep="last")
    rule_popularity = context.rule_popularity
    profile_history = historical_profile_features(interactions, cutoff)
    profile_context = profiles.merge(profile_history, on="profile_id", how="left")
    if "username" in profile_context and "username" in users:
        profile_context = profile_context.merge(
            users.drop_duplicates("username", keep="last"), on="username", how="left"
        )
    retrieval_seconds = time.perf_counter() - retrieval_started
    observed_rss.append(_rss_bytes())

    model = contract = None
    checkpoint_hash = "popularity"
    if model_family != "per_rule_popularity":
        if model_family not in {
            "dense_repaired_baseline",
            "embedding_fcn",
            "embedding_dcnv3",
            "embedding_mlp_baseline",
            "legacy_embedding_dcnv3",
            "fcn_official",
            "ecn_official",
            "hierarchical_fcn",
        }:
            raise ValueError(f"Unsupported live-test model family: {model_family}")
        model_output = model_artifact_path(domain, "v2", model_family, project_root=root)
        checkpoint_path = model_output / "best_model_v2.pth"
        ensure_checkpoint(
            domain,
            checkpoint_path,
            train_if_missing=train_if_missing,
            # Imported here, not at module scope: training lives in
            # `pipeline_v2`, which re-exports this module.
            train_command=lambda: _train_temporal_model()(
                root,
                domain,
                model_family,
                resume=False,
                epochs=100,
                batch_size=batch_size,
            ),
        )
        model, contract, checkpoint = load_v2_checkpoint(
            checkpoint_path,
            model_output / "feature_contract_v2.json",
        )
        checkpoint_hash = str(checkpoint.get("contract_hash", "unknown"))

    required_model_features = set()
    if contract is not None:
        required_model_features.update(contract.numeric_features)
        required_model_features.update(contract.categorical_features)
        required_model_features.update(contract.multivalue_features)
    profile_pair_columns = [
        column
        for column in profile_context.columns
        if column == "profile_id" or column in required_model_features
    ]
    item_pair_columns = [
        column
        for column in pool.columns
        if column in {"content_id", "type_id", "retrieval_score"}
        or column in required_model_features
    ]

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None

    target_output = Path(output_dir) if output_dir else root / domain / "result_v2"
    profile_ids = profile_context["profile_id"].astype(str).tolist()
    cache_fingerprint_value = hashlib.sha256(
        json.dumps(
            {
                "profiles": profile_ids,
                "candidate_pairs": sorted(
                    candidate_pairs[["profile_id", "content_id"]]
                    .astype(str)
                    .itertuples(index=False, name=None)
                ),
                "model": model_family,
                "checkpoint": checkpoint_hash,
                "profile_chunk_size": profile_chunk_size,
                # A different page shape produces different output, so cached
                # chunks from another shape must not be reused.
                "page_shape": [
                    assembly_config.min_contents,
                    assembly_config.max_contents,
                    assembly_config.page_content_budget,
                    assembly_config.page_dedup,
                ],
                # Content ordering rewrites retrieval_score without changing a
                # single (profile_id, content_id) pair, so it has to be part of
                # the key or a re-run silently replays the previous ordering.
                "content_ordering": [
                    content_ordering,
                    content_scorer_fusion_weight,
                    scoring_config_payload(scoring_config),
                    fcn_scorer_model.artifact_hash
                    if fcn_scorer_model is not None
                    else None,
                ],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    resume_dir = target_output / ".resume" / cache_fingerprint_value

    def score_once(use_resume_cache: bool, write_resume_cache: bool) -> tuple[dict[str, dict[str, Any]], dict[str, float | int]]:
        current: dict[str, dict[str, Any]] = {}
        encode_forward_seconds = 0.0
        assembly_seconds = 0.0
        scored_pairs = 0
        resume_chunks_loaded = 0
        for start in range(0, len(profile_context), profile_chunk_size):
            stop = min(start + profile_chunk_size, len(profile_context))
            chunk_number = start // profile_chunk_size
            cache_path = resume_dir / f"chunk_{chunk_number:05d}.json"
            chunk_profile_ids = set(
                profile_context.iloc[start:stop]["profile_id"].astype(str)
            )
            chunk_candidates = candidate_pairs[
                candidate_pairs["profile_id"].astype(str).isin(chunk_profile_ids)
            ].copy()
            if use_resume_cache and cache_path.exists():
                current.update(json.loads(cache_path.read_text(encoding="utf-8")))
                scored_pairs += len(chunk_candidates)
                resume_chunks_loaded += 1
                continue
            chunk_profiles = profile_context.iloc[start:stop][profile_pair_columns]
            forward_started = time.perf_counter()
            pairs = chunk_candidates.merge(
                chunk_profiles,
                on="profile_id",
                how="left",
                suffixes=("", "_profile"),
            ).merge(
                pool[item_pair_columns],
                on="content_id",
                how="left",
                suffixes=("", "_item"),
            )
            pairs["rule_id"] = pairs["candidate_rule_id"].astype("string")
            if model_family == "per_rule_popularity":
                pairs["score"] = pairs["retrieval_score"].astype(float)
            else:
                prepared_pairs = prepare_inference_input(
                    contract, pairs, model_family
                )
                pairs["score"] = predict(
                    model,
                    contract,
                    pairs,
                    model_family,
                    batch_size=batch_size,
                    amp=amp,
                    prepared=prepared_pairs,
                )
            if model_family == "hierarchical_fcn":
                pairs["model_rule_score"] = predict(
                    model,
                    contract,
                    pairs,
                    model_family,
                    batch_size=batch_size,
                    amp=amp,
                    output_key="rule_score",
                    prepared=prepared_pairs,
                )
                model_rule_scores = aggregate_model_rule_scores(
                    pairs,
                    membership,
                )
                profile_rule_scores = pd.concat(
                    [
                        all_profile_rule_scores[
                            all_profile_rule_scores["profile_id"]
                            .astype(str)
                            .isin(chunk_profile_ids)
                        ],
                        model_rule_scores,
                    ],
                    ignore_index=True,
                ).drop_duplicates(["profile_id", "rule_id"], keep="last")
            else:
                profile_rule_scores = all_profile_rule_scores[
                    all_profile_rule_scores["profile_id"]
                    .astype(str)
                    .isin(chunk_profile_ids)
                ]
            encode_forward_seconds += time.perf_counter() - forward_started
            scored_pairs += len(pairs)
            assembly_started = time.perf_counter()
            assembled = _assemble_chunk(
                pairs, profile_rule_scores, membership, rule_popularity, assembly_config
            )
            assembly_seconds += time.perf_counter() - assembly_started
            current.update(assembled)
            if write_resume_cache and not dry_run:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(assembled, sort_keys=True), encoding="utf-8")
                temporary.replace(cache_path)
        return current, {
            "encode_forward_seconds": encode_forward_seconds,
            "rule_assembly_seconds": assembly_seconds,
            "scored_pairs": scored_pairs,
            "resume_chunks_loaded": resume_chunks_loaded,
        }

    gpu_sampler = NvidiaSmiSampler(
        enabled=bool(
            sample_gpu
            and torch is not None
            and torch.cuda.is_available()
            and model_family != "per_rule_popularity"
        )
    )
    gpu_sampler.start()
    try:
        cold_started = time.perf_counter()
        recommendations, cold_stages = score_once(resume, True)
        cold_seconds = time.perf_counter() - cold_started
        observed_rss.append(_rss_bytes())
        warm_seconds: list[float] = []
        warm_stage_metrics: list[dict[str, float | int]] = []
        deterministic_hashes: list[str] = []
        for _ in range(warm_repetitions):
            started = time.perf_counter()
            current, stage_metrics = score_once(False, False)
            elapsed = time.perf_counter() - started
            homepage_check, text_check = serialize_assembled_output(current)
            deterministic_hashes.append(
                hashlib.sha256(
                    (json.dumps(homepage_check, sort_keys=True) + text_check).encode("utf-8")
                ).hexdigest()
            )
            warm_seconds.append(elapsed)
            warm_stage_metrics.append(stage_metrics)
            recommendations = current
            observed_rss.append(_rss_bytes())
    finally:
        gpu_sampling_metrics = gpu_sampler.stop()

    serialization_started = time.perf_counter()
    homepage, text = serialize_assembled_output(recommendations)
    catalog = context.catalog
    valid_membership = context.valid_membership
    validation_started = time.perf_counter()
    validation = validate_client_output(
        homepage,
        text,
        set(profile_ids),
        catalog,
        valid_membership=valid_membership,
        config=assembly_config,
    )
    validation_seconds = time.perf_counter() - validation_started
    if not validation.passed:
        errors = "; ".join(validation.errors[:5]) or "unknown validation failure"
        raise ValueError(f"Client output validation failed: {errors}")
    if not dry_run:
        homepage_path, content_output_path = atomic_write_client_outputs(
            target_output, recommendations, config=assembly_config
        )
        # Also emit the production MyTV rs_output-named files. `mid` is joined
        # from the canonical profile mapping; `gid` has no source in the
        # delivered data and stays the sample constant until VNPT supplies one.
        write_production_named_outputs(
            target_output,
            homepage,
            text,
            domain=domain,
            date=time.strftime("%Y%m%d", time.localtime()),
            member_ids=_member_id_map(profiles),
        )
    else:
        homepage_path = target_output / "rulename.json"
        content_output_path = target_output / "rule_content.txt"
    serialization_seconds = time.perf_counter() - serialization_started
    benchmark_values = warm_seconds or [cold_seconds]
    series = pd.Series(benchmark_values, dtype=float)
    benchmark_seconds = float(series.quantile(0.50))
    inference_pairs = len(candidate_pairs)
    output_bytes = len(json.dumps(homepage, ensure_ascii=False).encode("utf-8")) + len(text.encode("utf-8"))
    output_hash = hashlib.sha256(
        (json.dumps(homepage, sort_keys=True) + text).encode("utf-8")
    ).hexdigest()
    fallback_count = sum(
        row["fallback_count"]
        for payload in recommendations.values()
        for row in payload["normal_rules"]
    )
    distinct_recommended = {
        item["content_id"]
        for payload in recommendations.values()
        for row in payload["normal_rules"]
        for item in row["contents"]
    }
    encode_forward_seconds = (
        float(np.median([row["encode_forward_seconds"] for row in warm_stage_metrics]))
        if warm_stage_metrics
        else float(cold_stages["encode_forward_seconds"])
    )
    rule_assembly_seconds = (
        float(np.median([row["rule_assembly_seconds"] for row in warm_stage_metrics]))
        if warm_stage_metrics
        else float(cold_stages["rule_assembly_seconds"])
    )
    inference_metrics = {
        "load_seconds": load_seconds,
        "retrieval_seconds": retrieval_seconds,
        "cold_seconds": cold_seconds,
        "cold_end_to_end_seconds": load_seconds + retrieval_seconds + cold_seconds + validation_seconds,
        "warm_repetitions": len(warm_seconds),
        "warm_seconds": warm_seconds,
        "p50_seconds": benchmark_seconds,
        "p95_seconds": float(series.quantile(0.95)),
        "p99_seconds": float(series.quantile(0.99)),
        "profiles_per_second": len(profile_ids) / max(benchmark_seconds, 1e-12),
        "inference_time_seconds": benchmark_seconds,
        "inference_pairs": inference_pairs,
        "inference_seconds_per_pair": benchmark_seconds / max(inference_pairs, 1),
        "pairs_per_second": inference_pairs / max(benchmark_seconds, 1e-12),
        "batch_size": batch_size,
        "amp": amp,
        "encode_forward_seconds": encode_forward_seconds,
        "rule_assembly_seconds": rule_assembly_seconds,
        "validation_seconds": validation_seconds,
        "serialization_seconds": serialization_seconds,
        "processing_error_count": 0,
        "candidate_contents": int(candidate_pairs["content_id"].nunique()),
        "candidate_pairs_actual": int(len(candidate_pairs)),
        "catalog_contents": int(len(contents)),
        "profiles_scored": len(profile_ids),
        "profiles_requested": num_profiles,
        "output_bytes": output_bytes,
        "output_sha256": output_hash,
        "deterministic_warm_hash_rate": bool(deterministic_hashes)
        and len(set(deterministic_hashes)) <= 1,
        "dry_run": dry_run,
        "resume_enabled": resume,
        "resume_chunks_loaded": int(cold_stages["resume_chunks_loaded"]),
        "peak_observed_rss_bytes": max(observed_rss),
        **retrieval_metrics,
        **gpu_sampling_metrics,
    }
    inference_metrics["timed_stage_seconds"] = (
        encode_forward_seconds + rule_assembly_seconds
    )
    inference_metrics["unattributed_inference_seconds"] = max(
        0.0, benchmark_seconds - inference_metrics["timed_stage_seconds"]
    )
    emitted_content_slots = sum(
        len(rule.get("contents", []))
        for payload in recommendations.values()
        for rule in payload.get("normal_rules", [])
    )
    output_metrics = {
        **validation.metrics,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / max(emitted_content_slots, 1),
        "distinct_recommended_contents": len(distinct_recommended),
        "catalog_coverage": len(distinct_recommended) / max(len(contents), 1),
        "validation_errors": validation.errors,
    }
    evaluation_updates: dict[str, Any] = {}
    test_candidates_path = data_artifact_path(
        "canonical", domain, "temporal", "test_candidates.parquet", project_root=root
    )
    if test_candidates_path.exists() and not skip_offline_eval:
        test_candidates = pd.read_parquet(test_candidates_path)
        test_candidates = test_candidates[
            test_candidates["profile_id"].astype(str).isin(set(profile_ids))
        ]
        offline_cutoff = load_temporal_windows(
            data_artifact_path("canonical", domain, "temporal", project_root=root)
        )[4].history_end
        offline_popularity = historical_content_popularity(
            interactions, cutoff=offline_cutoff
        )
        (
            offline_scored,
            offline_rule_scores,
            offline_retrieval,
            offline_evaluation_truth,
            _,
        ) = (
            _serving_candidates_for_evaluation(
                root,
                domain,
                test_candidates,
                candidate_budget=1000,
                cutoff=offline_cutoff,
                exact_cowatch=exact_cowatch,
            )
        )
        if model_family == "per_rule_popularity":
            offline_scored["score"] = offline_scored["retrieval_score"].astype(float)
        else:
            offline_prepared = prepare_inference_input(
                contract, offline_scored, model_family
            )
            offline_scored["score"] = predict(
                model,
                contract,
                offline_scored,
                model_family,
                batch_size=batch_size,
                amp=amp,
                prepared=offline_prepared,
            )
            if model_family == "hierarchical_fcn":
                offline_scored["model_rule_score"] = predict(
                    model,
                    contract,
                    offline_scored,
                    model_family,
                    batch_size=batch_size,
                    amp=amp,
                    output_key="rule_score",
                    prepared=offline_prepared,
                )
                model_rule_scores = aggregate_model_rule_scores(
                    offline_scored,
                    full_membership,
                )
                offline_rule_scores = pd.concat(
                    [offline_rule_scores, model_rule_scores], ignore_index=True
                ).drop_duplicates(["profile_id", "rule_id"], keep="last")
        quality, _ = evaluate_scored_candidates(
            offline_scored,
            full_membership,
            offline_popularity,
            ground_truth_frame=offline_evaluation_truth,
            profile_rule_scores=offline_rule_scores,
        )
        evaluation_updates.update(quality)
        evaluation_updates.update(offline_retrieval)
    inference_metrics["peak_gpu_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated())
        if torch is not None and torch.cuda.is_available()
        else 0
    )
    data_metrics_path = data_artifact_path("canonical", domain, "data_metrics.json", project_root=root)
    data_metrics = json.loads(data_metrics_path.read_text(encoding="utf-8")) if data_metrics_path.exists() else {}
    store = MetricsStore(root)
    if run_id is not None:
        bundle = store.read_bundle(run_id, domain)
        if bundle.manifest.model_family != model_family:
            raise ValueError("metrics run model family does not match live-test model")
        bundle.data.update(data_metrics)
        bundle.inference.update(inference_metrics)
        bundle.output = output_metrics
        bundle.evaluation.update(evaluation_updates)
    else:
        manifest = create_manifest(
            root,
            domain,
            "sprint-5",
            model_family,
            config=serving_manifest_config(
                num_profiles=num_profiles,
                warm_repetitions=warm_repetitions,
                num_contents=num_contents,
                profile_chunk_size=profile_chunk_size,
                batch_size=batch_size,
                amp=amp,
                candidate_budget=candidate_budget,
                exact_cowatch=exact_cowatch,
                rule_ordering=rule_ordering,
                rule_scorer_fusion_weight=rule_scorer_fusion_weight,
                content_ordering=content_ordering,
                content_scorer_fusion_weight=content_scorer_fusion_weight,
                contents_per_rule_max=contents_per_rule_max,
                page_content_budget=page_content_budget,
                page_dedup=page_dedup,
                recency_candidates_per_rule=recency_candidates_per_rule,
                recency_reserved_per_rule=recency_reserved_per_rule,
                recency_max_age_days=recency_max_age_days,
                candidate_rules=candidate_rules,
                reserved_per_rule=reserved_per_rule,
                source_candidates_per_rule=source_candidates_per_rule,
                gbdt_backend=gbdt_backend,
                gbdt_device=gbdt_device,
                scoring_config=scoring_config_payload(scoring_config),
                rule_gbdt_backend=rule_gbdt_backend or gbdt_backend,
                content_gbdt_backend=content_gbdt_backend or gbdt_backend,
                fcn_batch_size=fcn_batch_size,
                fcn_checkpoint_hash=(
                    fcn_scorer_model.artifact_hash
                    if fcn_scorer_model is not None
                    else None
                ),
            ),
        )
        bundle = MetricBundle(
            manifest=manifest,
            data=data_metrics,
            inference=inference_metrics,
            output=output_metrics,
            evaluation=evaluation_updates,
        )
    directory = store.write_bundle(bundle)
    gate = evaluate_gate(bundle, bundle.manifest.phase)
    (directory / "gate_result.json").write_text(json.dumps(gate.to_dict(), indent=2), encoding="utf-8")
    report = render_markdown(bundle, gate)
    (directory / "report.md").write_text(report, encoding="utf-8")
    return bundle
