"""Learned graded rule scorer shared by serving evaluation and experiments.

Fits two gradient-boosted heads (scikit-learn or XGBoost, see ``gbdt``) (completion/watch-time gain,
profile-share target) on pre-cutoff temporal windows from interaction-log
features only, then RRF-fuses the learned rule ranks with the incumbent
count-affinity ranks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from recsys_vnpt.canonical import select_content_metadata_as_of
from recsys_vnpt.gbdt import head_device, make_head
from recsys_vnpt.temporal import load_temporal_windows, temporal_artifact_root


GAIN_COLUMNS = ("completion_gain", "watch_time_gain")
FEATURE_COLUMNS = (
    "rule_index",
    "historical_rule_events",
    "historical_rule_unique_contents",
    "historical_rule_watch_seconds",
    "historical_rule_events_30d",
    "historical_rule_events_90d",
    "historical_rule_watch_seconds_30d",
    "historical_rule_watch_seconds_90d",
    "historical_rule_recency_days",
    "historical_profile_events",
    "historical_profile_unique_contents",
    "historical_profile_watch_seconds",
    "historical_rule_event_share",
    "historical_rule_watch_share",
    "global_rule_event_share",
    "global_rule_watch_share",
    "global_rule_profiles",
    "rule_catalog_contents",
)


def membership_and_rules_as_of(
    canonical: Path,
    cutoff: pd.Timestamp,
    normal_membership: pd.DataFrame,
    rule_index: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], set[str]]:
    contents = pd.read_parquet(canonical / "contents.parquet", columns=["content_id"])
    current_ids = set(contents["content_id"].astype(str))
    versions_path = canonical / "content_versions.parquet"
    metadata_backfill_contents = 0
    if versions_path.exists():
        as_of = select_content_metadata_as_of(
            pd.read_parquet(versions_path), cutoff
        )
        as_of_ids = set(as_of["content_id"].astype(str)) & current_ids
        in_scope = as_of["content_id"].astype(str).isin(as_of_ids)
        metadata_backfill_contents = int(
            as_of.get(
                "metadata_backfill", pd.Series(False, index=as_of.index)
            ).fillna(False).astype(bool).loc[in_scope].sum()
        )
    else:
        as_of_ids = current_ids
    membership = normal_membership[
        normal_membership["content_id"].isin(as_of_ids)
    ].copy()
    support = (
        membership.groupby("rule_id")["content_id"]
        .nunique()
        .rename("rule_catalog_contents")
        .reset_index()
    )
    rules = rule_index.merge(support, on="rule_id", how="inner")
    rules = rules[rules["rule_catalog_contents"].ge(10)].copy()
    membership = membership[
        membership["rule_id"].isin(set(rules["rule_id"]))
    ].copy()
    metrics = {
        "cutoff": str(cutoff),
        "as_of_contents": int(len(as_of_ids)),
        "normal_memberships": int(len(membership)),
        "eligible_normal_rules": int(len(rules)),
        "metadata_backfill_contents": metadata_backfill_contents,
        "metadata_backfill_rate": metadata_backfill_contents
        / max(len(as_of_ids), 1),
        "rule_membership_backfill": True,
    }
    return membership, rules, metrics, as_of_ids


def cross_profiles_rules(
    profiles: pd.Series | list[str], rules: pd.DataFrame
) -> pd.DataFrame:
    profile_frame = pd.DataFrame(
        {"profile_id": pd.Series(profiles, dtype="string").drop_duplicates()}
    )
    profile_frame["_join"] = 1
    rule_frame = rules.copy()
    rule_frame["_join"] = 1
    return profile_frame.merge(rule_frame, on="_join", how="inner").drop(
        columns="_join"
    )


def window_targets(
    path: Path,
    membership: pd.DataFrame,
    rules: pd.DataFrame,
) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=[
            "profile_id",
            "content_id",
            "sample_type",
            *GAIN_COLUMNS,
        ],
    )
    observed = frame[frame["sample_type"].eq("observed_engagement")].copy()
    observed = observed.astype({"profile_id": str, "content_id": str})
    observed = observed.merge(membership, on="content_id", how="inner")
    for gain in GAIN_COLUMNS:
        observed[gain] = (
            pd.to_numeric(observed[gain], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )
    targets = (
        observed.groupby(["profile_id", "rule_id"], as_index=False)[
            list(GAIN_COLUMNS)
        ]
        .sum()
        .reset_index(drop=True)
    )
    profiles = frame["profile_id"].astype(str).drop_duplicates()
    output = cross_profiles_rules(profiles, rules)
    output = output.merge(
        targets, on=["profile_id", "rule_id"], how="left"
    )
    for gain in GAIN_COLUMNS:
        output[gain] = output[gain].fillna(0.0).astype("float32")
    return output


def rule_history(
    interactions: pd.DataFrame,
    cutoff: pd.Timestamp,
    membership: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pre-cutoff events on rule content, and each rule's share of all of them.

    Both depend only on the cutoff, never on which profiles are scored, so a
    sharded run computes them once and reuses them for every shard.
    """
    history = interactions[
        pd.to_datetime(interactions["create_date"], errors="coerce").le(cutoff)
    ].copy()
    history = history.astype({"profile_id": str, "content_id": str})
    history = history[
        history["content_id"].isin(set(membership["content_id"].astype(str)))
    ].copy()
    history["duration"] = pd.to_numeric(
        history["duration"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    history["_event_time"] = pd.to_datetime(
        history["create_date"], errors="coerce"
    )
    history["_age_days"] = (
        (cutoff - history["_event_time"]).dt.total_seconds() / 86400.0
    ).clip(lower=0.0)
    history["_recent_30"] = history["_age_days"].le(30).astype("int8")
    history["_recent_90"] = history["_age_days"].le(90).astype("int8")
    history["_duration_30"] = history["duration"] * history["_recent_30"]
    history["_duration_90"] = history["duration"] * history["_recent_90"]

    expanded_all = history.merge(membership, on="content_id", how="inner")
    global_rule = (
        expanded_all.groupby("rule_id", as_index=False)
        .agg(
            global_rule_events=("content_id", "size"),
            global_rule_watch_seconds=("duration", "sum"),
            global_rule_profiles=("profile_id", "nunique"),
        )
        .reset_index(drop=True)
    )
    global_rule["global_rule_event_share"] = global_rule[
        "global_rule_events"
    ] / max(float(global_rule["global_rule_events"].sum()), 1.0)
    global_rule["global_rule_watch_share"] = global_rule[
        "global_rule_watch_seconds"
    ] / max(float(global_rule["global_rule_watch_seconds"].sum()), 1.0)
    return history, global_rule


def history_features(
    interactions: pd.DataFrame,
    cutoff: pd.Timestamp,
    profiles: pd.Series | list[str],
    membership: pd.DataFrame,
    rules: pd.DataFrame,
    prepared: tuple[pd.DataFrame, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """One row per (profile, eligible rule) with the scorer's input features.

    ``prepared`` is ``rule_history`` for the same cutoff and membership.
    """
    profile_ids = set(map(str, profiles))
    history, global_rule = (
        prepared
        if prepared is not None
        else rule_history(interactions, cutoff, membership)
    )

    requested = history[history["profile_id"].isin(profile_ids)].copy()
    profile_totals = (
        requested.groupby("profile_id", as_index=False)
        .agg(
            historical_profile_events=("content_id", "size"),
            historical_profile_unique_contents=("content_id", "nunique"),
            historical_profile_watch_seconds=("duration", "sum"),
        )
        .reset_index(drop=True)
    )
    expanded = requested.merge(membership, on="content_id", how="inner")
    profile_rule = (
        expanded.groupby(["profile_id", "rule_id"], as_index=False)
        .agg(
            historical_rule_events=("content_id", "size"),
            historical_rule_unique_contents=("content_id", "nunique"),
            historical_rule_watch_seconds=("duration", "sum"),
            historical_rule_events_30d=("_recent_30", "sum"),
            historical_rule_events_90d=("_recent_90", "sum"),
            historical_rule_watch_seconds_30d=("_duration_30", "sum"),
            historical_rule_watch_seconds_90d=("_duration_90", "sum"),
            historical_rule_recency_days=("_age_days", "min"),
        )
        .reset_index(drop=True)
    )

    output = cross_profiles_rules(profiles, rules)
    output = output.merge(
        profile_rule, on=["profile_id", "rule_id"], how="left"
    )
    output = output.merge(profile_totals, on="profile_id", how="left")
    output = output.merge(
        global_rule[
            [
                "rule_id",
                "global_rule_event_share",
                "global_rule_watch_share",
                "global_rule_profiles",
            ]
        ],
        on="rule_id",
        how="left",
    )
    output["historical_rule_event_share"] = output[
        "historical_rule_events"
    ] / output["historical_profile_events"].clip(lower=1.0)
    output["historical_rule_watch_share"] = output[
        "historical_rule_watch_seconds"
    ] / output["historical_profile_watch_seconds"].clip(lower=1.0)
    output["historical_rule_recency_days"] = output[
        "historical_rule_recency_days"
    ].fillna(3650.0)
    for column in FEATURE_COLUMNS:
        if column == "rule_index":
            continue
        output[column] = (
            pd.to_numeric(output[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype("float32")
        )
    output["rule_index"] = output["rule_index"].astype("float32")
    return output


def rank_score(frame: pd.DataFrame, column: str, constant: float = 60.0) -> pd.Series:
    ranks = frame.groupby("profile_id", sort=False)[column].rank(
        method="first", ascending=False
    )
    return 1.0 / (constant + ranks)


INTERACTION_COLUMNS = ["profile_id", "content_id", "create_date", "duration"]


@dataclass(frozen=True)
class RuleScorerModel:
    """The two fitted heads and the rule coding they were trained with."""

    heads: dict[str, Any]
    rule_index: pd.DataFrame
    metrics: dict[str, Any]


@dataclass(frozen=True)
class RuleServingScope:
    """Everything the scorer reads at one serving cutoff, shared across shards."""

    cutoff: pd.Timestamp
    membership: pd.DataFrame
    rules: pd.DataFrame
    metrics: dict[str, Any]
    interactions: pd.DataFrame
    history: tuple[pd.DataFrame, pd.DataFrame]


def normal_rule_index(canonical: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normal-rule membership and the fixed integer coding of every normal rule."""
    from recsys_vnpt.rule_assembly import SPECIAL_RULES

    full_membership = pd.read_parquet(
        canonical / "content_rule_membership.parquet",
        columns=["content_id", "rule_id"],
    ).drop_duplicates().astype({"content_id": str, "rule_id": str})
    normal_membership = full_membership[
        ~full_membership["rule_id"].isin(SPECIAL_RULES)
    ].copy()
    rule_index = pd.DataFrame(
        {"rule_id": sorted(normal_membership["rule_id"].astype(str).unique())}
    )
    rule_index["rule_index"] = np.arange(len(rule_index), dtype=np.int16)
    return normal_membership, rule_index


def training_windows(canonical: Path, evaluation_split: str) -> list[Any]:
    """Strictly pre-cutoff training windows.

    The first three temporal windows for a validation-split evaluation; a
    test-split evaluation and serving add the validation window (the locked
    untouched-test refit protocol). Windows come from the temporal artifact,
    so relative-window deployments train on their own periods.
    """
    if evaluation_split not in {"validation", "test"}:
        raise ValueError("rule scorer split must be validation or test")
    return list(
        load_temporal_windows(temporal_artifact_root(canonical))[
            : 4 if evaluation_split == "test" else 3
        ]
    )


def fit_rule_scorer(
    canonical: Path,
    interactions: pd.DataFrame,
    windows: list[Any],
    window_scopes: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], set[str]]],
    rule_index: pd.DataFrame,
    seed: int = 42,
    target_normalization: str = "profile_share",
    observed_weight: float = 3.0,
    learning_rate: float = 0.06,
    max_iter: int = 80,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 50,
    l2_regularization: float = 1.0,
    backend: str = "sklearn",
    device: str = "auto",
) -> RuleScorerModel:
    """Fit both gain heads on the temporal windows."""
    event_times = pd.to_datetime(interactions["create_date"], errors="coerce")
    minimum_event, maximum_event = event_times.min(), event_times.max()
    if (
        pd.isna(maximum_event)
        or windows[0].label_start > maximum_event
        or windows[-1].label_end < minimum_event
    ):
        raise ValueError(
            "rule scorer temporal windows "
            f"({windows[0].label_start:%Y-%m-%d}.."
            f"{windows[-1].label_end:%Y-%m-%d}) do not overlap the "
            f"interaction span ({minimum_event}..{maximum_event}); rebuild "
            "temporal artifacts (pipeline_cli build-temporal --window-mode relative)"
        )

    train_parts: list[pd.DataFrame] = []
    for window in windows:
        window_membership, window_rules, _, _ = window_scopes[window.name]
        targets = window_targets(
            temporal_artifact_root(canonical) / f"{window.name}.parquet",
            window_membership,
            window_rules,
        )
        features = history_features(
            interactions,
            window.history_end,
            targets["profile_id"],
            window_membership,
            window_rules,
        )
        part = features.merge(
            targets[["profile_id", "rule_id", *GAIN_COLUMNS]],
            on=["profile_id", "rule_id"],
            how="inner",
        )
        part["window"] = window.name
        train_parts.append(part)
    training = pd.concat(train_parts, ignore_index=True)
    if training.empty or not any(
        (training[gain] > 0).any() for gain in GAIN_COLUMNS
    ):
        raise ValueError(
            "rule scorer has no positive training rows in windows "
            f"{[w.name for w in windows]} (ending "
            f"{windows[-1].label_end if windows else 'n/a'}); "
            "rebuild temporal artifacts for this data period "
            "(pipeline_cli build-temporal --window-mode relative)"
        )
    if float(training["historical_profile_events"].sum()) == 0:
        # All-zero features would make the model learn a per-rule constant and
        # silently de-personalize every profile after RRF fusion.
        raise ValueError(
            "rule scorer training features are degenerate (no pre-window "
            "history overlaps the temporal windows); rebuild temporal "
            "artifacts (pipeline_cli build-temporal --window-mode relative)"
        )

    x_train = training[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    heads: dict[str, Any] = {}
    fit_metrics: dict[str, Any] = {}
    for gain in GAIN_COLUMNS:
        if target_normalization == "profile_share":
            denominators = training.groupby(
                ["window", "profile_id"], sort=False
            )[gain].transform("sum").clip(lower=1e-12)
            target = (training[gain] / denominators).to_numpy(dtype=np.float32)
        else:
            target = np.log1p(training[gain].to_numpy(dtype=np.float32))
        weights = np.where(target > 0, observed_weight, 1.0).astype(np.float32)
        head = make_head(
            backend,
            device,
            [column == "rule_index" for column in FEATURE_COLUMNS],
            seed,
            learning_rate,
            max_iter,
            max_leaf_nodes,
            min_samples_leaf,
            l2_regularization,
        )
        head.fit(x_train, target, sample_weight=weights)
        heads[gain] = head
        fit_metrics[gain] = {
            "iterations": int(head.n_iter_),
            "target_positive_rate": float(np.mean(target > 0)),
        }
    return RuleScorerModel(
        heads=heads,
        rule_index=rule_index,
        metrics={
            "rule_scorer_backend": backend,
            "rule_scorer_device": head_device(next(iter(heads.values()))),
            "rule_scorer_target_normalization": target_normalization,
            "rule_scorer_training_windows": [w.name for w in windows],
            "rule_scorer_training_rows": int(len(training)),
            "rule_scorer_feature_columns": list(FEATURE_COLUMNS),
            "rule_scorer_fit": fit_metrics,
        },
    )


def train_rule_scorer(
    canonical: Path,
    interactions_loader: Callable[[set[str], list[str]], pd.DataFrame],
    evaluation_split: str = "test",
    **fit_parameters: Any,
) -> RuleScorerModel:
    """Fit a scorer from the canonical artifacts alone, for persisting."""
    normal_membership, rule_index = normal_rule_index(canonical)
    windows = training_windows(canonical, evaluation_split)
    window_scopes = {
        window.name: membership_and_rules_as_of(
            canonical, window.history_end, normal_membership, rule_index
        )
        for window in windows
    }
    content_ids = set().union(*(scope[3] for scope in window_scopes.values()))
    interactions = interactions_loader(content_ids, INTERACTION_COLUMNS)
    return fit_rule_scorer(
        canonical, interactions, windows, window_scopes, rule_index, **fit_parameters
    )


def rule_serving_scope(
    canonical: Path,
    cutoff: pd.Timestamp,
    model: RuleScorerModel,
    interactions_loader: Callable[[set[str], list[str]], pd.DataFrame],
) -> RuleServingScope:
    """As-of rule scope and history at a serving cutoff.

    A rule the model never saw keeps a NaN code, which the heads treat as a
    missing category, instead of dropping out of the page.
    """
    normal_membership, current_index = normal_rule_index(canonical)
    rule_index = current_index[["rule_id"]].merge(
        model.rule_index, on="rule_id", how="left"
    )
    rule_index["rule_index"] = rule_index["rule_index"].astype("float32")
    membership, rules, metrics, content_ids = membership_and_rules_as_of(
        canonical, cutoff, normal_membership, rule_index
    )
    interactions = interactions_loader(content_ids, INTERACTION_COLUMNS)
    return RuleServingScope(
        cutoff=cutoff,
        membership=membership,
        rules=rules,
        metrics=metrics,
        interactions=interactions,
        history=rule_history(interactions, cutoff, membership),
    )


def apply_rule_scorer(
    model: RuleScorerModel,
    scope: RuleServingScope,
    profiles: pd.Series,
    incumbent_scores: pd.DataFrame,
    fusion_weight: float = 1.25,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score every eligible rule for ``profiles`` and RRF-fuse with the incumbent."""
    if fusion_weight <= 0:
        raise ValueError("rule scorer fusion weight must be positive")
    features = history_features(
        scope.interactions,
        scope.cutoff,
        profiles,
        scope.membership,
        scope.rules,
        prepared=scope.history,
    )
    x_serving = features[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    # Chunked predict caps sklearn's float64 conversion buffer; full-frame
    # predict OOMs on memory-pressured hosts at ~1M rows.
    chunk = 200_000
    for gain, head in model.heads.items():
        features[f"predicted_{gain}"] = np.concatenate(
            [
                head.predict(x_serving[start : start + chunk])
                for start in range(0, len(x_serving), chunk)
            ]
        ) if len(x_serving) else np.empty(0)
    features["learned_rule_score"] = rank_score(
        features, "predicted_completion_gain"
    ) + rank_score(features, "predicted_watch_time_gain")

    fused = features[["profile_id", "rule_id"]].copy()
    fused = fused.merge(
        incumbent_scores.rename(columns={"rule_score": "incumbent_rule_score"}),
        on=["profile_id", "rule_id"],
        how="left",
    )
    fused["incumbent_rule_score"] = fused["incumbent_rule_score"].fillna(0.0)
    fused["rule_score"] = rank_score(
        fused, "incumbent_rule_score"
    ) + fusion_weight * rank_score(
        features, "learned_rule_score"
    ).to_numpy()
    metrics = {
        "rule_ordering": "learned_rrf",
        "rule_scorer_fusion_weight": float(fusion_weight),
        **model.metrics,
        **{
            f"rule_scorer_evaluation_{key}": value
            for key, value in scope.metrics.items()
        },
    }
    return fused[["profile_id", "rule_id", "rule_score"]], metrics
