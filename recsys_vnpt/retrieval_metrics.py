"""Measuring what retrieval reached.

Recall and gain-recall by profile, the recall curves, and the attribution that
says which channel first surfaced each positive. None of it feeds the pools
themselves, which is why it lives apart from the builder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Any


def candidate_recall_by_profile(
    candidates: pd.DataFrame,
    positives: pd.DataFrame,
) -> tuple[float, dict[str, float]]:
    """Return micro recall and per-profile recall for profile-specific pools."""
    required = {"profile_id", "content_id"}
    if not required.issubset(candidates) or not required.issubset(positives):
        raise ValueError("candidate recall requires profile_id and content_id")
    candidate_map = {
        str(profile): set(group["content_id"].astype(str))
        for profile, group in candidates.groupby("profile_id")
    }
    per_profile: dict[str, float] = {}
    hits = total = 0
    for profile, group in positives.groupby(positives["profile_id"].astype(str)):
        truth = set(group["content_id"].astype(str))
        profile_hits = len(truth & candidate_map.get(str(profile), set()))
        hits += profile_hits
        total += len(truth)
        per_profile[str(profile)] = profile_hits / len(truth) if truth else 0.0
    return hits / total if total else 0.0, per_profile


def candidate_gain_recall_by_profile(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    gain_column: str,
    k: int | None = None,
) -> tuple[float, dict[str, float]]:
    """Return micro and per-profile gain retained by a candidate pool."""
    required = {"profile_id", "content_id"}
    if not required.issubset(candidates) or not required.issubset(truth):
        raise ValueError("candidate gain recall requires profile_id and content_id")
    if gain_column not in truth:
        raise ValueError(f"candidate gain recall is missing {gain_column}")
    if k is not None and k < 1:
        raise ValueError("candidate gain recall K must be positive")
    ranked = candidates[["profile_id", "content_id"]].copy()
    ranked["profile_id"] = ranked["profile_id"].astype(str)
    ranked["content_id"] = ranked["content_id"].astype(str)
    if "candidate_rank" in candidates:
        ranked["candidate_rank"] = pd.to_numeric(
            candidates["candidate_rank"], errors="coerce"
        )
    else:
        ranked["candidate_rank"] = ranked.groupby("profile_id").cumcount() + 1
    if k is not None:
        ranked = ranked[ranked["candidate_rank"].le(k)]
    candidate_keys = set(
        ranked[["profile_id", "content_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    work = truth[["profile_id", "content_id", gain_column]].copy()
    work["profile_id"] = work["profile_id"].astype(str)
    work["content_id"] = work["content_id"].astype(str)
    work[gain_column] = pd.to_numeric(
        work[gain_column], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    work["_hit"] = [
        (profile_id, content_id) in candidate_keys
        for profile_id, content_id in work[
            ["profile_id", "content_id"]
        ].itertuples(index=False, name=None)
    ]
    total_gain = float(work[gain_column].sum())
    hit_gain = float(work.loc[work["_hit"], gain_column].sum())
    per_profile: dict[str, float] = {}
    for profile_id, group in work.groupby("profile_id", sort=True):
        denominator = float(group[gain_column].sum())
        numerator = float(group.loc[group["_hit"], gain_column].sum())
        per_profile[str(profile_id)] = (
            numerator / denominator if denominator > 0 else 0.0
        )
    return hit_gain / total_gain if total_gain > 0 else 0.0, per_profile


def candidate_gain_recall_curve(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    gain_column: str,
    k_values: tuple[int, ...] = (100, 200, 500, 1000),
) -> dict[str, float]:
    return {
        f"candidate_{gain_column}_recall@{k}": candidate_gain_recall_by_profile(
            candidates, truth, gain_column, k=k
        )[0]
        for k in sorted(set(k_values))
    }


def ranking_map_recall_metrics(
    rankings: dict[str, list[tuple[str, float]]],
    truth: pd.DataFrame,
    k: int,
    gain_columns: tuple[str, ...] = ("completion_gain", "watch_time_gain"),
) -> dict[str, float | int]:
    """Measure a retriever ranking map without materializing a pair DataFrame."""
    if k < 1:
        raise ValueError("ranking-map recall K must be positive")
    required = {"profile_id", "content_id"}
    if not required.issubset(truth):
        raise ValueError("ranking-map recall requires profile_id and content_id")
    selected = {
        str(profile_id): {
            str(content_id) for content_id, _ in values[:k]
        }
        for profile_id, values in rankings.items()
    }
    work = truth.copy()
    work["profile_id"] = work["profile_id"].astype(str)
    work["content_id"] = work["content_id"].astype(str)
    work = work.drop_duplicates(["profile_id", "content_id"])
    hit_mask = np.asarray(
        [
            content_id in selected.get(profile_id, set())
            for profile_id, content_id in work[
                ["profile_id", "content_id"]
            ].itertuples(index=False, name=None)
        ],
        dtype=bool,
    )
    metrics: dict[str, float | int] = {
        f"external_candidate_recall@{k}": float(hit_mask.mean())
        if len(hit_mask)
        else 0.0,
        "external_truth_hits": int(hit_mask.sum()),
        "external_truth_pairs": int(len(work)),
    }
    for gain_column in gain_columns:
        if gain_column not in work:
            continue
        gains = pd.to_numeric(work[gain_column], errors="coerce").fillna(0.0).clip(
            lower=0.0
        )
        total = float(gains.sum())
        metrics[f"external_{gain_column}_recall@{k}"] = (
            float(gains[hit_mask].sum()) / total if total > 0 else 0.0
        )
    return metrics


def ranking_map_incremental_metrics(
    rankings: dict[str, list[tuple[str, float]]],
    incumbent_candidates: pd.DataFrame,
    truth: pd.DataFrame,
    k: int,
    gain_columns: tuple[str, ...] = ("completion_gain", "watch_time_gain"),
) -> dict[str, float | int]:
    """Report the oracle contribution of an external top-K over an incumbent set."""
    incumbent = {
        str(profile_id): set(group["content_id"].astype(str))
        for profile_id, group in incumbent_candidates.groupby("profile_id")
    }
    external = {
        str(profile_id): {str(content_id) for content_id, _ in values[:k]}
        for profile_id, values in rankings.items()
    }
    work = truth.copy()
    work["profile_id"] = work["profile_id"].astype(str)
    work["content_id"] = work["content_id"].astype(str)
    work = work.drop_duplicates(["profile_id", "content_id"])
    incumbent_hits: list[bool] = []
    external_hits: list[bool] = []
    for profile_id, content_id in work[
        ["profile_id", "content_id"]
    ].itertuples(index=False, name=None):
        incumbent_hits.append(content_id in incumbent.get(profile_id, set()))
        external_hits.append(content_id in external.get(profile_id, set()))
    incumbent_mask = np.asarray(incumbent_hits, dtype=bool)
    external_mask = np.asarray(external_hits, dtype=bool)
    incremental_mask = external_mask & ~incumbent_mask
    union_mask = external_mask | incumbent_mask
    metrics: dict[str, float | int] = {
        "external_incremental_truth_hits": int(incremental_mask.sum()),
        "external_oracle_union_truth_hits": int(union_mask.sum()),
        f"external_oracle_union_recall@{k}+incumbent": float(union_mask.mean())
        if len(union_mask)
        else 0.0,
    }
    for gain_column in gain_columns:
        if gain_column not in work:
            continue
        gains = pd.to_numeric(work[gain_column], errors="coerce").fillna(0.0).clip(
            lower=0.0
        )
        total = float(gains.sum())
        metrics[f"external_incremental_{gain_column}"] = float(
            gains[incremental_mask].sum()
        )
        metrics[f"external_oracle_union_{gain_column}_recall"] = (
            float(gains[union_mask].sum()) / total if total > 0 else 0.0
        )
    return metrics


def candidate_recall_curve(
    candidates: pd.DataFrame,
    positives: pd.DataFrame,
    k_values: tuple[int, ...] = (100, 200, 500, 1000),
) -> dict[str, float]:
    """Measure deterministic micro recall at actual candidate ranks."""
    required = {"profile_id", "content_id"}
    if not required.issubset(candidates) or not required.issubset(positives):
        raise ValueError("candidate recall curve requires profile_id and content_id")
    if any(k < 1 for k in k_values):
        raise ValueError("candidate recall K values must be positive")
    ranked = candidates[["profile_id", "content_id"]].copy()
    ranked["profile_id"] = ranked["profile_id"].astype(str)
    ranked["content_id"] = ranked["content_id"].astype(str)
    if "candidate_rank" in candidates:
        ranked["candidate_rank"] = pd.to_numeric(
            candidates["candidate_rank"], errors="coerce"
        )
    else:
        ranked["candidate_rank"] = ranked.groupby("profile_id").cumcount() + 1
    ranked = ranked.drop_duplicates(["profile_id", "content_id"], keep="first")
    truth = positives[["profile_id", "content_id"]].drop_duplicates().copy()
    truth["profile_id"] = truth["profile_id"].astype(str)
    truth["content_id"] = truth["content_id"].astype(str)
    matched = truth.merge(
        ranked,
        on=["profile_id", "content_id"],
        how="left",
    )
    return {
        f"candidate_recall@{k}": float(
            matched["candidate_rank"].le(k).sum() / max(len(matched), 1)
        )
        for k in sorted(set(k_values))
    }


def positive_hit_source_counts(
    candidates: pd.DataFrame,
    positives: pd.DataFrame,
) -> dict[str, int]:
    required = {"profile_id", "content_id", "retrieval_source"}
    if not required.issubset(candidates):
        raise ValueError("source attribution requires retrieval_source")
    truth = positives[["profile_id", "content_id"]].drop_duplicates().astype(str)
    ranked = candidates[
        ["profile_id", "content_id", "retrieval_source"]
    ].drop_duplicates(["profile_id", "content_id"], keep="first").copy()
    ranked["profile_id"] = ranked["profile_id"].astype(str)
    ranked["content_id"] = ranked["content_id"].astype(str)
    matched = truth.merge(ranked, on=["profile_id", "content_id"], how="inner")
    return {
        str(source): int(count)
        for source, count in matched["retrieval_source"].value_counts().items()
    }


def retrieval_truth_attribution(
    candidates: pd.DataFrame,
    positives: pd.DataFrame,
    truth_attributes: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Attribute final-pool truth hits without treating overlapping sources as additive."""

    keys = ["profile_id", "content_id"]
    required_candidates = {*keys, "retrieval_source"}
    if not required_candidates.issubset(candidates):
        raise ValueError("truth attribution requires candidate source columns")
    if not set(keys).issubset(positives):
        raise ValueError("truth attribution requires positive pair keys")

    gain_columns = [
        column
        for column in ("completion_gain", "watch_time_gain")
        if column in positives
    ]
    truth = positives[[*keys, *gain_columns]].copy()
    truth[keys] = truth[keys].astype(str)
    aggregations = {column: "max" for column in gain_columns}
    truth = (
        truth.groupby(keys, as_index=False, sort=True).agg(aggregations)
        if aggregations
        else truth.drop_duplicates(keys)
    )
    for column in gain_columns:
        truth[column] = pd.to_numeric(truth[column], errors="coerce").fillna(0.0)

    candidate_columns = [
        column
        for column in (
            *keys,
            "retrieval_source",
            "retrieval_sources",
            "is_incumbent_candidate",
            "is_external_candidate",
            "is_safety_anchor",
        )
        if column in candidates
    ]
    ranked = candidates[candidate_columns].drop_duplicates(keys, keep="first").copy()
    ranked[keys] = ranked[keys].astype(str)
    if "retrieval_sources" not in ranked:
        ranked["retrieval_sources"] = ranked["retrieval_source"]
    ranked["retrieval_sources"] = ranked["retrieval_sources"].fillna(
        ranked["retrieval_source"]
    )
    matched = truth.merge(ranked, on=keys, how="inner", validate="one_to_one")

    truth_pairs = int(len(truth))
    gain_totals = {
        column: float(truth[column].sum()) for column in gain_columns
    }

    def source_row(frame: pd.DataFrame) -> dict[str, float | int]:
        row: dict[str, float | int] = {
            "truth_hits": int(len(frame)),
            "truth_recall": len(frame) / max(truth_pairs, 1),
        }
        for column in gain_columns:
            hit_gain = float(frame[column].sum())
            row[f"{column}_hit_sum"] = hit_gain
            row[f"{column}_recall"] = (
                hit_gain / gain_totals[column]
                if gain_totals[column] > 0
                else 0.0
            )
        return row

    primary_source = {
        str(source): source_row(group)
        for source, group in matched.groupby(
            "retrieval_source", dropna=False, observed=True, sort=True
        )
    }
    exploded = matched.assign(
        _source=matched["retrieval_sources"].astype(str).str.split("|")
    ).explode("_source")
    exploded = exploded[exploded["_source"].astype(bool)]
    inclusive_witness = {
        str(source): source_row(group)
        for source, group in exploded.groupby(
            "_source", dropna=False, observed=True, sort=True
        )
    }
    exclusive = matched[
        ~matched["retrieval_sources"].astype(str).str.contains("|", regex=False)
    ]
    exclusive_source = {
        str(source): source_row(group)
        for source, group in exclusive.groupby(
            "retrieval_sources", dropna=False, observed=True, sort=True
        )
    }
    source_counts = exploded.groupby(
        keys, dropna=False, observed=True
    )["_source"].transform("count").clip(lower=1)
    exploded["_fractional_credit"] = 1.0 / source_counts
    fractional_credit: dict[str, dict[str, float]] = {}
    for source, group in exploded.groupby(
        "_source", dropna=False, observed=True, sort=True
    ):
        row = {
            "truth_hit_credit": float(group["_fractional_credit"].sum()),
        }
        row["truth_recall_credit"] = row["truth_hit_credit"] / max(
            truth_pairs, 1
        )
        for column in gain_columns:
            gain_credit = float(
                (group[column] * group["_fractional_credit"]).sum()
            )
            row[f"{column}_hit_credit"] = gain_credit
            row[f"{column}_recall_credit"] = (
                gain_credit / gain_totals[column]
                if gain_totals[column] > 0
                else 0.0
            )
        fractional_credit[str(source)] = row

    hit_keys = set(matched[keys].itertuples(index=False, name=None))
    attributed_truth = truth.copy()
    attributed_truth["_hit"] = [
        pair in hit_keys
        for pair in attributed_truth[keys].itertuples(index=False, name=None)
    ]
    cohorts: dict[str, dict[str, dict[str, float | int]]] = {}
    if truth_attributes is not None and not truth_attributes.empty:
        missing = sorted(set(keys) - set(truth_attributes))
        if missing:
            raise ValueError(
                f"truth attribution attributes are missing columns: {missing}"
            )
        attributes = truth_attributes.copy()
        attributes[keys] = attributes[keys].astype(str)
        attributes = attributes.drop_duplicates(keys, keep="last")
        attribute_columns = [column for column in attributes if column not in keys]
        attributed_truth = attributed_truth.merge(
            attributes, on=keys, how="left", validate="one_to_one"
        )
        for column in attribute_columns:
            by_value: dict[str, dict[str, float | int]] = {}
            for value, group in attributed_truth.groupby(
                column, dropna=False, observed=True, sort=True
            ):
                label = (
                    "null"
                    if pd.isna(value)
                    else str(value).lower()
                    if isinstance(value, (bool, np.bool_))
                    else str(value)
                )
                hits = group[group["_hit"]]
                row: dict[str, float | int] = {
                    "truth_pairs": int(len(group)),
                    "truth_hits": int(len(hits)),
                    "candidate_recall": len(hits) / max(len(group), 1),
                }
                for gain_column in gain_columns:
                    total_gain = float(group[gain_column].sum())
                    hit_gain = float(hits[gain_column].sum())
                    row[f"{gain_column}_sum"] = total_gain
                    row[f"{gain_column}_hit_sum"] = hit_gain
                    row[f"{gain_column}_recall"] = (
                        hit_gain / total_gain if total_gain > 0 else 0.0
                    )
                by_value[label] = row
            cohorts[column] = by_value

    flag_intersections: dict[str, dict[str, float | int]] = {}
    flag_columns = [
        column
        for column in (
            "is_incumbent_candidate",
            "is_external_candidate",
            "is_safety_anchor",
        )
        if column in matched
    ]
    if flag_columns:
        for values, group in matched.groupby(
            flag_columns, dropna=False, observed=True, sort=True
        ):
            values = values if isinstance(values, tuple) else (values,)
            label = "|".join(
                f"{column}={str(bool(value)).lower()}"
                for column, value in zip(flag_columns, values, strict=True)
            )
            flag_intersections[label] = source_row(group)

    return {
        "truth_pairs": truth_pairs,
        "truth_hits": int(len(matched)),
        "candidate_recall": len(matched) / max(truth_pairs, 1),
        "primary_source": primary_source,
        "inclusive_witness": inclusive_witness,
        "fractional_credit": fractional_credit,
        "exclusive_source": exclusive_source,
        "flag_intersections": flag_intersections,
        "cohorts": cohorts,
    }
