from __future__ import annotations

import numpy as np
import pandas as pd

from recsys_vnpt.config import DOMAIN_CONFIGS, DomainConfig, get_domain_config


REQUIRED_INTERACTION_COLUMNS = {"profile_id", "content_id", "duration"}
REQUIRED_ITEM_COLUMNS = {"content_id", "content_duration"}


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _aggregate_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    _require_columns(interactions, REQUIRED_INTERACTION_COLUMNS, "interactions")
    work = interactions.copy()
    work["duration"] = pd.to_numeric(work["duration"], errors="coerce").fillna(0.0)

    group_cols = ["profile_id", "content_id"]
    agg = {
        "duration": ["sum", "max", "count"],
    }
    if "username" in work.columns:
        agg["username"] = "first"

    grouped = work.groupby(group_cols, dropna=False).agg(agg).reset_index()
    grouped.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in grouped.columns.to_flat_index()
    ]
    grouped = grouped.rename(
        columns={
            "duration_sum": "total_duration",
            "duration_max": "max_duration",
            "duration_count": "watch_count",
            "username_first": "username",
        }
    )
    return grouped


def _policy_for_row(config: DomainConfig, row: pd.Series) -> DomainConfig | None:
    if not config.uses_family_label_policy:
        return config
    family = str(row.get("content_family", "")).lower()
    return DOMAIN_CONFIGS.get(family)


def _label_row(config: DomainConfig, row: pd.Series) -> tuple[int, str, str]:
    row_policy = _policy_for_row(config, row)
    if row_policy is None:
        return 0, "excluded_unsupported_family", "excluded"

    if not np.isfinite(row["max_watch_percent"]):
        return 0, "excluded_missing_duration", "excluded"

    threshold = row_policy.watch_percent_threshold
    if threshold is not None and row["max_watch_percent"] >= threshold:
        return 1, "positive_watch_percent", "observed_positive"
    if row_policy.repeat_watch_positive and int(row["watch_count"]) >= 2:
        return 1, "positive_repeat_watch", "observed_positive"
    return 0, "negative_low_watch", "observed_negative"


def build_labeled_interactions(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    domain: str,
) -> pd.DataFrame:
    """Aggregate implicit interactions and attach explicit label metadata."""

    config = get_domain_config(domain)
    _require_columns(items, REQUIRED_ITEM_COLUMNS, "items")

    aggregated = _aggregate_interactions(interactions)
    item_features = items.copy()
    item_features["content_id"] = item_features["content_id"].astype(str)
    item_features["content_duration"] = pd.to_numeric(
        item_features["content_duration"], errors="coerce"
    )
    aggregated["content_id"] = aggregated["content_id"].astype(str)

    merged = aggregated.merge(item_features, on="content_id", how="left")
    merged["max_watch_percent"] = merged["max_duration"] / merged["content_duration"]
    merged.loc[~np.isfinite(merged["max_watch_percent"]), "max_watch_percent"] = np.nan

    labels = merged.apply(lambda row: _label_row(config, row), axis=1)
    merged[["label", "label_reason", "sample_type"]] = pd.DataFrame(
        labels.tolist(), index=merged.index
    )
    merged["label"] = merged["label"].astype(int)

    preferred = [
        "profile_id",
        "username",
        "content_id",
        "label",
        "label_reason",
        "sample_type",
        "watch_count",
        "total_duration",
        "max_duration",
        "max_watch_percent",
    ]
    ordered = [col for col in preferred if col in merged.columns]
    ordered.extend(col for col in merged.columns if col not in ordered)
    return merged[ordered]


def finalize_merged_interactions(merged: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Aggregate a merged interaction-user-item frame and apply shared labels."""

    config = get_domain_config(domain)
    _require_columns(merged, REQUIRED_INTERACTION_COLUMNS | {"content_duration"}, "merged")

    work = merged.copy()
    work["duration"] = pd.to_numeric(work["duration"], errors="coerce").fillna(0.0)
    work["content_id"] = work["content_id"].astype(str)
    work["content_duration"] = pd.to_numeric(work["content_duration"], errors="coerce")

    group_cols = ["profile_id", "content_id"]
    feature_cols = [
        col
        for col in work.columns
        if col not in set(group_cols + ["duration", "label", "percent_duration", "watch_count"])
    ]
    agg_spec = {col: "first" for col in feature_cols}
    agg_spec["duration"] = ["sum", "max", "count"]

    grouped = work.groupby(group_cols, dropna=False).agg(agg_spec).reset_index()
    grouped.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple)
        else str(col)
        for col in grouped.columns.to_flat_index()
    ]
    rename = {
        "duration_sum": "total_duration",
        "duration_max": "max_duration",
        "duration_count": "watch_count",
    }
    rename.update({f"{col}_first": col for col in feature_cols})
    grouped = grouped.rename(columns=rename)
    grouped["max_watch_percent"] = grouped["max_duration"] / grouped["content_duration"]
    grouped.loc[~np.isfinite(grouped["max_watch_percent"]), "max_watch_percent"] = np.nan

    labels = grouped.apply(lambda row: _label_row(config, row), axis=1)
    grouped[["label", "label_reason", "sample_type"]] = pd.DataFrame(
        labels.tolist(), index=grouped.index
    )
    grouped["label"] = grouped["label"].astype(int)

    preferred = [
        "profile_id",
        "username",
        "content_id",
        "label",
        "label_reason",
        "sample_type",
        "watch_count",
        "total_duration",
        "max_duration",
        "max_watch_percent",
    ]
    ordered = [col for col in preferred if col in grouped.columns]
    ordered.extend(col for col in grouped.columns if col not in ordered)
    return grouped[ordered]
