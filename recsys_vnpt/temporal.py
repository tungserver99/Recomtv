from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from recsys_vnpt.canonical import historical_behavior_features
from recsys_vnpt.config import DOMAIN_CONFIGS, get_domain_config
from recsys_vnpt.source_manifest import schema_hash


TARGET_PAIR_LEAKAGE_COLUMNS = {
    "duration",
    "watch_count",
    "total_duration",
    "max_duration",
    "max_watch_percent",
    "percent_duration",
    "watch_percent",
    "completion_gain",
    "watch_time_gain",
    "episode_count",
    "total_watch_seconds",
    "max_episode_watch_seconds",
}

TARGET_METADATA_COLUMNS = {
    "label",
    "label_reason",
    "sample_type",
    "sample_weight",
    "target_semantics",
    "completion_gain",
    "watch_time_gain",
    "episode_count",
    "total_watch_seconds",
    "max_episode_watch_seconds",
}


@dataclass(frozen=True)
class TemporalWindow:
    name: str
    history_end: pd.Timestamp
    label_start: pd.Timestamp
    label_end: pd.Timestamp
    split: str


def _day_end(value: str) -> pd.Timestamp:
    return (
        pd.Timestamp(value).normalize()
        + pd.offsets.Day(1)
        - pd.Timedelta(1, unit="ns")
    )


def derive_relative_temporal_windows(
    max_event_date: pd.Timestamp,
    time_train: int = 24,
    time_valid: int = 24,
    time_test: int = 24,
) -> tuple[TemporalWindow, ...]:
    """Create adjacent hourly windows ending at the exact latest event time."""
    for name, value in (("time_train", time_train), ("time_valid", time_valid), ("time_test", time_test)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    end = pd.Timestamp(max_event_date)
    if pd.isna(end):
        raise ValueError("max_event_date must be a valid timestamp")
    durations = [time_train, time_train, time_train, time_valid, time_test]
    boundaries = [end - pd.Timedelta(hours=sum(durations))]
    for hours in durations:
        boundaries.append(boundaries[-1] + pd.Timedelta(hours=hours))
    windows: list[TemporalWindow] = []
    names = ("train_1", "train_2", "train_3", "validation", "test")
    splits = ("train", "train", "train", "validation", "test")
    for index, (name, split) in enumerate(zip(names, splits)):
        label_start = boundaries[index]
        next_start = boundaries[index + 1]
        label_end = end if index == len(names) - 1 else next_start - pd.Timedelta(1, unit="ns")
        windows.append(
            TemporalWindow(
                name,
                label_start - pd.Timedelta(1, unit="ns"),
                label_start,
                label_end,
                split,
            )
        )
    return tuple(windows)


def write_temporal_windows(directory, windows: tuple[TemporalWindow, ...], time_config: dict | None = None) -> None:
    import json
    from pathlib import Path

    payload = [
        {
            "name": window.name,
            "history_end": str(window.history_end),
            "label_start": str(window.label_start),
            "label_end": str(window.label_end),
            "split": window.split,
        }
        for window in windows
    ]
    Path(directory).mkdir(parents=True, exist_ok=True)
    envelope = {"schema_version": 2, "time_config": time_config, "windows": payload}
    (Path(directory) / "windows.json").write_text(json.dumps(envelope, indent=2), encoding="utf-8")


def load_temporal_windows(directory) -> tuple[TemporalWindow, ...]:
    """Window definitions actually used to build the temporal artifacts.

    Falls back to the pinned defaults for legacy artifacts without
    windows.json so old evidence stays reproducible."""
    import json
    from pathlib import Path

    path = Path(directory) / "windows.json"
    if not path.exists():
        return DEFAULT_TEMPORAL_WINDOWS
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("windows", payload) if isinstance(payload, dict) else payload
    return tuple(
        TemporalWindow(
            str(row["name"]),
            pd.Timestamp(row["history_end"]),
            pd.Timestamp(row["label_start"]),
            pd.Timestamp(row["label_end"]),
            str(row["split"]),
        )
        for row in rows
    )


def temporal_window_fingerprint(windows, window_mode, time_train=24, time_valid=24, time_test=24) -> str:
    payload = {
        "window_mode": window_mode,
        "time_train": time_train,
        "time_valid": time_valid,
        "time_test": time_test,
        "windows": [asdict(window) for window in windows],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def validate_window_positives(labeled_windows: dict[str, pd.DataFrame]) -> None:
    for name, frame in labeled_windows.items():
        positives = int(pd.to_numeric(frame.get("label", pd.Series(dtype=float)), errors="coerce").fillna(0).eq(1).sum())
        if positives == 0:
            raise ValueError(f"window {name} has no positive samples")


DEFAULT_TEMPORAL_WINDOWS = (
    TemporalWindow(
        "train_1",
        _day_end("2024-12-31"),
        pd.Timestamp("2025-01-01"),
        _day_end("2025-01-10"),
        "train",
    ),
    TemporalWindow(
        "train_2",
        _day_end("2025-01-10"),
        pd.Timestamp("2025-01-11"),
        _day_end("2025-01-20"),
        "train",
    ),
    TemporalWindow(
        "train_3",
        _day_end("2025-01-20"),
        pd.Timestamp("2025-01-21"),
        _day_end("2025-01-31"),
        "train",
    ),
    TemporalWindow(
        "validation",
        _day_end("2025-01-31"),
        pd.Timestamp("2025-02-01"),
        _day_end("2025-02-10"),
        "validation",
    ),
    TemporalWindow(
        "test",
        _day_end("2025-02-10"),
        pd.Timestamp("2025-02-11"),
        _day_end("2025-02-20"),
        "test",
    ),
)


def require_leakage_safe_features(columns: Iterable[str]) -> None:
    leaked = sorted(set(columns) & TARGET_PAIR_LEAKAGE_COLUMNS)
    if leaked:
        raise ValueError(f"target-pair leakage features are not allowed: {leaked}")


def _event_time(interactions: pd.DataFrame, time_col: str) -> pd.Series:
    if time_col not in interactions:
        raise ValueError(f"interactions are missing {time_col}")
    return pd.to_datetime(interactions[time_col], errors="coerce")


def historical_profile_features(
    interactions: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    time_col: str = "create_date",
) -> pd.DataFrame:
    required = {"profile_id", "content_id", "duration", time_col}
    missing = sorted(required - set(interactions.columns))
    if missing:
        raise ValueError(f"interactions are missing columns: {missing}")
    work = interactions.copy()
    work["_event_time"] = _event_time(work, time_col)
    work = work[work["_event_time"].notna() & (work["_event_time"] <= pd.Timestamp(cutoff))]
    work["duration"] = pd.to_numeric(work["duration"], errors="coerce").fillna(0.0)
    grouped = work.groupby("profile_id", dropna=False).agg(
        historical_interaction_count=("content_id", "size"),
        historical_unique_contents=("content_id", "nunique"),
        historical_total_duration=("duration", "sum"),
        historical_mean_duration=("duration", "mean"),
        historical_last_event=("_event_time", "max"),
    ).reset_index()
    grouped["historical_recency_days"] = (
        pd.Timestamp(cutoff) - grouped.pop("historical_last_event")
    ).dt.total_seconds().div(86400).clip(lower=0)
    return grouped


def future_positive_pairs(
    interactions: pd.DataFrame,
    window: TemporalWindow,
    label_col: str = "label",
    time_col: str = "create_date",
) -> pd.DataFrame:
    required = {"profile_id", "content_id", time_col}
    missing = sorted(required - set(interactions.columns))
    if missing:
        raise ValueError(f"interactions are missing columns: {missing}")
    work = interactions.copy()
    work["_event_time"] = _event_time(work, time_col)
    work = work[work["_event_time"].between(window.label_start, window.label_end)]
    if label_col in work:
        labels = pd.to_numeric(work[label_col], errors="coerce").fillna(0).astype(int)
        work = work[labels.eq(1)]
    return work[["profile_id", "content_id"]].dropna().astype("string").drop_duplicates()


def label_future_window(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    window: TemporalWindow,
    domain: str,
    time_col: str = "create_date",
) -> pd.DataFrame:
    required = {"profile_id", "content_id", "duration", time_col}
    missing = sorted(required - set(interactions.columns))
    if missing or not {"content_id", "content_duration"}.issubset(items.columns):
        raise ValueError(f"temporal labeling inputs are missing columns: {missing}")
    work = interactions.copy()
    work["_event_time"] = _event_time(work, time_col)
    work = work[work["_event_time"].between(window.label_start, window.label_end)]
    work["duration"] = pd.to_numeric(work["duration"], errors="coerce").fillna(0.0)
    grouped = work.groupby(["profile_id", "content_id"], dropna=False).agg(
        watch_count=("duration", "size"),
        max_duration=("duration", "max"),
    ).reset_index()
    item_columns = ["content_id", "content_duration"] + (["content_family"] if "content_family" in items else [])
    metadata = items[item_columns].drop_duplicates("content_id", keep="last").copy()
    grouped["content_id"] = grouped["content_id"].astype("string")
    metadata["content_id"] = metadata["content_id"].astype("string")
    grouped = grouped.merge(metadata, on="content_id", how="inner")
    grouped["content_duration"] = pd.to_numeric(grouped["content_duration"], errors="coerce")
    grouped["max_watch_percent"] = grouped["max_duration"] / grouped["content_duration"]

    config = get_domain_config(domain)
    labels: list[int] = []
    for row in grouped.itertuples(index=False):
        policy = config
        if config.uses_family_label_policy:
            policy = DOMAIN_CONFIGS.get(str(getattr(row, "content_family", "")).lower())
        positive = False
        if policy is not None:
            threshold = policy.watch_percent_threshold
            positive = bool(threshold is not None and np.isfinite(row.max_watch_percent) and row.max_watch_percent >= threshold)
            positive = positive or bool(policy.repeat_watch_positive and row.watch_count >= 2)
        labels.append(int(positive))
    grouped["label"] = labels
    return grouped


def build_pair_features(
    pairs: pd.DataFrame,
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    users: pd.DataFrame | None,
    cutoff: str | pd.Timestamp,
    time_col: str = "create_date",
    behavior: pd.DataFrame | None = None,
    profile_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """``profile_features``, when given, replaces the history aggregation over
    ``interactions``, so callers scoring in chunks aggregate it once."""
    if profile_features is None:
        profile_features = historical_profile_features(interactions, cutoff, time_col=time_col)
    output = pairs.copy()
    output["profile_id"] = output["profile_id"].astype("string")
    output["content_id"] = output["content_id"].astype("string")
    output = output.merge(profile_features, on="profile_id", how="left")
    if behavior is not None:
        output = output.merge(
            historical_behavior_features(behavior, cutoff),
            on="profile_id",
            how="left",
        )
    metadata = items.drop_duplicates("content_id", keep="last").copy()
    metadata["content_id"] = metadata["content_id"].astype("string")
    output = output.merge(metadata, on="content_id", how="left")
    if users is not None and "username" in output and "username" in users:
        output = output.merge(users.drop_duplicates("username", keep="last"), on="username", how="left")
    for column in output.columns:
        if column.startswith("historical_") and column != "historical_rule_ids":
            output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)
    # These arrive from the raw catalog as strings. model_runner selects numeric
    # features with is_numeric_dtype, so leaving them as objects drops them
    # without a word even though they are on its allowlist.
    for column in ("content_duration", "content_publish_year"):
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    require_leakage_safe_features(
        [
            column
            for column in output.columns
            if column not in TARGET_METADATA_COLUMNS
        ]
    )
    return output


def deterministic_negative_samples(
    positives: pd.DataFrame,
    catalog: pd.DataFrame,
    consumed: pd.DataFrame | None = None,
    ratio: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    if ratio < 1:
        raise ValueError("ratio must be at least 1")
    required = {"profile_id", "content_id"}
    if not required.issubset(positives.columns) or "content_id" not in catalog:
        raise ValueError("negative sampling requires profile_id/content_id and catalog content_id")
    content_ids = sorted(catalog["content_id"].dropna().astype(str).unique())
    popularity = catalog.get("popularity", pd.Series(0.0, index=catalog.index))
    popularity_map = dict(zip(catalog["content_id"].astype(str), pd.to_numeric(popularity, errors="coerce").fillna(0.0)))
    ranked = sorted(content_ids, key=lambda cid: (popularity_map.get(cid, 0.0), cid))
    buckets = [list(bucket) for bucket in np.array_split(np.array(ranked, dtype=object), 4)]
    consumed_map: dict[str, set[str]] = {}
    if consumed is not None and not consumed.empty:
        consumed_map = {
            str(profile): set(group["content_id"].astype(str))
            for profile, group in consumed.groupby("profile_id")
        }
    positive_map = {
        str(profile): set(group["content_id"].astype(str))
        for profile, group in positives.groupby("profile_id")
    }
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for profile in sorted(positive_map):
        excluded = positive_map[profile] | consumed_map.get(profile, set())
        target = ratio * len(positive_map[profile])
        selected: list[str] = []
        shuffled = [list(rng.permutation(bucket)) for bucket in buckets]
        while len(selected) < target and any(shuffled):
            for bucket in shuffled:
                while bucket and bucket[-1] in excluded:
                    bucket.pop()
                if bucket and len(selected) < target:
                    cid = str(bucket.pop())
                    if cid not in selected:
                        selected.append(cid)
        rows.extend({"profile_id": profile, "content_id": cid, "label": 0, "sample_type": "sampled_negative"} for cid in selected)
    return pd.DataFrame(rows, columns=["profile_id", "content_id", "label", "sample_type"])


def observed_negative_pairs(labeled_pairs: pd.DataFrame) -> pd.DataFrame:
    """Return observed label-window pairs that did not satisfy the positive policy."""
    required = {"profile_id", "content_id", "label"}
    if not required.issubset(labeled_pairs):
        raise ValueError("observed negatives require profile_id, content_id, and label")
    output = labeled_pairs[
        pd.to_numeric(labeled_pairs["label"], errors="coerce").fillna(0).eq(0)
    ][["profile_id", "content_id"]].drop_duplicates()
    output["label"] = 0
    output["sample_type"] = "observed_low_watch"
    output["sample_weight"] = 1.0
    return output


def mixed_negative_samples(
    positives: pd.DataFrame,
    catalog: pd.DataFrame,
    consumed: pd.DataFrame | None = None,
    membership: pd.DataFrame | None = None,
    ratio: int = 4,
    seed: int = 42,
    impression_negatives: dict[str, list[str]] | None = None,
    impression_weight: float = 1.0,
) -> pd.DataFrame:
    """Sample same-rule, popularity-matched and uniform unseen negatives.

    When ``impression_negatives`` supplies shown-not-watched content for a
    profile, it replaces the popularity-matched slot (equal count, weight 1.0
    because it is observed evidence rather than a sampled contrast).
    """
    if ratio < 1:
        raise ValueError("ratio must be at least 1")
    required = {"profile_id", "content_id"}
    if not required.issubset(positives) or "content_id" not in catalog:
        raise ValueError("mixed negatives require profile/content IDs and a catalog")

    content_ids = sorted(catalog["content_id"].dropna().astype(str).unique())
    popularity = pd.to_numeric(
        catalog.get("popularity", pd.Series(0.0, index=catalog.index)),
        errors="coerce",
    ).fillna(0.0)
    popularity_map = dict(zip(catalog["content_id"].astype(str), popularity))
    ranked = sorted(content_ids, key=lambda value: (popularity_map.get(value, 0.0), value))
    quartiles = [list(map(str, values)) for values in np.array_split(ranked, 4)]
    quartile_by_content = {
        content_id: index
        for index, values in enumerate(quartiles)
        for content_id in values
    }

    rules_by_content: dict[str, set[str]] = {}
    contents_by_rule: dict[str, list[str]] = {}
    if membership is not None and not membership.empty:
        if not {"content_id", "rule_id"}.issubset(membership):
            raise ValueError("membership requires content_id and rule_id")
        normalized_membership = membership[["content_id", "rule_id"]].dropna().astype(str)
        rules_by_content = {
            content_id: set(group["rule_id"])
            for content_id, group in normalized_membership.groupby("content_id")
        }
        contents_by_rule = {
            rule_id: sorted(set(group["content_id"].astype(str)))
            for rule_id, group in normalized_membership.groupby("rule_id")
        }

    consumed_map = (
        {
            str(profile): set(group["content_id"].astype(str))
            for profile, group in consumed.groupby("profile_id")
        }
        if consumed is not None and not consumed.empty
        else {}
    )
    positive_map = {
        str(profile): set(group["content_id"].astype(str))
        for profile, group in positives.groupby("profile_id")
    }
    rows: list[dict[str, object]] = []
    for profile_id, group in positives.astype({"profile_id": str, "content_id": str}).groupby(
        "profile_id", sort=True
    ):
        excluded = consumed_map.get(str(profile_id), set()) | positive_map[str(profile_id)]
        selected: set[str] = set()
        ordered_positive_ids = sorted(group["content_id"].unique())
        profile_seed = int(
            hashlib.sha256(f"{seed}|{profile_id}".encode("utf-8")).hexdigest()[:8],
            16,
        )
        rng = np.random.default_rng(profile_seed)

        impression_pool = list(
            (impression_negatives or {}).get(str(profile_id), [])
        )
        impression_cursor = 0

        def take(
            candidates: Iterable[str],
            count: int,
            sample_type: str,
            weight: float = 0.25,
        ) -> None:
            if count <= 0:
                return
            pool = (
                candidates
                if isinstance(candidates, (list, tuple, np.ndarray))
                else sorted(map(str, candidates))
            )
            if not pool:
                return
            blocked = excluded | selected
            target_size = len(selected) + count
            probe_count = min(len(pool), max(32, count * 16))
            probe_indices = rng.choice(
                len(pool), size=probe_count, replace=False
            )

            def append(content_id: str) -> None:
                selected.add(content_id)
                rows.append(
                    {
                        "profile_id": str(profile_id),
                        "content_id": content_id,
                        "label": 0,
                        "sample_type": sample_type,
                        "sample_weight": weight,
                    }
                )

            for index in probe_indices:
                content_id = str(pool[int(index)])
                if content_id in blocked:
                    continue
                append(content_id)
                blocked.add(content_id)
                if len(selected) >= target_size:
                    return
            start = int(rng.integers(0, len(pool)))
            for offset in range(len(pool)):
                content_id = str(pool[(start + offset) % len(pool)])
                if content_id in blocked:
                    continue
                append(content_id)
                blocked.add(content_id)
                if len(selected) >= target_size:
                    return

        for positive_id in ordered_positive_ids:
            target_before = len(selected)
            same_rule_target = min(2, ratio)
            for rule_id in sorted(rules_by_content.get(positive_id, set())):
                remaining_same_rule = same_rule_target - (
                    len(selected) - target_before
                )
                if remaining_same_rule <= 0:
                    break
                take(
                    contents_by_rule.get(rule_id, []),
                    remaining_same_rule,
                    "same_rule_hard_negative",
                )
            if ratio >= 3:
                # Real shown-not-watched evidence replaces the popularity-matched
                # slot when available; the sampled bucket is the fallback so the
                # negative count per positive never changes.
                filled_impression = False
                while impression_cursor < len(impression_pool):
                    candidate = str(impression_pool[impression_cursor])
                    impression_cursor += 1
                    if candidate in excluded or candidate in selected:
                        continue
                    take([candidate], 1, "shown_not_watched", weight=impression_weight)
                    filled_impression = True
                    break
                if not filled_impression:
                    bucket = quartiles[quartile_by_content.get(positive_id, 0)]
                    take(bucket, 1, "popularity_matched_negative")
            if ratio >= 4:
                take(content_ids, 1, "uniform_negative")
            remaining_for_positive = ratio - (len(selected) - target_before)
            take(content_ids, remaining_for_positive, "uniform_fill_negative")

    return pd.DataFrame(
        rows,
        columns=[
            "profile_id",
            "content_id",
            "label",
            "sample_type",
            "sample_weight",
        ],
    )


def select_smoke_profile_cohort(
    labeled_windows: dict[str, pd.DataFrame],
    maximum_profiles: int | None,
    seed: int = 42,
) -> set[str] | None:
    """Select profiles after labels exist, prioritizing test and validation evidence."""
    if maximum_profiles is None:
        return None
    if maximum_profiles < 1:
        raise ValueError("maximum_profiles must be positive")
    rows: list[pd.DataFrame] = []
    split_weight = {"test": 4, "validation": 2}
    for name, frame in labeled_windows.items():
        positive = frame[
            pd.to_numeric(frame["label"], errors="coerce").fillna(0).eq(1)
        ][["profile_id"]].copy()
        positive["evidence_weight"] = split_weight.get(name, 1)
        rows.append(positive)
    if not rows:
        return set()
    evidence = pd.concat(rows, ignore_index=True)
    if evidence.empty:
        return set()
    summary = evidence.groupby(evidence["profile_id"].astype(str)).agg(
        evidence_weight=("evidence_weight", "sum"),
        positive_pairs=("profile_id", "size"),
    ).reset_index(names="profile_id")
    summary["_seed_order"] = summary["profile_id"].map(
        lambda value: hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()
    )
    selected = summary.sort_values(
        ["evidence_weight", "positive_pairs", "_seed_order", "profile_id"],
        ascending=[False, False, True, True],
        kind="stable",
    ).head(maximum_profiles)
    return set(selected["profile_id"].astype(str))


def split_aware_catalog_cap(
    items: pd.DataFrame,
    membership: pd.DataFrame,
    positive_frames: Iterable[pd.DataFrame],
    popularity: pd.Series | dict[str, float] | None,
    maximum_contents: int | None,
    minimum_rule_contents: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cap only after retaining all temporal positives and rule safety contents."""
    if maximum_contents is None:
        return items.copy(), membership.copy()
    if maximum_contents < 1:
        raise ValueError("maximum_contents must be positive")
    positive_ids = {
        str(content_id)
        for frame in positive_frames
        for content_id in frame["content_id"].dropna().astype(str)
    }
    if len(positive_ids) > maximum_contents:
        raise ValueError(
            f"content cap {maximum_contents} cannot retain {len(positive_ids)} positives"
        )
    scores = pd.Series(popularity, dtype=float) if popularity is not None else pd.Series(dtype=float)
    scores.index = scores.index.astype(str)
    normalized_membership = membership[["content_id", "rule_id"]].drop_duplicates().copy()
    normalized_membership["content_id"] = normalized_membership["content_id"].astype(str)
    normalized_membership["rule_id"] = normalized_membership["rule_id"].astype(str)
    normalized_membership["_score"] = (
        normalized_membership["content_id"].map(scores).fillna(0.0)
    )
    selected = set(positive_ids)
    for _, group in normalized_membership.sort_values(
        ["rule_id", "_score", "content_id"],
        ascending=[True, False, True],
        kind="stable",
    ).groupby("rule_id", sort=True):
        selected.update(group["content_id"].head(minimum_rule_contents))
    ranked = (
        items.assign(
            _score=items["content_id"].astype(str).map(scores).fillna(0.0)
        )
        .sort_values(["_score", "content_id"], ascending=[False, True], kind="stable")
        ["content_id"]
        .astype(str)
    )
    if len(selected) > maximum_contents:
        safety_only = selected - positive_ids
        ordered_safety = [content_id for content_id in ranked if content_id in safety_only]
        selected = positive_ids | set(
            ordered_safety[: maximum_contents - len(positive_ids)]
        )
    else:
        for content_id in ranked:
            if len(selected) >= maximum_contents:
                break
            selected.add(content_id)
    capped_items = items[items["content_id"].astype(str).isin(selected)].copy()
    capped_membership = membership[
        membership["content_id"].astype(str).isin(selected)
    ].copy()
    return capped_items.reset_index(drop=True), capped_membership.reset_index(drop=True)


def evaluation_candidates(
    positives: pd.DataFrame,
    catalog: pd.DataFrame,
    consumed: pd.DataFrame | None = None,
    negatives_per_profile: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    if negatives_per_profile < 1:
        raise ValueError("negatives_per_profile must be at least 1")
    if "content_id" not in catalog or not {"profile_id", "content_id"}.issubset(
        positives
    ):
        raise ValueError("evaluation candidates require profile/content IDs")
    content_ids = sorted(catalog["content_id"].dropna().astype(str).unique())
    popularity = pd.to_numeric(
        catalog.get("popularity", pd.Series(0.0, index=catalog.index)),
        errors="coerce",
    ).fillna(0.0)
    popularity_map = dict(zip(catalog["content_id"].astype(str), popularity))
    ranked = sorted(
        content_ids, key=lambda content_id: (popularity_map.get(content_id, 0), content_id)
    )
    buckets = [
        np.asarray(values, dtype=object)
        for values in np.array_split(np.asarray(ranked, dtype=object), 4)
    ]
    consumed_map = (
        {
            str(profile): set(group["content_id"].astype(str))
            for profile, group in consumed.groupby("profile_id")
        }
        if consumed is not None and not consumed.empty
        else {}
    )
    positive_map = {
        str(profile): set(group["content_id"].astype(str))
        for profile, group in positives.groupby("profile_id")
    }
    negative_profiles: list[str] = []
    negative_contents: list[str] = []
    for profile in sorted(positive_map):
        excluded = positive_map[profile] | consumed_map.get(profile, set())
        selected: list[str] = []
        selected_set: set[str] = set()
        profile_seed = int(
            hashlib.sha256(f"{seed}|{profile}".encode("utf-8")).hexdigest()[:8],
            16,
        )
        rng = np.random.default_rng(profile_seed)

        def draw(pool: np.ndarray, target: int) -> None:
            if target <= 0 or len(pool) == 0:
                return
            blocked = excluded | selected_set
            sample_size = min(
                len(pool), max(target * 2, target + len(blocked) + 16)
            )
            indices = rng.choice(len(pool), size=sample_size, replace=False)
            for index in indices:
                content_id = str(pool[int(index)])
                if content_id not in blocked:
                    selected.append(content_id)
                    selected_set.add(content_id)
                    blocked.add(content_id)
                    if len(selected_set) >= target_before + target:
                        return
            if len(selected_set) < target_before + target:
                start = int(rng.integers(0, len(pool)))
                for offset in range(len(pool)):
                    content_id = str(pool[(start + offset) % len(pool)])
                    if content_id in blocked:
                        continue
                    selected.append(content_id)
                    selected_set.add(content_id)
                    blocked.add(content_id)
                    if len(selected_set) >= target_before + target:
                        return

        quotient, remainder = divmod(negatives_per_profile, len(buckets))
        for bucket_index, bucket in enumerate(buckets):
            target_before = len(selected_set)
            draw(bucket, quotient + int(bucket_index < remainder))
        if len(selected) < negatives_per_profile:
            target_before = len(selected_set)
            draw(
                np.asarray(content_ids, dtype=object),
                negatives_per_profile - len(selected),
            )
        selected = selected[:negatives_per_profile]
        negative_profiles.extend([profile] * len(selected))
        negative_contents.extend(selected)

    has_graded_targets = any(
        column in positives for column in ("completion_gain", "watch_time_gain")
    )
    negatives = pd.DataFrame(
        {
            "profile_id": negative_profiles,
            "content_id": negative_contents,
            "label": 0,
            "sample_type": (
                "unobserved_sample" if has_graded_targets else "sampled_negative"
            ),
        }
    )
    target_columns = [
        column
        for column in (
            "completion_gain",
            "watch_time_gain",
            "target_semantics",
        )
        if column in positives
    ]
    positive_rows = positives[["profile_id", "content_id", *target_columns]].copy()
    positive_rows["label"] = 1
    positive_rows["sample_type"] = (
        "observed_engagement" if has_graded_targets else "future_positive"
    )
    if has_graded_targets:
        for column in ("completion_gain", "watch_time_gain"):
            if column not in negatives:
                negatives[column] = 0.0
        if "target_semantics" in positive_rows:
            semantics = positive_rows["target_semantics"].dropna()
            negatives["target_semantics"] = (
                str(semantics.iloc[0]) if len(semantics) else "engagement_v1"
            )
    return pd.concat([positive_rows, negatives], ignore_index=True).drop_duplicates(
        ["profile_id", "content_id"], keep="first"
    )


def exclude_consumed_pairs(
    pairs: pd.DataFrame,
    consumed: pd.DataFrame,
) -> pd.DataFrame:
    """Remove profile-content pairs that are already present in pre-cutoff history."""
    required = {"profile_id", "content_id"}
    if not required.issubset(pairs) or not required.issubset(consumed):
        raise ValueError("consumed-pair filtering requires profile_id and content_id")
    if pairs.empty or consumed.empty:
        return pairs.copy()
    blocked = consumed[["profile_id", "content_id"]].drop_duplicates().copy()
    blocked["profile_id"] = blocked["profile_id"].astype("string")
    blocked["content_id"] = blocked["content_id"].astype("string")
    blocked["_consumed"] = True
    work = pairs.copy()
    original_columns = list(work.columns)
    work["profile_id"] = work["profile_id"].astype("string")
    work["content_id"] = work["content_id"].astype("string")
    work = work.merge(
        blocked,
        on=["profile_id", "content_id"],
        how="left",
    )
    return work[~work["_consumed"].eq(True)][original_columns].reset_index(
        drop=True
    )


def persist_temporal_artifact(frame: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    row_hashes = pd.util.hash_pandas_object(frame, index=False, categorize=True).to_numpy()
    metadata = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "schema_hash": schema_hash(frame.columns),
        "content_hash": hashlib.sha256(row_hashes.tobytes()).hexdigest(),
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output
