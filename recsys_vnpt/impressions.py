"""Impression (shown-not-watched) ingest from MyTV behavior logs.

Behavior logs mix three id namespaces: VOD content ids, live-TV channel
numbers (2-4 digits, from channel_by_number / channel_ch_plus_minus) and
Google IMA ad-event payloads. Only display actions on VOD-shaped ids are
real impressions; see the measurement that
established this (catalog join 0.71-0.96 once scoped, versus 0.30 unscoped).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd


# Ordered by evidence quality: pure display events first, then `focus`
# (cursor landed on the item without playing it). Candidates are consumed in
# this order so the cleanest evidence is used before the noisier depth.
DISPLAY_ACTIONS = ("impressed", "poster", "focus")
PRIMARY_DISPLAY_ACTIONS = ("impressed", "poster")
MINIMUM_VOD_ID_LENGTH = 5
_USED_COLUMNS = ("profile_id", "content_id", "action", "start_time")


def _read_behavior_part(path: Path) -> pd.DataFrame:
    # Files are named .gz.csv but are plain CSV text.
    frame = pd.read_csv(
        path,
        dtype=str,
        on_bad_lines="skip",
        compression=None,
        usecols=lambda column: column in _USED_COLUMNS,
    )
    missing = sorted(set(_USED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"behavior part {path.name} is missing columns: {missing}")
    return frame


def build_impressions(
    behavior_files: Iterable[str | Path],
    catalog_content_ids: set[str],
    display_actions: tuple[str, ...] = DISPLAY_ACTIONS,
    batch_size: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Scope behavior rows to VOD display impressions and dedup per day.

    Returns one row per (profile_id, content_id, event_date) plus audit
    metrics describing how many rows each scoping stage removed. Files are
    processed in batches so the raw log (tens of millions of rows) is never
    held whole; only the deduped result accumulates.
    """
    paths = [Path(path) for path in behavior_files]
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if len(paths) > batch_size:
        parts: list[pd.DataFrame] = []
        totals: dict[str, int] = {}
        for start in range(0, len(paths), batch_size):
            frame, batch_metrics = build_impressions(
                paths[start : start + batch_size],
                catalog_content_ids,
                display_actions,
                batch_size=batch_size,
            )
            parts.append(frame)
            for key, value in batch_metrics.items():
                if isinstance(value, int):
                    totals[key] = totals.get(key, 0) + value
        impressions = (
            pd.concat(parts, ignore_index=True)
            .sort_values(
                ["profile_id", "content_id", "event_date", "source_tier"],
                kind="stable",
            )
            .drop_duplicates(
                ["profile_id", "content_id", "event_date"], keep="first"
            )
            .reset_index(drop=True)
        )
        totals.update(
            {
                "behavior_parts_read": len(paths),
                "impression_rows": int(len(impressions)),
                "impression_profiles": int(impressions["profile_id"].nunique()),
                "impression_contents": int(impressions["content_id"].nunique()),
                "impression_display_actions": list(display_actions),
                "impression_catalog_join_rate": (
                    totals.get("impression_catalog_matched_rows", 0)
                    / max(totals.get("impression_vod_shaped_rows", 0), 1)
                ),
                "impression_date_parse_rate": (
                    totals.get("impression_dated_rows", 0)
                    / max(totals.get("impression_catalog_matched_rows", 0), 1)
                ),
            }
        )
        return impressions, totals

    frames = [_read_behavior_part(path) for path in paths]
    raw = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=list(_USED_COLUMNS))
    )
    raw_rows = int(len(raw))

    work = raw[raw["action"].isin(display_actions)].copy()
    display_rows = int(len(work))

    work["content_id"] = work["content_id"].astype("string")
    work = work[
        work["content_id"].notna()
        & work["content_id"].str.len().ge(MINIMUM_VOD_ID_LENGTH)
        & work["content_id"].str.isdigit().fillna(False)
    ]
    vod_rows = int(len(work))

    work = work[work["content_id"].astype(str).isin(catalog_content_ids)]
    catalog_rows = int(len(work))

    # start_time is "YYYYMMDD HH:MM:SS"; unparseable rows are dropped and counted.
    event_time = pd.to_datetime(work["start_time"], errors="coerce")
    work = work.assign(event_time=event_time)
    work = work[work["event_time"].notna()]
    dated_rows = int(len(work))

    work["profile_id"] = work["profile_id"].astype(str)
    work["content_id"] = work["content_id"].astype(str)
    work["event_date"] = work["event_time"].dt.normalize()
    work["source_tier"] = (
        ~work["action"].isin(PRIMARY_DISPLAY_ACTIONS)
    ).astype("int8")
    impressions = (
        work[["profile_id", "content_id", "event_date", "source_tier"]]
        .sort_values(
            ["profile_id", "content_id", "event_date", "source_tier"],
            kind="stable",
        )
        .drop_duplicates(["profile_id", "content_id", "event_date"], keep="first")
        .reset_index(drop=True)
    )

    metrics = {
        "behavior_parts_read": len(paths),
        "behavior_raw_rows": raw_rows,
        "impression_display_rows": display_rows,
        "impression_vod_shaped_rows": vod_rows,
        "impression_catalog_matched_rows": catalog_rows,
        "impression_dated_rows": dated_rows,
        "impression_rows": int(len(impressions)),
        "impression_profiles": int(impressions["profile_id"].nunique()),
        "impression_contents": int(impressions["content_id"].nunique()),
        "impression_display_actions": list(display_actions),
        "impression_catalog_join_rate": (
            catalog_rows / vod_rows if vod_rows else 0.0
        ),
        "impression_date_parse_rate": (
            dated_rows / catalog_rows if catalog_rows else 0.0
        ),
    }
    return impressions, metrics


def window_impression_negatives(
    impressions: pd.DataFrame,
    watched_pairs: pd.DataFrame,
    label_start: pd.Timestamp,
    label_end: pd.Timestamp,
    consumed: pd.DataFrame | None = None,
) -> dict[str, list[str]]:
    """Shown-not-watched candidates per profile inside one label window.

    A pair qualifies when the content was displayed to the profile during the
    window, the profile did not watch it in that window, and the pair was not
    already consumed before the window (same exclusion the positives use).
    """
    required = {"profile_id", "content_id", "event_date"}
    missing = sorted(required - set(impressions))
    if missing:
        raise ValueError(f"impressions are missing columns: {missing}")
    if impressions.empty:
        return {}

    dates = pd.to_datetime(impressions["event_date"], errors="coerce")
    scoped = impressions[dates.between(label_start, label_end)].copy()
    if scoped.empty:
        return {}
    scoped["profile_id"] = scoped["profile_id"].astype(str)
    scoped["content_id"] = scoped["content_id"].astype(str)

    blocked: set[tuple[str, str]] = set()
    for frame in (watched_pairs, consumed):
        if frame is None or frame.empty:
            continue
        if not {"profile_id", "content_id"}.issubset(frame):
            raise ValueError("exclusion frames require profile_id/content_id")
        blocked.update(
            frame[["profile_id", "content_id"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )

    tiers = (
        scoped["source_tier"].astype(int)
        if "source_tier" in scoped
        else pd.Series(0, index=scoped.index)
    )
    scoped = scoped.assign(source_tier=tiers)
    output: dict[str, list[str]] = {}
    for profile_id, group in scoped.groupby("profile_id", sort=True):
        ranked = (
            group.groupby("content_id", sort=True)["source_tier"]
            .min()
            .sort_values(kind="stable")
        )
        candidates = [
            content_id
            for content_id in ranked.index.astype(str)
            if (profile_id, content_id) not in blocked
        ]
        if candidates:
            output[profile_id] = candidates
    return output
