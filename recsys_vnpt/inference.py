from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def filter_recommendable_items(
    items: pd.DataFrame,
    consumed_content_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Apply safe default candidate filters before model scoring."""

    consumed = {str(item) for item in consumed_content_ids or set()}
    filtered = items.copy()
    filtered["content_id"] = filtered["content_id"].astype(str)

    if "content_status" in filtered.columns:
        filtered = filtered[filtered["content_status"].astype(str) == "1"]
    if "tag_names" in filtered.columns:
        filtered = filtered[filtered["tag_names"].fillna("").astype(str).str.contains(r"\w", regex=True)]
    if "recommendable" in filtered.columns:
        filtered = filtered[filtered["recommendable"].astype(bool)]
    if consumed:
        filtered = filtered[~filtered["content_id"].isin(consumed)]

    return filtered.reset_index(drop=True)
