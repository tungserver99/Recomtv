from __future__ import annotations

import pandas as pd


class PopularityBaseline:
    def __init__(self) -> None:
        self._scores: dict[str, float] = {}

    def fit(self, training_frame: pd.DataFrame) -> "PopularityBaseline":
        if not {"content_id", "label"}.issubset(training_frame.columns):
            raise ValueError("PopularityBaseline.fit requires content_id and label columns")
        positives = training_frame[pd.to_numeric(training_frame["label"], errors="coerce").fillna(0).astype(int) == 1]
        counts = positives.groupby("content_id").size().sort_values(ascending=False)
        self._scores = {str(content_id): float(score) for content_id, score in counts.items()}
        return self

    def recommend(
        self,
        candidates: pd.DataFrame,
        consumed_content_ids: set[str] | None = None,
        top_n: int = 200,
    ) -> list[str]:
        if "content_id" not in candidates.columns:
            raise ValueError("recommend requires candidates with a content_id column")
        consumed = {str(content_id) for content_id in consumed_content_ids or set()}
        content_ids = [str(content_id) for content_id in candidates["content_id"].tolist()]
        content_ids = [content_id for content_id in content_ids if content_id not in consumed]
        content_ids.sort(key=lambda content_id: (-self._scores.get(content_id, 0.0), content_id))
        return content_ids[:top_n]
