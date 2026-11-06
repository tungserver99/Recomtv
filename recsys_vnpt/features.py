from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_EXCLUDE_COLUMNS = {
    "username",
    "profile_id",
    "content_id",
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
class FeatureSchema:
    feature_columns: list[str]

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        exclude_columns: set[str] | None = None,
    ) -> "FeatureSchema":
        excluded = DEFAULT_EXCLUDE_COLUMNS if exclude_columns is None else exclude_columns
        columns: list[str] = []
        for column in df.columns:
            if column in excluded:
                continue
            converted = pd.to_numeric(df[column], errors="coerce")
            if converted.notna().any():
                columns.append(column)
        return cls(columns)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=df.index)
        for column in self.feature_columns:
            if column in df.columns:
                output[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
            else:
                output[column] = 0.0
        return output.astype("float32")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"feature_columns": self.feature_columns}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureSchema":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(feature_columns=list(data["feature_columns"]))
