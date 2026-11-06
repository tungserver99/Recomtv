from __future__ import annotations

import random

import numpy as np
import pandas as pd

from recsys_vnpt.features import FeatureSchema


def prepare_feature_matrix(
    frame: pd.DataFrame,
    schema: FeatureSchema | None = None,
) -> tuple[np.ndarray, np.ndarray, FeatureSchema]:
    if "label" not in frame.columns:
        raise ValueError("Training frame must contain a label column")
    schema = FeatureSchema.fit(frame) if schema is None else schema
    matrix = schema.transform(frame).to_numpy(dtype=np.float32)
    labels = pd.to_numeric(frame["label"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return matrix, labels, schema


def profile_group_split(
    frame: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "profile_id" not in frame.columns:
        raise ValueError("profile_group_split requires a profile_id column")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    profiles = list(pd.Series(frame["profile_id"]).drop_duplicates())
    rng = random.Random(random_state)
    rng.shuffle(profiles)
    val_count = max(1, int(round(len(profiles) * test_size)))
    val_profiles = set(profiles[:val_count])

    val = frame[frame["profile_id"].isin(val_profiles)].copy()
    train = frame[~frame["profile_id"].isin(val_profiles)].copy()
    if train.empty and not val.empty:
        moved_profile = next(iter(val_profiles))
        train = val[val["profile_id"] == moved_profile].copy()
        val = val[val["profile_id"] != moved_profile].copy()
    return train.reset_index(drop=True), val.reset_index(drop=True)
