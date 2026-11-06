from __future__ import annotations

import pandas as pd

from recsys_vnpt.labels import build_labeled_interactions


def build_training_frame(
    interactions: pd.DataFrame,
    users: pd.DataFrame,
    items: pd.DataFrame,
    domain: str,
) -> pd.DataFrame:
    """Build the shared labeled training frame from raw interaction, user, and item tables."""

    labeled = build_labeled_interactions(interactions, items, domain=domain)
    if "username" in labeled.columns and "username" in users.columns:
        labeled = labeled.merge(users, on="username", how="left")
    return labeled
