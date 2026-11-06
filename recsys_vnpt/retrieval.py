from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from recsys_vnpt.rule_assembly import SPECIAL_RULES


EXTERNAL_REPLACEMENT_WITNESS_SOURCES = frozenset(
    {
        "per_rule_popularity",
        "co_watch",
        "deep_rule_popularity",
        "catalog_recency",
    }
)


def pair_keys_hash(
    frame: pd.DataFrame,
    columns: tuple[str, str] = ("profile_id", "content_id"),
) -> str:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"pair hash is missing columns: {missing}")
    hash_frame = (
        frame[list(columns)]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values(list(columns), kind="stable")
    )
    return hashlib.sha256(
        pd.util.hash_pandas_object(
            hash_frame, index=False, categorize=True
        ).to_numpy().tobytes()
    ).hexdigest()


def _content_set_digest(content_ids: set[str]) -> bytes:
    digest = hashlib.sha256()
    for content_id in sorted(map(str, content_ids)):
        encoded = content_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.digest()


def _profile_pair_set_hash(
    profile_digests: list[tuple[str, bytes]],
) -> str:
    digest = hashlib.sha256()
    for profile_id, content_digest in sorted(profile_digests):
        encoded = str(profile_id).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(content_digest)
    return digest.hexdigest()


def per_profile_score_ranks(
    candidates: pd.DataFrame,
    scores: np.ndarray | pd.Series | list[float],
) -> np.ndarray:
    """Dense 1-based rank of scores within each profile, deterministic on ties."""

    ranked = candidates[["profile_id", "content_id"]].copy()
    ranked["_row_order"] = np.arange(len(ranked), dtype=np.int64)
    ranked["_base_score"] = np.asarray(scores, dtype=float).reshape(-1)
    ranked = ranked.sort_values(
        ["profile_id", "_base_score", "content_id", "_row_order"],
        ascending=[True, False, True, True],
        kind="stable",
    )
    ranked["_base_rank"] = ranked.groupby(
        "profile_id", sort=False, observed=True
    ).cumcount() + 1
    return (
        ranked.sort_values("_row_order", kind="stable")["_base_rank"]
        .to_numpy(dtype=float)
    )


def fuse_content_scores_with_external_rank(
    candidates: pd.DataFrame,
    base_scores: np.ndarray | pd.Series | list[float],
    external_weight: float,
    rrf_constant: int = 60,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fuse a serving scorer rank with external retrieval rank without calibration."""

    required = {"profile_id", "content_id"}
    missing = sorted(required - set(candidates))
    if missing:
        raise ValueError(f"content score fusion is missing columns: {missing}")
    if external_weight < 0:
        raise ValueError("content external RRF weight cannot be negative")
    if rrf_constant < 1:
        raise ValueError("content RRF constant must be positive")
    base = np.asarray(base_scores, dtype=float).reshape(-1)
    if len(base) != len(candidates):
        raise ValueError("content score fusion length does not match candidates")
    if not np.isfinite(base).all():
        raise ValueError("content base scores must be finite")
    if external_weight == 0:
        return base.astype(np.float32), {
            "content_score_fusion": "disabled",
            "content_external_rrf_weight": 0.0,
            "content_rrf_constant": int(rrf_constant),
            "content_external_rank_coverage": 0.0,
        }

    base_rank = per_profile_score_ranks(candidates, base)
    external_rank = pd.to_numeric(
        candidates.get(
            "external_rank", pd.Series(np.nan, index=candidates.index)
        ),
        errors="coerce",
    ).to_numpy(dtype=float)
    valid_external = np.isfinite(external_rank) & (external_rank > 0)
    external_component = np.zeros(len(candidates), dtype=float)
    external_component[valid_external] = external_weight / (
        rrf_constant + external_rank[valid_external]
    )
    fused = 1.0 / (rrf_constant + base_rank) + external_component
    return fused.astype(np.float32), {
        "content_score_fusion": "base_rank_external_rank_rrf",
        "content_external_rrf_weight": float(external_weight),
        "content_rrf_constant": int(rrf_constant),
        "content_external_rank_coverage": float(valid_external.mean())
        if len(valid_external)
        else 0.0,
    }


@dataclass(frozen=True)
class RetrievalConfig:
    global_candidates: int = 1000
    safety_candidates_per_rule: int = 50
    minimum_rule_contents: int = 10
    minimum_normal_rules: int = 20
    candidate_budget: int = 1000
    candidate_rules: int = 40
    reserved_per_rule: int = 10
    source_candidates_per_rule: int = 20
    cowatch_candidates_per_rule: int = 0
    tag_candidates_per_rule: int = 0
    two_tower_candidates_per_rule: int = 200
    exclude_consumed: bool = True
    exact_cowatch_enabled: bool = False
    exact_cowatch_candidates: int = 1000
    external_candidate_mode: str = "rerank"
    external_candidate_slots: int = 100
    external_rrf_weight: float = 0.25
    external_rrf_constant: int = 60
    external_replacement_requires_history: bool = True
    external_replacement_requires_witness: bool = False
    recency_candidates_per_rule: int = 0
    recency_reserved_per_rule: int = 0
    recency_max_age_days: int = 0

    def __post_init__(self) -> None:
        if min(
            self.global_candidates,
            self.safety_candidates_per_rule,
            self.minimum_rule_contents,
            self.minimum_normal_rules,
            self.candidate_budget,
            self.candidate_rules,
            self.reserved_per_rule,
            self.source_candidates_per_rule,
            self.two_tower_candidates_per_rule,
            self.exact_cowatch_candidates,
        ) < 1:
            raise ValueError("retrieval limits must be positive")
        if self.candidate_rules < self.minimum_normal_rules:
            raise ValueError("candidate_rules must cover the requested normal rules")
        if (
            self.candidate_rules * self.reserved_per_rule
            > self.candidate_budget
        ):
            raise ValueError("reserved rule slots exceed the candidate budget")
        if self.external_candidate_mode not in {"rerank", "replace"}:
            raise ValueError("external_candidate_mode must be rerank or replace")
        if not 0 <= self.external_candidate_slots <= self.candidate_budget:
            raise ValueError("external candidate slots must fit the candidate budget")
        if self.external_rrf_weight < 0:
            raise ValueError("external RRF weight cannot be negative")
        if self.external_rrf_constant < 1:
            raise ValueError("external RRF constant must be positive")
        if min(
            self.recency_candidates_per_rule,
            self.recency_reserved_per_rule,
            self.recency_max_age_days,
        ) < 0:
            raise ValueError("recency retrieval limits cannot be negative")
        if self.recency_reserved_per_rule > self.reserved_per_rule:
            raise ValueError("recency reserved slots exceed per-rule reserved slots")
        if self.recency_reserved_per_rule > self.recency_candidates_per_rule:
            raise ValueError("recency reserved slots require enough recency candidates")
        if self.cowatch_candidates_per_rule < 0:
            raise ValueError("cowatch candidates per rule cannot be negative")
        if self.tag_candidates_per_rule < 0:
            raise ValueError("tag candidates per rule cannot be negative")


@dataclass
class CandidateBlendAudit:
    lost_incumbents: pd.DataFrame
    profiles: pd.DataFrame


def historical_content_popularity(
    interactions: pd.DataFrame,
    cutoff: str | pd.Timestamp | None = None,
) -> pd.Series:
    if "content_id" not in interactions:
        raise ValueError("interactions are missing content_id")
    work = interactions
    if cutoff is not None:
        if "create_date" not in work:
            raise ValueError("cutoff popularity requires create_date")
        event_time = pd.to_datetime(work["create_date"], errors="coerce")
        work = work[event_time <= pd.Timestamp(cutoff)]
    if "label" in work:
        labels = pd.to_numeric(work["label"], errors="coerce").fillna(0)
        work = work[labels.eq(1)]
    return work.groupby(work["content_id"].astype("string")).size().astype(float)


def profile_rule_affinity_scores(
    profiles: pd.DataFrame,
    membership: pd.DataFrame,
    popularity: pd.Series | dict[str, float] | None,
    interactions: pd.DataFrame | None = None,
    cutoff: str | pd.Timestamp | None = None,
    affinity: str = "count",
    half_life_days: float | None = None,
) -> pd.DataFrame:
    """Score every normal rule cheaply before content reranking.

    ``affinity="count"`` weighs every historical event equally (incumbent).
    ``affinity="gain"`` weighs events by watched seconds (``duration``), the
    same interaction-log signal the graded truth is built from. An optional
    ``half_life_days`` applies exponential recency decay relative to cutoff.
    """
    if "profile_id" not in profiles or not {"content_id", "rule_id"}.issubset(
        membership
    ):
        raise ValueError("rule affinity requires profiles and content-rule membership")
    if affinity not in {"count", "gain"}:
        raise ValueError("rule affinity must be count or gain")
    if half_life_days is not None and half_life_days <= 0:
        raise ValueError("rule affinity half-life must be positive")
    if half_life_days is not None and cutoff is None:
        raise ValueError("rule affinity recency decay requires a cutoff")
    rules = membership[["content_id", "rule_id"]].drop_duplicates().astype(str)
    rules = rules[~rules["rule_id"].isin(SPECIAL_RULES)]
    scores = pd.Series(popularity, dtype=float) if popularity is not None else pd.Series(dtype=float)
    scores.index = scores.index.astype(str)
    rules["_content_score"] = rules["content_id"].map(scores).fillna(0.0)
    base = rules.groupby("rule_id")["_content_score"].sum().astype(float)
    if not base.empty and float(base.max()) > 0:
        base = base / float(base.max())

    history_events = pd.DataFrame(columns=["profile_id", "rule_id", "events"])
    if interactions is not None and not interactions.empty:
        history = interactions.copy()
        if cutoff is not None:
            history = history[
                pd.to_datetime(history["create_date"], errors="coerce")
                <= pd.Timestamp(cutoff)
            ]
        requested = set(profiles["profile_id"].astype(str))
        history["profile_id"] = history["profile_id"].astype(str)
        history["content_id"] = history["content_id"].astype(str)
        history = history[history["profile_id"].isin(requested)]
        if affinity == "gain":
            if "duration" not in history:
                raise ValueError("gain rule affinity requires interaction duration")
            history["_weight"] = (
                pd.to_numeric(history["duration"], errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0)
            )
        else:
            history["_weight"] = 1.0
        if half_life_days is not None:
            age_days = (
                pd.Timestamp(cutoff)
                - pd.to_datetime(history["create_date"], errors="coerce")
            ).dt.total_seconds() / 86400.0
            history["_weight"] *= np.power(
                0.5,
                age_days.clip(lower=0.0).fillna(0.0) / float(half_life_days),
            )
        history_events = (
            history[["profile_id", "content_id", "_weight"]]
            .merge(rules[["content_id", "rule_id"]], on="content_id", how="inner")
            .groupby(["profile_id", "rule_id"])["_weight"]
            .sum()
            .rename("events")
            .reset_index()
        )
    event_map = {
        (str(row.profile_id), str(row.rule_id)): float(row.events)
        for row in history_events.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    normal_rules = sorted(base.index.astype(str))
    for profile_id in profiles["profile_id"].astype(str).drop_duplicates():
        raw = {
            rule_id: float(base.get(rule_id, 0.0))
            + float(np.log1p(event_map.get((profile_id, rule_id), 0.0)))
            for rule_id in normal_rules
        }
        maximum = max(raw.values(), default=1.0)
        rows.extend(
            {
                "profile_id": profile_id,
                "rule_id": rule_id,
                "rule_score": score / maximum if maximum > 0 else 0.0,
            }
            for rule_id, score in raw.items()
        )
    return pd.DataFrame(rows, columns=["profile_id", "rule_id", "rule_score"])


def _normalize_catalog(
    contents: pd.DataFrame,
    membership: pd.DataFrame,
    popularity: pd.Series | dict[str, float] | None,
) -> pd.DataFrame:
    required_content = {"content_id", "type_id"}
    required_membership = {"content_id", "rule_id"}
    if not required_content.issubset(contents) or not required_membership.issubset(membership):
        raise ValueError("retrieval requires content_id/type_id and content_id/rule_id")
    catalog = contents.drop_duplicates("content_id", keep="last").copy()
    catalog["content_id"] = catalog["content_id"].astype("string")
    if "content_status" in catalog:
        catalog = catalog[catalog["content_status"].astype(str).eq("1")]
    if "tag_names" in catalog:
        catalog = catalog[catalog["tag_names"].fillna("").astype(str).str.contains(r"\w", regex=True)]
    if "recommendable" in catalog:
        catalog = catalog[catalog["recommendable"].fillna(False).astype(bool)]
    popularity_map = pd.Series(popularity, dtype=float) if popularity is not None else pd.Series(dtype=float)
    popularity_map.index = popularity_map.index.astype("string")
    catalog["retrieval_score"] = catalog["content_id"].map(popularity_map).fillna(0.0).astype(float)
    rules = membership[["content_id", "rule_id"]].drop_duplicates().copy()
    rules["content_id"] = rules["content_id"].astype("string")
    rules["rule_id"] = rules["rule_id"].astype("string")
    rules = rules[~rules["rule_id"].isin(SPECIAL_RULES)]
    return catalog.merge(rules, on="content_id", how="inner")


def retrievable_content_ids(
    contents: pd.DataFrame,
    membership: pd.DataFrame,
    popularity: pd.Series | dict[str, float] | None,
    config: RetrievalConfig = RetrievalConfig(),
) -> set[str]:
    """Return content IDs that can participate in a quota-valid normal rule."""
    expanded = _normalize_catalog(contents, membership, popularity)
    counts = expanded.groupby("rule_id")["content_id"].nunique()
    eligible_rules = set(
        counts[counts >= config.minimum_rule_contents].index.astype(str)
    )
    return set(
        expanded[expanded["rule_id"].astype(str).isin(eligible_rules)][
            "content_id"
        ].astype(str)
    )


@dataclass(frozen=True)
class RetrievalTruth:
    raw: pd.DataFrame
    novel: pd.DataFrame
    eligible: pd.DataFrame
    metrics: dict[str, Any]


def prepare_retrieval_truth(
    positives: pd.DataFrame,
    interactions: pd.DataFrame,
    eligible_content_ids: set[str],
    cutoff: str | pd.Timestamp | None = None,
) -> RetrievalTruth:
    """Apply the production not-consumed/catalog policy to temporal positives."""
    required = {"profile_id", "content_id"}
    if not required.issubset(positives) or not required.issubset(interactions):
        raise ValueError("retrieval truth requires profile_id and content_id")
    raw = positives.copy()
    if "label" in raw:
        raw = raw[
            pd.to_numeric(raw["label"], errors="coerce").fillna(0).eq(1)
        ]
    preserved = [
        column
        for column in (
            "completion_gain",
            "watch_time_gain",
            "target_semantics",
        )
        if column in raw
    ]
    raw = raw[["profile_id", "content_id", *preserved]].dropna(
        subset=["profile_id", "content_id"]
    )
    raw["profile_id"] = raw["profile_id"].astype(str)
    raw["content_id"] = raw["content_id"].astype(str)
    aggregations: dict[str, str] = {
        column: ("max" if column.endswith("_gain") else "first")
        for column in preserved
    }
    if aggregations:
        raw = raw.groupby(
            ["profile_id", "content_id"], as_index=False, sort=True
        ).agg(aggregations)
    else:
        raw = raw.drop_duplicates(["profile_id", "content_id"])
    raw["label"] = 1
    raw["sample_type"] = (
        "observed_engagement" if preserved else "future_positive"
    )

    history = interactions.copy()
    if cutoff is not None:
        if "create_date" not in history:
            raise ValueError("retrieval truth cutoff requires create_date")
        history = history[
            pd.to_datetime(history["create_date"], errors="coerce")
            <= pd.Timestamp(cutoff)
        ]
    consumed_keys = set(
        history[["profile_id", "content_id"]]
        .dropna()
        .drop_duplicates()
        .astype(str)
        .itertuples(index=False, name=None)
    )
    repeat_mask = [
        (str(profile_id), str(content_id)) in consumed_keys
        for profile_id, content_id in raw[
            ["profile_id", "content_id"]
        ].itertuples(index=False, name=None)
    ]
    novel = raw.loc[~np.asarray(repeat_mask, dtype=bool)].copy()
    eligible_ids = set(map(str, eligible_content_ids))
    eligible = novel[novel["content_id"].astype(str).isin(eligible_ids)].copy()
    raw_profiles = set(raw["profile_id"].astype(str))
    eligible_profiles = set(eligible["profile_id"].astype(str))
    metrics = {
        "raw_positive_pairs": int(len(raw)),
        "repeat_positive_pairs_excluded": int(len(raw) - len(novel)),
        "novel_positive_pairs": int(len(novel)),
        "retrievable_positive_pairs": int(len(eligible)),
        "unavailable_novel_positive_pairs": int(len(novel) - len(eligible)),
        "candidate_truth_catalog_coverage": len(eligible) / max(len(novel), 1),
        "candidate_truth_policy_coverage": len(eligible) / max(len(raw), 1),
        "raw_truth_profiles": int(len(raw_profiles)),
        "retrievable_truth_profiles": int(len(eligible_profiles)),
        "truth_profiles_excluded_by_policy": int(
            len(raw_profiles - eligible_profiles)
        ),
    }
    for gain_column in ("completion_gain", "watch_time_gain"):
        if gain_column in raw:
            raw_gain = float(
                pd.to_numeric(raw[gain_column], errors="coerce").fillna(0.0).sum()
            )
            novel_gain = float(
                pd.to_numeric(novel[gain_column], errors="coerce")
                .fillna(0.0)
                .sum()
            )
            eligible_gain = float(
                pd.to_numeric(eligible[gain_column], errors="coerce")
                .fillna(0.0)
                .sum()
            )
            metrics.update(
                {
                    f"raw_{gain_column}_sum": raw_gain,
                    f"novel_{gain_column}_sum": novel_gain,
                    f"retrievable_{gain_column}_sum": eligible_gain,
                    f"candidate_truth_{gain_column}_coverage": (
                        eligible_gain / novel_gain if novel_gain > 0 else 0.0
                    ),
                }
            )
    return RetrievalTruth(raw=raw, novel=novel, eligible=eligible, metrics=metrics)


def balanced_catalog_cap(
    contents: pd.DataFrame,
    membership: pd.DataFrame,
    popularity: pd.Series | dict[str, float] | None,
    maximum_contents: int | None,
    config: RetrievalConfig = RetrievalConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expanded = _normalize_catalog(contents, membership, popularity)
    counts = expanded.groupby("rule_id")["content_id"].nunique()
    eligible = counts[counts >= config.minimum_rule_contents].index
    expanded = expanded[expanded["rule_id"].isin(eligible)].copy()
    if len(eligible) < config.minimum_normal_rules:
        raise ValueError(
            f"Only {len(eligible)} rules have at least {config.minimum_rule_contents} contents; "
            f"need {config.minimum_normal_rules}"
        )
    if maximum_contents is None or expanded["content_id"].nunique() <= maximum_contents:
        selected_ids = set(expanded["content_id"].astype(str))
    else:
        if maximum_contents < config.minimum_normal_rules * config.minimum_rule_contents:
            raise ValueError(
                "content cap is too small for the output contract: "
                f"need at least {config.minimum_normal_rules * config.minimum_rule_contents}"
            )
        rule_strength = expanded.groupby("rule_id")["retrieval_score"].sum().sort_values(ascending=False)
        selected_rules = list(rule_strength.head(config.minimum_normal_rules).index)
        selected_ids: set[str] = set()
        for rank in range(config.minimum_rule_contents):
            for rule_id in selected_rules:
                ranked = expanded[expanded["rule_id"].eq(rule_id)].sort_values(
                    ["retrieval_score", "content_id"], ascending=[False, True], kind="stable"
                )
                if rank < len(ranked):
                    selected_ids.add(str(ranked.iloc[rank]["content_id"]))
        global_ranked = expanded.sort_values(
            ["retrieval_score", "content_id"], ascending=[False, True], kind="stable"
        )["content_id"].drop_duplicates()
        for content_id in global_ranked.astype(str):
            if len(selected_ids) >= maximum_contents:
                break
            selected_ids.add(content_id)
    capped_contents = contents[contents["content_id"].astype("string").isin(selected_ids)].drop_duplicates("content_id", keep="last")
    capped_membership = membership[membership["content_id"].astype("string").isin(selected_ids)].drop_duplicates(["content_id", "rule_id"])
    return capped_contents.reset_index(drop=True), capped_membership.reset_index(drop=True)


def build_candidate_pool(
    contents: pd.DataFrame,
    membership: pd.DataFrame,
    popularity: pd.Series | dict[str, float] | None,
    config: RetrievalConfig = RetrievalConfig(),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    expanded = _normalize_catalog(contents, membership, popularity)
    counts = expanded.groupby("rule_id")["content_id"].nunique()
    eligible_rules = counts[counts >= config.minimum_rule_contents].index
    expanded = expanded[expanded["rule_id"].isin(eligible_rules)].copy()
    global_ids = expanded.sort_values(
        ["retrieval_score", "content_id"], ascending=[False, True], kind="stable"
    )["content_id"].drop_duplicates().head(config.global_candidates)
    safety = (
        expanded.sort_values(
            ["rule_id", "retrieval_score", "content_id"],
            ascending=[True, False, True],
            kind="stable",
        )
        .groupby("rule_id", sort=True)
        .head(config.safety_candidates_per_rule)
    )
    candidate_ids = set(global_ids.astype(str)) | set(safety["content_id"].astype(str))
    pool = contents[contents["content_id"].astype("string").isin(candidate_ids)].drop_duplicates("content_id", keep="last").copy()
    pool["content_id"] = pool["content_id"].astype("string")
    popularity_map = pd.Series(popularity, dtype=float) if popularity is not None else pd.Series(dtype=float)
    popularity_map.index = popularity_map.index.astype("string")
    pool["retrieval_score"] = pool["content_id"].map(popularity_map).fillna(0.0).astype(float)
    metrics = {
        "catalog_contents": int(contents["content_id"].nunique()),
        "eligible_normal_rules": int(len(eligible_rules)),
        "global_candidates": int(len(global_ids)),
        "safety_candidate_rows": int(len(safety)),
        "retrieval_union_contents": int(len(pool)),
    }
    return pool.reset_index(drop=True), metrics


def candidate_recall(
    candidate_content_ids: set[str],
    positives: pd.DataFrame,
) -> float:
    truth = positives["content_id"].dropna().astype(str)
    if truth.empty:
        return 0.0
    return float(truth.isin(set(map(str, candidate_content_ids))).mean())


def _tag_tokens(value: Any) -> set[str]:
    if pd.isna(value):
        return set()
    return {
        token.strip().casefold()
        for token in str(value).split(",")
        if token.strip()
    }


def _exact_cosine_cowatch_rankings(
    profile_ids: list[str],
    history: pd.DataFrame,
    eligible_content_ids: set[str],
    candidates_per_profile: int,
    batch_size: int = 128,
) -> dict[str, list[str]]:
    """Return exact sparse user-kNN rankings from pre-cutoff binary history."""
    from scipy.sparse import csr_matrix
    from sklearn.preprocessing import normalize

    rankings = {str(profile_id): [] for profile_id in profile_ids}
    if history.empty or not eligible_content_ids:
        return rankings
    events = history[["profile_id", "content_id"]].dropna().drop_duplicates().copy()
    events["profile_id"] = events["profile_id"].astype(str)
    events["content_id"] = events["content_id"].astype(str)
    items = sorted(set(map(str, eligible_content_ids)))
    item_index = {content_id: index for index, content_id in enumerate(items)}
    events = events[events["content_id"].isin(item_index)]
    if events.empty:
        return rankings
    users = sorted(events["profile_id"].unique())
    user_index = {profile_id: index for index, profile_id in enumerate(users)}
    matrix = csr_matrix(
        (
            np.ones(len(events), dtype=np.float32),
            (
                events["profile_id"].map(user_index).to_numpy(),
                events["content_id"].map(item_index).to_numpy(),
            ),
        ),
        shape=(len(users), len(items)),
    )
    matrix.data[:] = 1.0
    matrix.eliminate_zeros()
    normalized = normalize(matrix, norm="l2", axis=1, copy=True)
    requested = [str(profile_id) for profile_id in profile_ids]
    for start in range(0, len(requested), batch_size):
        batch_profiles = requested[start : start + batch_size]
        seen = [profile_id for profile_id in batch_profiles if profile_id in user_index]
        if not seen:
            continue
        row_indices = np.asarray(
            [user_index[profile_id] for profile_id in seen], dtype=np.int64
        )
        similarities = normalized[row_indices] @ normalized.T
        scores = (similarities @ matrix).tocsr()
        for local_index, profile_id in enumerate(seen):
            values = scores.getrow(local_index).toarray().ravel()
            values[matrix.getrow(user_index[profile_id]).indices] = -np.inf
            available = np.flatnonzero(np.isfinite(values) & (values > 0))
            if available.size == 0:
                continue
            limit = min(candidates_per_profile, int(available.size))
            if available.size > limit:
                selected = available[
                    np.argpartition(values[available], -limit)[-limit:]
                ]
            else:
                selected = available
            selected = selected[
                np.lexsort((selected, -values[selected]))
            ][:limit]
            rankings[profile_id] = [items[int(index)] for index in selected]
    return rankings


@dataclass(frozen=True)
class RetrievalIndex:
    """Catalog rankings and the co-watch lookup, independent of who is served.

    Built once per cutoff and shared by every profile shard: the co-watch
    lookup spans the whole history, so rebuilding it per shard would cost
    O(history) each time. Nothing in it is mutated while serving.
    """

    expanded: pd.DataFrame
    eligible_rules: set[str]
    content_tags: dict[str, set[str]]
    base_rule_score: pd.Series
    history: pd.DataFrame
    history_sets: dict[str, list[str]]
    content_profiles: dict[str, list[str]]
    static_ranked: pd.DataFrame
    rule_rankings: dict[str, list[str]]
    recency_rankings: dict[str, list[str]]
    rules_by_content: dict[str, set[str]]
    pop_score_by_content: dict[str, float]
    global_ranking: list[str]
    tag_rule_rankings: dict[tuple[str, str], list[str]]
    catalog_preparation_seconds: float
    history_preparation_seconds: float
    index_preparation_seconds: float


def prepare_retrieval_index(
    contents: pd.DataFrame,
    membership: pd.DataFrame,
    popularity: pd.Series | dict[str, float] | None,
    interactions: pd.DataFrame | None = None,
    cutoff: str | pd.Timestamp | None = None,
    config: RetrievalConfig = RetrievalConfig(),
) -> RetrievalIndex:
    retrieval_started = time.perf_counter()
    expanded = _normalize_catalog(contents, membership, popularity)
    counts = expanded.groupby("rule_id")["content_id"].nunique()
    eligible_rules = set(
        counts[counts >= config.minimum_rule_contents].index.astype(str)
    )
    if len(eligible_rules) < config.minimum_normal_rules:
        raise ValueError(
            f"Only {len(eligible_rules)} rules satisfy the content quota"
        )
    expanded = expanded[expanded["rule_id"].astype(str).isin(eligible_rules)].copy()
    maximum_popularity = float(expanded["retrieval_score"].max())
    expanded["_popularity_score"] = (
        expanded["retrieval_score"] / maximum_popularity
        if maximum_popularity > 0
        else 0.0
    )
    # tag_candidates_per_rule=0 disables the tag-affinity channel. Enabling it
    # at source-cap allocation costs candidate recall by displacing
    # deep-popularity slots, so it stays an explicit opt-in knob.
    content_tags = (
        {
            str(row.content_id): _tag_tokens(getattr(row, "tag_names", ""))
            for row in contents.drop_duplicates("content_id").itertuples(index=False)
        }
        if config.tag_candidates_per_rule > 0
        else {}
    )
    base_rule_score = (
        expanded.groupby("rule_id")["retrieval_score"].sum().astype(float)
    )
    if not base_rule_score.empty and float(base_rule_score.max()) > 0:
        base_rule_score = base_rule_score / float(base_rule_score.max())
    catalog_preparation_seconds = time.perf_counter() - retrieval_started

    history_started = time.perf_counter()
    history = pd.DataFrame(columns=["profile_id", "content_id"])
    if interactions is not None and not interactions.empty:
        history = interactions.copy()
        if cutoff is not None:
            if "create_date" not in history:
                raise ValueError("retrieval cutoff requires create_date")
            history = history[
                pd.to_datetime(history["create_date"], errors="coerce")
                <= pd.Timestamp(cutoff)
            ]
        history["profile_id"] = history["profile_id"].astype(str)
        history["content_id"] = history["content_id"].astype(str)

    # Lightweight co-watch lookup, bounded per target profile below.
    history_ranked = history.copy()
    history_ranked["_event_time"] = (
        pd.to_datetime(history_ranked["create_date"], errors="coerce")
        if "create_date" in history_ranked
        else pd.NaT
    )
    history_ranked = history_ranked.sort_values(
        ["profile_id", "_event_time", "content_id"],
        ascending=[True, False, True],
        kind="stable",
    )
    history_sets = {
        str(profile): group["content_id"]
        .astype(str)
        .drop_duplicates()
        .head(100)
        .tolist()
        for profile, group in history_ranked.groupby("profile_id", sort=True)
    }
    content_profiles: dict[str, list[str]] = {}
    for profile, content_ids in history_sets.items():
        for content_id in content_ids[:20]:
            content_profiles.setdefault(content_id, []).append(profile)
    history_preparation_seconds = time.perf_counter() - history_started

    index_started = time.perf_counter()
    static_ranked = expanded.sort_values(
        ["rule_id", "_popularity_score", "content_id"],
        ascending=[True, False, True],
        kind="stable",
    ).drop_duplicates(["rule_id", "content_id"])
    rule_rankings = {
        str(rule_id): group["content_id"].astype(str).tolist()
        for rule_id, group in static_ranked.groupby("rule_id", sort=True)
    }
    recency_rankings: dict[str, list[str]] = {}
    if config.recency_candidates_per_rule > 0:
        recency_column = next(
            (
                column
                for column in ("publish_date", "_publish_date", "start_time")
                if column in expanded
            ),
            None,
        )
        if recency_column is not None:
            recency_ranked = expanded.copy()
            recency_ranked["_recency_time"] = pd.to_datetime(
                recency_ranked[recency_column], errors="coerce"
            )
            if cutoff is not None:
                recency_cutoff = pd.Timestamp(cutoff)
                if config.recency_max_age_days > 0:
                    recency_ranked = recency_ranked[
                        recency_ranked["_recency_time"].between(
                            recency_cutoff
                            - pd.to_timedelta(
                                int(config.recency_max_age_days), unit="D"
                            ),
                            recency_cutoff,
                            inclusive="both",
                        )
                    ]
                else:
                    recency_ranked = recency_ranked[
                        recency_ranked["_recency_time"] <= recency_cutoff
                    ]
            recency_ranked = recency_ranked.sort_values(
                ["rule_id", "_recency_time", "_popularity_score", "content_id"],
                ascending=[True, False, False, True],
                kind="stable",
                na_position="last",
            ).drop_duplicates(["rule_id", "content_id"])
            recency_rankings = {
                str(rule_id): group["content_id"].astype(str).tolist()
                for rule_id, group in recency_ranked.groupby("rule_id", sort=True)
            }
    rules_by_content = {
        str(content_id): set(group["rule_id"].astype(str))
        for content_id, group in expanded.groupby("content_id")
    }
    pop_score_by_content = (
        expanded.groupby("content_id")["_popularity_score"].max().to_dict()
    )
    global_ranking = (
        expanded.sort_values(
            ["_popularity_score", "content_id"],
            ascending=[False, True],
            kind="stable",
        )["content_id"]
        .drop_duplicates()
        .astype(str)
        .tolist()
    )
    tag_rule_rankings: dict[tuple[str, str], list[str]] = {}
    for row in static_ranked[["content_id", "rule_id"]].itertuples(index=False):
        for tag in content_tags.get(str(row.content_id), set()):
            tag_rule_rankings.setdefault((str(row.rule_id), tag), []).append(
                str(row.content_id)
            )
    index_preparation_seconds = time.perf_counter() - index_started
    return RetrievalIndex(
        expanded=expanded,
        eligible_rules=eligible_rules,
        content_tags=content_tags,
        base_rule_score=base_rule_score,
        history=history,
        history_sets=history_sets,
        content_profiles=content_profiles,
        static_ranked=static_ranked,
        rule_rankings=rule_rankings,
        recency_rankings=recency_rankings,
        rules_by_content=rules_by_content,
        pop_score_by_content=pop_score_by_content,
        global_ranking=global_ranking,
        tag_rule_rankings=tag_rule_rankings,
        catalog_preparation_seconds=catalog_preparation_seconds,
        history_preparation_seconds=history_preparation_seconds,
        index_preparation_seconds=index_preparation_seconds,
    )


def build_personalized_candidate_pools(
    profiles: pd.DataFrame,
    contents: pd.DataFrame,
    membership: pd.DataFrame,
    popularity: pd.Series | dict[str, float] | None,
    interactions: pd.DataFrame | None = None,
    cutoff: str | pd.Timestamp | None = None,
    config: RetrievalConfig = RetrievalConfig(),
    external_rankings: dict[str, list[tuple[str, float]]] | None = None,
    external_source: str = "hybrid_two_tower",
    capture_blend_audit: bool = False,
    evaluation_truth: pd.DataFrame | None = None,
    index: RetrievalIndex | None = None,
) -> (
    tuple[pd.DataFrame, dict[str, Any]]
    | tuple[pd.DataFrame, dict[str, Any], CandidateBlendAudit]
):
    """Retrieve a deterministic <=budget catalog union for each profile."""
    retrieval_started = time.perf_counter()
    if "profile_id" not in profiles:
        raise ValueError("personalized retrieval requires profile_id")
    if index is None:
        index = prepare_retrieval_index(
            contents, membership, popularity, interactions, cutoff, config
        )
    expanded = index.expanded
    eligible_rules = index.eligible_rules
    content_tags = index.content_tags
    base_rule_score = index.base_rule_score
    history = index.history
    history_sets = index.history_sets
    content_profiles = index.content_profiles
    static_ranked = index.static_ranked
    rule_rankings = index.rule_rankings
    recency_rankings = index.recency_rankings
    rules_by_content = index.rules_by_content
    pop_score_by_content = index.pop_score_by_content
    global_ranking = index.global_ranking
    tag_rule_rankings = index.tag_rule_rankings
    catalog_preparation_seconds = index.catalog_preparation_seconds

    history_started = time.perf_counter()
    requested = set(profiles["profile_id"].astype(str))
    requested_history = history[history["profile_id"].isin(requested)]

    history_rules = (
        requested_history[["profile_id", "content_id"]]
        .merge(
            membership[["content_id", "rule_id"]].astype(str).drop_duplicates(),
            on="content_id",
            how="inner",
        )
        .groupby(["profile_id", "rule_id"])
        .size()
        .rename("events")
        .reset_index()
    )
    rule_events = {
        str(profile): {
            str(row.rule_id): float(row.events)
            for row in group.itertuples(index=False)
        }
        for profile, group in history_rules.groupby("profile_id")
    }
    consumed_map = {
        str(profile): set(group["content_id"].astype(str))
        for profile, group in requested_history.groupby("profile_id")
    }
    tag_preferences: dict[str, dict[str, int]] = {}
    for profile, content_ids in consumed_map.items():
        counts_by_tag: dict[str, int] = {}
        for content_id in content_ids:
            for tag in content_tags.get(content_id, set()):
                counts_by_tag[tag] = counts_by_tag.get(tag, 0) + 1
        tag_preferences[profile] = counts_by_tag

    history_preparation_seconds = index.history_preparation_seconds + (
        time.perf_counter() - history_started
    )

    index_started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    requested_profile_ids = list(
        profiles["profile_id"].astype(str).drop_duplicates()
    )
    exact_cowatch_rankings = (
        _exact_cosine_cowatch_rankings(
            requested_profile_ids,
            history,
            set(expanded["content_id"].astype(str)),
            config.exact_cowatch_candidates,
        )
        if config.exact_cowatch_enabled
        else {profile_id: [] for profile_id in requested_profile_ids}
    )
    exact_cowatch_profiles = sum(bool(values) for values in exact_cowatch_rankings.values())
    external_rankings = external_rankings or {}
    external_profiles = sum(
        bool(external_rankings.get(profile_id))
        for profile_id in requested_profile_ids
    )
    safety_slot_counts: list[int] = []
    anchor_retention_rates: list[float] = []
    incumbent_retention_rates: list[float] = []
    external_overlap_counts: list[int] = []
    external_promoted_counts: list[int] = []
    external_novel_counts: list[int] = []
    external_replacement_blocked_profiles = 0
    external_witness_rejected_candidates = 0
    external_witness_rejected_profiles = 0
    lost_incumbent_rows: list[dict[str, Any]] = []
    blend_profile_rows: list[dict[str, Any]] = []
    blend_pair_digests: dict[str, list[tuple[str, bytes]]] = {
        "preblend_incumbent": [],
        "blend_gained": [],
        "blend_lost": [],
        "blend_union": [],
    }
    truth_by_profile: dict[str, dict[str, tuple[float, float]]] = {}
    blend_truth_totals = {
        "truth_pairs": 0.0,
        "preblend_hits": 0.0,
        "final_hits": 0.0,
        "gained_hits": 0.0,
        "lost_hits": 0.0,
        "lost_anchor_hits": 0.0,
        "external_incremental_hits": 0.0,
        "external_union_hits": 0.0,
        "completion_gain_total": 0.0,
        "completion_gain_preblend": 0.0,
        "completion_gain_final": 0.0,
        "completion_gain_gained": 0.0,
        "completion_gain_lost": 0.0,
        "completion_gain_external_incremental": 0.0,
        "completion_gain_external_union": 0.0,
        "watch_time_gain_total": 0.0,
        "watch_time_gain_preblend": 0.0,
        "watch_time_gain_final": 0.0,
        "watch_time_gain_gained": 0.0,
        "watch_time_gain_lost": 0.0,
        "watch_time_gain_external_incremental": 0.0,
        "watch_time_gain_external_union": 0.0,
    }
    if evaluation_truth is not None:
        required_truth = {"profile_id", "content_id"}
        if not required_truth.issubset(evaluation_truth):
            raise ValueError(
                "blend audit truth requires profile_id and content_id"
            )
        truth_work = evaluation_truth.copy()
        truth_work[["profile_id", "content_id"]] = truth_work[
            ["profile_id", "content_id"]
        ].astype(str)
        for gain_column in ("completion_gain", "watch_time_gain"):
            truth_work[gain_column] = pd.to_numeric(
                truth_work.get(
                    gain_column,
                    pd.Series(0.0, index=truth_work.index),
                ),
                errors="coerce",
            ).fillna(0.0).clip(lower=0.0)
        truth_work = truth_work.drop_duplicates(
            ["profile_id", "content_id"], keep="last"
        )
        blend_truth_totals["truth_pairs"] = float(len(truth_work))
        for gain_column in ("completion_gain", "watch_time_gain"):
            blend_truth_totals[f"{gain_column}_total"] = float(
                truth_work[gain_column].sum()
            )
        truth_by_profile = {
            str(profile_id): {
                str(row.content_id): (
                    float(row.completion_gain),
                    float(row.watch_time_gain),
                )
                for row in group.itertuples(index=False)
            }
            for profile_id, group in truth_work.groupby(
                "profile_id", sort=False, observed=True
            )
        }
    index_preparation_seconds = index.index_preparation_seconds + (
        time.perf_counter() - index_started
    )
    profile_loop_started = time.perf_counter()
    for profile_id in profiles["profile_id"].astype(str).drop_duplicates():
        events = rule_events.get(profile_id, {})
        scored_rules = [
            (
                rule_id,
                float(base_rule_score.get(rule_id, 0.0))
                + float(np.log1p(events.get(rule_id, 0.0))),
            )
            for rule_id in eligible_rules
        ]
        scored_rules.sort(key=lambda pair: (-pair[1], pair[0]))
        selected_rules = scored_rules[: config.candidate_rules]
        maximum_rule_score = max((score for _, score in selected_rules), default=1.0)
        consumed = consumed_map.get(profile_id, set())
        preferences = tag_preferences.get(profile_id, {})
        maximum_tag_count = max(preferences.values(), default=1)

        neighbors: set[str] = set()
        for content_id in history_sets.get(profile_id, [])[:20]:
            neighbors.update(content_profiles.get(content_id, [])[:200])
            if len(neighbors) >= 200:
                break
        co_watch: dict[str, int] = {}
        for neighbor in sorted(neighbors)[:200]:
            for content_id in history_sets.get(neighbor, []):
                if content_id not in consumed:
                    co_watch[content_id] = co_watch.get(content_id, 0) + 1
        maximum_cowatch = max(co_watch.values(), default=1)
        ranked_cowatch = sorted(
            co_watch, key=lambda content_id: (-co_watch[content_id], content_id)
        )
        selected_rule_ids = {rule_id for rule_id, _ in selected_rules}
        external_rows: list[tuple[str, float]] = []
        external_rank_by_content: dict[str, int] = {}
        for content_id, score in external_rankings.get(profile_id, []):
            normalized_content_id = str(content_id)
            if normalized_content_id in external_rank_by_content:
                continue
            try:
                finite_score = float(score)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(finite_score):
                continue
            if normalized_content_id not in rules_by_content:
                continue
            external_rank_by_content[normalized_content_id] = len(external_rows) + 1
            external_rows.append((normalized_content_id, finite_score))
        cowatch_cap = (
            config.cowatch_candidates_per_rule
            or config.source_candidates_per_rule
        )
        cowatch_by_rule: dict[str, list[str]] = {
            rule_id: [] for rule_id in selected_rule_ids
        }
        for content_id in ranked_cowatch:
            for rule_id in sorted(
                rules_by_content.get(content_id, set()) & selected_rule_ids
            ):
                values = cowatch_by_rule[rule_id]
                if len(values) < cowatch_cap:
                    values.append(content_id)
            if cowatch_by_rule and all(
                len(values) >= cowatch_cap
                for values in cowatch_by_rule.values()
            ):
                break

        candidate_rows: dict[str, dict[str, Any]] = {}
        reserved: set[str] = set()
        top_preference_tags = [
            tag
            for tag, _ in sorted(
                preferences.items(), key=lambda pair: (-pair[1], pair[0])
            )[:5]
        ]
        personalized_score_cache: dict[str, float] = {}
        for rule_id, raw_rule_score in selected_rules:
            rule_candidates: dict[str, dict[str, Any]] = {}
            normalized_rule_score = (
                raw_rule_score / maximum_rule_score if maximum_rule_score > 0 else 0.0
            )
            tag_ids: list[str] = []
            for tag in top_preference_tags:
                tag_ids.extend(
                    tag_rule_rankings.get((rule_id, tag), [])[
                        : config.tag_candidates_per_rule
                    ]
                )
            cowatch_ids = cowatch_by_rule.get(rule_id, [])
            sources: list[tuple[str, list[str]]] = [
                (
                    "per_rule_popularity",
                    rule_rankings[rule_id][: config.source_candidates_per_rule],
                ),
                (
                    "profile_tag_affinity",
                    list(dict.fromkeys(tag_ids))[
                        : config.source_candidates_per_rule
                    ],
                ),
                ("co_watch", cowatch_ids),
            ]
            if config.recency_candidates_per_rule > 0:
                sources.append(
                    (
                        "catalog_recency",
                        recency_rankings.get(rule_id, [])[ :
                            config.recency_candidates_per_rule
                        ],
                    )
                )
            sources.append(
                (
                    "deep_rule_popularity",
                    rule_rankings[rule_id][
                        : config.two_tower_candidates_per_rule
                    ],
                )
            )
            sources_by_content: dict[str, list[str]] = {}
            for source, source_ids in sources:
                for content_id in source_ids:
                    content_sources = sources_by_content.setdefault(content_id, [])
                    if source not in content_sources:
                        content_sources.append(source)
            for content_id, content_sources in sources_by_content.items():
                if config.exclude_consumed and content_id in consumed:
                    continue
                personalized_score = personalized_score_cache.get(content_id)
                if personalized_score is None:
                    tag_score = sum(
                        preferences.get(tag, 0)
                        for tag in content_tags.get(content_id, set())
                    ) / maximum_tag_count
                    cowatch_score = co_watch.get(content_id, 0) / maximum_cowatch
                    personalized_score = (
                        float(pop_score_by_content.get(content_id, 0.0))
                        + tag_score
                        + cowatch_score
                    )
                    personalized_score_cache[content_id] = personalized_score
                rule_candidates[content_id] = {
                    "profile_id": profile_id,
                    "content_id": content_id,
                    "retrieval_score": personalized_score
                    + normalized_rule_score,
                    "retrieval_source": content_sources[0],
                    "retrieval_sources": "|".join(content_sources),
                    "candidate_rule_id": rule_id,
                    "rule_score": normalized_rule_score,
                }
            ranked_rule = sorted(
                rule_candidates.values(),
                key=lambda row: (-row["retrieval_score"], row["content_id"]),
            )
            rule_reserved: list[str] = []
            for content_id in recency_rankings.get(rule_id, [
            ])[: config.recency_reserved_per_rule]:
                if content_id in rule_candidates and content_id not in rule_reserved:
                    rule_reserved.append(content_id)
            for row in ranked_rule:
                if len(rule_reserved) >= config.reserved_per_rule:
                    break
                if row["content_id"] not in rule_reserved:
                    rule_reserved.append(row["content_id"])
            reserved.update(rule_reserved)
            for row in ranked_rule:
                previous = candidate_rows.get(row["content_id"])
                if previous is None or row["retrieval_score"] > previous["retrieval_score"]:
                    candidate_rows[row["content_id"]] = row

        # Fill unused budget with a deterministic global tower. This makes K the
        # actual configured budget while reserved rule slots stay first.
        for content_id in global_ranking:
            if len(candidate_rows) >= config.candidate_budget:
                break
            if content_id in candidate_rows:
                continue
            if config.exclude_consumed and content_id in consumed:
                continue
            content_rules = rules_by_content.get(content_id, set())
            selected_match = next(
                (
                    (rule_id, score)
                    for rule_id, score in selected_rules
                    if rule_id in content_rules
                ),
                None,
            )
            if selected_match is None:
                # Pure global fill: keep the content under its own real
                # membership with zero rule affinity instead of fabricating
                # a claim on the profile's top rule.
                rule_id = min(content_rules) if content_rules else ""
                raw_rule_score = 0.0
            else:
                rule_id, raw_rule_score = selected_match
            normalized_rule_score = (
                raw_rule_score / maximum_rule_score if maximum_rule_score > 0 else 0.0
            )
            candidate_rows[content_id] = {
                "profile_id": profile_id,
                "content_id": content_id,
                "retrieval_score": float(
                    pop_score_by_content.get(content_id, 0.0)
                )
                + normalized_rule_score,
                "retrieval_source": "global_popularity_fill",
                "retrieval_sources": "global_popularity_fill",
                "candidate_rule_id": rule_id,
                "rule_score": normalized_rule_score,
            }

        ranked_heuristic = sorted(
            candidate_rows.values(),
            key=lambda row: (
                row["content_id"] not in reserved,
                -row["retrieval_score"],
                row["content_id"],
            ),
        )
        exact_ids = exact_cowatch_rankings.get(profile_id, [])
        if exact_ids:
            safety_rows = [
                row for row in ranked_heuristic if row["content_id"] in reserved
            ]
            safety_slot_counts.append(len(safety_rows))
            incumbent_rows = list(safety_rows)
            incumbent_ids = {row["content_id"] for row in incumbent_rows}
            for exact_rank, content_id in enumerate(exact_ids, start=1):
                if len(incumbent_rows) >= config.candidate_budget:
                    break
                if content_id in incumbent_ids:
                    continue
                content_rules = rules_by_content.get(content_id, set())
                selected_match = next(
                    (
                        (rule_id, score)
                        for rule_id, score in selected_rules
                        if rule_id in content_rules
                    ),
                    None,
                )
                if selected_match is None:
                    rule_id = min(content_rules) if content_rules else ""
                    raw_rule_score = 0.0
                else:
                    rule_id, raw_rule_score = selected_match
                normalized_rule_score = (
                    raw_rule_score / maximum_rule_score
                    if maximum_rule_score > 0
                    else 0.0
                )
                incumbent_rows.append(
                    {
                        "profile_id": profile_id,
                        "content_id": content_id,
                        "retrieval_score": (
                            1.0 - exact_rank / max(len(exact_ids), 1)
                        )
                        + normalized_rule_score,
                        "retrieval_source": "exact_cosine_cowatch",
                        "retrieval_sources": "exact_cosine_cowatch",
                        "candidate_rule_id": rule_id,
                        "rule_score": normalized_rule_score,
                    }
                )
                incumbent_ids.add(content_id)
            for row in ranked_heuristic:
                if len(incumbent_rows) >= config.candidate_budget:
                    break
                if row["content_id"] not in incumbent_ids:
                    incumbent_rows.append(row)
                    incumbent_ids.add(row["content_id"])
            incumbent_candidates = incumbent_rows[: config.candidate_budget]
        else:
            safety_slot_counts.append(
                sum(row["content_id"] in reserved for row in ranked_heuristic)
            )
            incumbent_candidates = ranked_heuristic[: config.candidate_budget]

        incumbent_ids = {
            row["content_id"] for row in incumbent_candidates
        }
        incumbent_rank_by_content = {
            row["content_id"]: rank
            for rank, row in enumerate(incumbent_candidates, start=1)
        }
        anchor_ids = incumbent_ids & reserved
        for incumbent_rank, row in enumerate(incumbent_candidates, start=1):
            row["incumbent_rank"] = incumbent_rank
            row["is_incumbent_candidate"] = True
            row["is_safety_anchor"] = row["content_id"] in anchor_ids
            external_rank = external_rank_by_content.get(row["content_id"])
            row["external_rank"] = external_rank
            row["is_external_candidate"] = external_rank is not None
            if external_rank is not None:
                sources = row.get("retrieval_sources", row["retrieval_source"]).split("|")
                if external_source not in sources:
                    sources.append(external_source)
                row["retrieval_sources"] = "|".join(sources)
            row["external_fusion_score"] = (
                1.0 / (config.external_rrf_constant + incumbent_rank)
                + (
                    config.external_rrf_weight
                    / (config.external_rrf_constant + external_rank)
                    if external_rank is not None
                    else 0.0
                )
            )

        replacement_allowed = (
            not config.external_replacement_requires_history or bool(consumed)
        )
        profile_witness_rejections = 0
        if external_rows and (
            config.external_candidate_mode == "rerank" or not replacement_allowed
        ):
            if (
                config.external_candidate_mode == "replace"
                and not replacement_allowed
            ):
                external_replacement_blocked_profiles += 1
            ranked_candidates = sorted(
                incumbent_candidates,
                key=lambda row: (
                    -row["external_fusion_score"],
                    row["incumbent_rank"],
                    row["content_id"],
                ),
            )
            external_promoted = 0
            external_novel = 0
        elif external_rows and config.external_candidate_slots > 0:
            protected_count = max(
                len(anchor_ids),
                config.candidate_budget - config.external_candidate_slots,
            )
            protected_count = min(protected_count, len(incumbent_candidates))
            ranked_candidates = list(incumbent_candidates[:protected_count])
            selected_ids = {row["content_id"] for row in ranked_candidates}
            external_promoted = 0
            external_novel = 0
            for content_id, _ in external_rows:
                if len(ranked_candidates) >= config.candidate_budget:
                    break
                if content_id in selected_ids:
                    continue
                if config.exclude_consumed and content_id in consumed:
                    continue
                content_rules = rules_by_content.get(content_id, set())
                selected_match = next(
                    (
                        (rule_id, score)
                        for rule_id, score in selected_rules
                        if rule_id in content_rules
                    ),
                    None,
                )
                if selected_match is None:
                    continue
                rule_id, raw_rule_score = selected_match
                normalized_rule_score = (
                    raw_rule_score / maximum_rule_score
                    if maximum_rule_score > 0
                    else 0.0
                )
                external_rank = external_rank_by_content[content_id]
                previous = candidate_rows.get(content_id)
                row = dict(previous) if previous is not None else {
                    "profile_id": profile_id,
                    "content_id": content_id,
                    "retrieval_score": normalized_rule_score,
                    "candidate_rule_id": rule_id,
                    "rule_score": normalized_rule_score,
                }
                previous_sources = row.get("retrieval_sources", "")
                sources = [value for value in previous_sources.split("|") if value]
                if external_source not in sources:
                    sources.append(external_source)
                row.update(
                    {
                        "retrieval_source": external_source,
                        "retrieval_sources": "|".join(sources),
                        "incumbent_rank": incumbent_rank_by_content.get(content_id),
                        "external_rank": external_rank,
                        "is_external_candidate": True,
                        "is_incumbent_candidate": content_id in incumbent_ids,
                        "is_safety_anchor": content_id in anchor_ids,
                        "external_fusion_score": (
                            (
                                1.0
                                / (
                                    config.external_rrf_constant
                                    + incumbent_rank_by_content[content_id]
                                )
                                if content_id in incumbent_rank_by_content
                                else 0.0
                            )
                            + config.external_rrf_weight
                            / (config.external_rrf_constant + external_rank)
                        ),
                    }
                )
                ranked_candidates.append(row)
                selected_ids.add(content_id)
                external_promoted += 1
                external_novel += int(content_id not in incumbent_ids)
            for row in incumbent_candidates[protected_count:]:
                if len(ranked_candidates) >= config.candidate_budget:
                    break
                if row["content_id"] not in selected_ids:
                    ranked_candidates.append(row)
                    selected_ids.add(row["content_id"])
            ranked_candidates = sorted(
                ranked_candidates,
                key=lambda row: (
                    -row["external_fusion_score"],
                    row["incumbent_rank"]
                    if row["incumbent_rank"] is not None
                    else config.candidate_budget + int(row["external_rank"] or 0),
                    row["content_id"],
                ),
            )
            if config.external_replacement_requires_witness:
                rejected_rows = [
                    row
                    for row in ranked_candidates
                    if not row["is_incumbent_candidate"]
                    and not EXTERNAL_REPLACEMENT_WITNESS_SOURCES.intersection(
                        row.get("retrieval_sources", "").split("|")
                    )
                ]
                if rejected_rows:
                    rejected_ids = {
                        row["content_id"] for row in rejected_rows
                    }
                    ranked_candidates = [
                        row
                        for row in ranked_candidates
                        if row["content_id"] not in rejected_ids
                    ]
                    selected_ids = {
                        row["content_id"] for row in ranked_candidates
                    }
                    for row in incumbent_candidates:
                        if len(ranked_candidates) >= config.candidate_budget:
                            break
                        if row["content_id"] not in selected_ids:
                            ranked_candidates.append(row)
                            selected_ids.add(row["content_id"])
                    ranked_candidates = sorted(
                        ranked_candidates,
                        key=lambda row: (
                            -row["external_fusion_score"],
                            row["incumbent_rank"]
                            if row["incumbent_rank"] is not None
                            else config.candidate_budget
                            + int(row["external_rank"] or 0),
                            row["content_id"],
                        ),
                    )
                    profile_witness_rejections = len(rejected_rows)
                    external_witness_rejected_candidates += len(
                        rejected_rows
                    )
                    external_promoted -= len(rejected_rows)
                    external_novel -= len(rejected_rows)
            external_witness_rejected_profiles += int(
                profile_witness_rejections > 0
            )
        else:
            ranked_candidates = incumbent_candidates
            external_promoted = 0
            external_novel = 0

        final_ids = {row["content_id"] for row in ranked_candidates}
        lost_ids = incumbent_ids - final_ids
        gained_ids = final_ids - incumbent_ids
        profile_truth = truth_by_profile.get(profile_id, {})
        if profile_truth:
            truth_ids = set(profile_truth)
            external_ids = set(external_rank_by_content)
            preblend_hits = truth_ids & incumbent_ids
            final_hits = truth_ids & final_ids
            gained_hits = truth_ids & gained_ids
            lost_hits = truth_ids & lost_ids
            external_incremental_hits = (
                truth_ids & external_ids
            ) - incumbent_ids
            external_union_hits = truth_ids & (external_ids | incumbent_ids)
            blend_truth_totals["preblend_hits"] += len(preblend_hits)
            blend_truth_totals["final_hits"] += len(final_hits)
            blend_truth_totals["gained_hits"] += len(gained_hits)
            blend_truth_totals["lost_hits"] += len(lost_hits)
            blend_truth_totals["lost_anchor_hits"] += len(
                lost_hits & anchor_ids
            )
            blend_truth_totals["external_incremental_hits"] += len(
                external_incremental_hits
            )
            blend_truth_totals["external_union_hits"] += len(
                external_union_hits
            )
            for content_id, gains in profile_truth.items():
                for gain_column, gain in zip(
                    ("completion_gain", "watch_time_gain"), gains
                ):
                    if content_id in preblend_hits:
                        blend_truth_totals[f"{gain_column}_preblend"] += gain
                    if content_id in final_hits:
                        blend_truth_totals[f"{gain_column}_final"] += gain
                    if content_id in gained_hits:
                        blend_truth_totals[f"{gain_column}_gained"] += gain
                    if content_id in lost_hits:
                        blend_truth_totals[f"{gain_column}_lost"] += gain
                    if content_id in external_incremental_hits:
                        blend_truth_totals[
                            f"{gain_column}_external_incremental"
                        ] += gain
                    if content_id in external_union_hits:
                        blend_truth_totals[
                            f"{gain_column}_external_union"
                        ] += gain
        if capture_blend_audit:
            lost_candidates = [
                row
                for row in incumbent_candidates
                if row["content_id"] in lost_ids
            ]
            for row in lost_candidates:
                lost_incumbent_rows.append(dict(row))
            for name, content_ids in (
                ("preblend_incumbent", incumbent_ids),
                ("blend_gained", gained_ids),
                ("blend_lost", lost_ids),
                ("blend_union", final_ids | lost_ids),
            ):
                blend_pair_digests[name].append(
                    (profile_id, _content_set_digest(content_ids))
                )
            blend_profile_rows.append(
                {
                    "profile_id": profile_id,
                    "has_consumed_history": bool(consumed),
                    "replacement_allowed": bool(replacement_allowed),
                    "replacement_witness_required": bool(
                        config.external_replacement_requires_witness
                    ),
                    "witness_rejected_count": int(profile_witness_rejections),
                    "replacement_blocked_no_history": bool(
                        external_rows
                        and config.external_candidate_mode == "replace"
                        and not replacement_allowed
                    ),
                    "incumbent_count": len(incumbent_ids),
                    "retained_count": len(incumbent_ids & final_ids),
                    "gained_count": len(final_ids - incumbent_ids),
                    "lost_count": len(lost_candidates),
                    "lost_safety_anchor_count": sum(
                        bool(row.get("is_safety_anchor", False))
                        for row in lost_candidates
                    ),
                }
            )
        anchor_retention_rates.append(
            len(anchor_ids & final_ids) / max(len(anchor_ids), 1)
        )
        incumbent_retention_rates.append(
            len(incumbent_ids & final_ids) / max(len(incumbent_ids), 1)
        )
        external_overlap_counts.append(
            len(final_ids & set(external_rank_by_content))
        )
        external_promoted_counts.append(external_promoted)
        external_novel_counts.append(external_novel)
        for candidate_rank, row in enumerate(ranked_candidates, start=1):
            row["candidate_rank"] = candidate_rank
            rows.append(row)

    profile_loop_seconds = time.perf_counter() - profile_loop_started
    materialization_started = time.perf_counter()
    output = pd.DataFrame(
        rows,
        columns=[
            "profile_id",
            "content_id",
            "retrieval_score",
            "retrieval_source",
            "retrieval_sources",
            "candidate_rule_id",
            "rule_score",
            "incumbent_rank",
            "external_rank",
            "external_fusion_score",
            "is_incumbent_candidate",
            "is_external_candidate",
            "is_safety_anchor",
            "candidate_rank",
        ],
    )
    counts_by_profile = (
        output.groupby("profile_id")["content_id"].nunique()
        if not output.empty
        else pd.Series(dtype=float)
    )
    candidate_pairs_hash = pair_keys_hash(output)
    consumed_candidate_overlap = (
        int(
            sum(
                len(
                    set(group["content_id"].astype(str))
                    & consumed_map.get(str(profile_id), set())
                )
                for profile_id, group in output.groupby("profile_id")
            )
        )
        if not output.empty
        else 0
    )
    materialization_seconds = time.perf_counter() - materialization_started
    retrieval_build_seconds = time.perf_counter() - retrieval_started
    metrics = {
        "retrieval_profiles": int(profiles["profile_id"].nunique()),
        "candidate_budget": config.candidate_budget,
        "candidate_rules": config.candidate_rules,
        "reserved_per_rule": config.reserved_per_rule,
        "retrieval_strategy": (
            "rule_safety_exact_cosine_cowatch"
            if config.exact_cowatch_enabled
            else "rule_safety_heuristic"
        ),
        "learned_retriever": bool(external_rankings),
        "exact_cowatch_enabled": config.exact_cowatch_enabled,
        "exact_cowatch_profiles": int(exact_cowatch_profiles),
        "exact_cowatch_candidates": config.exact_cowatch_candidates,
        "external_retrieval_enabled": bool(external_rankings),
        "external_retrieval_source": external_source,
        "external_retrieval_profiles": int(external_profiles),
        "external_candidate_mode": config.external_candidate_mode,
        "external_candidate_slots": config.external_candidate_slots,
        "external_rrf_weight": config.external_rrf_weight,
        "external_rrf_constant": config.external_rrf_constant,
        "external_replacement_requires_history": (
            config.external_replacement_requires_history
        ),
        "external_replacement_requires_witness": (
            config.external_replacement_requires_witness
        ),
        "external_replacement_witness_sources": sorted(
            EXTERNAL_REPLACEMENT_WITNESS_SOURCES
        ),
        "recency_candidates_per_rule": config.recency_candidates_per_rule,
        "recency_reserved_per_rule": config.recency_reserved_per_rule,
        "recency_max_age_days": config.recency_max_age_days,
        "catalog_recency_source_enabled": bool(recency_rankings),
        "external_replacement_blocked_profiles": int(
            external_replacement_blocked_profiles
        ),
        "external_witness_rejected_candidates": int(
            external_witness_rejected_candidates
        ),
        "external_witness_rejected_profiles": int(
            external_witness_rejected_profiles
        ),
        "mean_external_overlap_candidates": float(np.mean(external_overlap_counts))
        if external_overlap_counts
        else 0.0,
        "mean_external_promoted_candidates": float(np.mean(external_promoted_counts))
        if external_promoted_counts
        else 0.0,
        "mean_external_novel_candidates": float(np.mean(external_novel_counts))
        if external_novel_counts
        else 0.0,
        "mean_incumbent_candidate_retention_rate": float(
            np.mean(incumbent_retention_rates)
        )
        if incumbent_retention_rates
        else 1.0,
        "minimum_incumbent_candidate_retention_rate": float(
            min(incumbent_retention_rates)
        )
        if incumbent_retention_rates
        else 1.0,
        "mean_safety_anchor_retention_rate": float(np.mean(anchor_retention_rates))
        if anchor_retention_rates
        else 1.0,
        "minimum_safety_anchor_retention_rate": float(min(anchor_retention_rates))
        if anchor_retention_rates
        else 1.0,
        "hybrid_two_tower_enabled": bool(external_rankings)
        and "hybrid" in external_source,
        "hybrid_two_tower_profiles": int(external_profiles)
        if "hybrid" in external_source
        else 0,
        "sasrec_lite_enabled": bool(external_rankings)
        and "sasrec" in external_source,
        "sasrec_lite_profiles": int(external_profiles)
        if "sasrec" in external_source
        else 0,
        "mean_rule_safety_slots": float(np.mean(safety_slot_counts))
        if safety_slot_counts
        else 0.0,
        "rule_popularity_candidates_per_rule": config.two_tower_candidates_per_rule,
        "exclude_consumed": config.exclude_consumed,
        "consumed_candidate_overlap": consumed_candidate_overlap,
        "candidate_pairs": int(len(output)),
        "candidate_pairs_hash": candidate_pairs_hash,
        "mean_candidates_per_profile": float(counts_by_profile.mean())
        if not counts_by_profile.empty
        else 0.0,
        "max_candidates_per_profile": int(counts_by_profile.max())
        if not counts_by_profile.empty
        else 0,
        "candidate_recall_metric_name": f"candidate_recall@{config.candidate_budget}",
        "retrieval_build_time_seconds": retrieval_build_seconds,
        "retrieval_stage_timings": {
            "catalog_preparation_seconds": catalog_preparation_seconds,
            "history_preparation_seconds": history_preparation_seconds,
            "index_preparation_seconds": index_preparation_seconds,
            "profile_loop_seconds": profile_loop_seconds,
            "materialization_seconds": materialization_seconds,
        },
    }
    if evaluation_truth is not None:
        truth_pairs = int(blend_truth_totals["truth_pairs"])
        preblend_hits = int(blend_truth_totals["preblend_hits"])
        final_hits = int(blend_truth_totals["final_hits"])
        gained_hits = int(blend_truth_totals["gained_hits"])
        lost_hits = int(blend_truth_totals["lost_hits"])
        external_union_hits = int(
            blend_truth_totals["external_union_hits"]
        )
        metrics.update(
            {
                "blend_truth_audit_reference": "preblend_incumbent",
                "blend_truth_pairs": truth_pairs,
                "preblend_incumbent_truth_hits": preblend_hits,
                "preblend_incumbent_candidate_recall": preblend_hits
                / max(truth_pairs, 1),
                "final_blend_truth_hits": final_hits,
                "final_blend_candidate_recall": final_hits
                / max(truth_pairs, 1),
                "blend_gained_truth_hits": gained_hits,
                "blend_lost_truth_hits": lost_hits,
                "blend_net_truth_hits": gained_hits - lost_hits,
                "blend_lost_safety_anchor_truth_hits": int(
                    blend_truth_totals["lost_anchor_hits"]
                ),
                "external_incremental_reference": "preblend_incumbent",
                "external_incremental_truth_hits": int(
                    blend_truth_totals["external_incremental_hits"]
                ),
                "external_oracle_union_truth_hits": external_union_hits,
                f"external_oracle_union_recall@{config.candidate_budget}+incumbent": (
                    external_union_hits / max(truth_pairs, 1)
                ),
            }
        )
        for gain_column in ("completion_gain", "watch_time_gain"):
            total = float(blend_truth_totals[f"{gain_column}_total"])
            preblend = float(
                blend_truth_totals[f"{gain_column}_preblend"]
            )
            final = float(blend_truth_totals[f"{gain_column}_final"])
            gained = float(blend_truth_totals[f"{gain_column}_gained"])
            lost = float(blend_truth_totals[f"{gain_column}_lost"])
            metrics.update(
                {
                    f"preblend_incumbent_{gain_column}_recall": preblend
                    / max(total, 1e-12),
                    f"final_blend_{gain_column}_recall": final
                    / max(total, 1e-12),
                    f"blend_gained_{gain_column}": gained,
                    f"blend_lost_{gain_column}": lost,
                    f"blend_net_{gain_column}": gained - lost,
                    f"external_incremental_{gain_column}": float(
                        blend_truth_totals[
                            f"{gain_column}_external_incremental"
                        ]
                    ),
                    f"external_oracle_union_{gain_column}_recall": float(
                        blend_truth_totals[
                            f"{gain_column}_external_union"
                        ]
                    )
                    / max(total, 1e-12),
                }
            )
    if not capture_blend_audit:
        return output, metrics

    lost_columns = [
        "profile_id",
        "content_id",
        "retrieval_score",
        "retrieval_source",
        "retrieval_sources",
        "candidate_rule_id",
        "rule_score",
        "incumbent_rank",
        "external_rank",
        "external_fusion_score",
        "is_incumbent_candidate",
        "is_external_candidate",
        "is_safety_anchor",
    ]
    profile_columns = [
        "profile_id",
        "has_consumed_history",
        "replacement_allowed",
        "replacement_witness_required",
        "witness_rejected_count",
        "replacement_blocked_no_history",
        "incumbent_count",
        "retained_count",
        "gained_count",
        "lost_count",
        "lost_safety_anchor_count",
    ]
    audit = CandidateBlendAudit(
        lost_incumbents=pd.DataFrame(
            lost_incumbent_rows, columns=lost_columns
        ),
        profiles=pd.DataFrame(blend_profile_rows, columns=profile_columns),
    )
    metrics.update(
        {
            "blend_audit_captured": True,
            "blend_lost_pairs": int(len(audit.lost_incumbents)),
            "blend_profiles": int(len(audit.profiles)),
            "blend_profiles_with_gains": int(
                audit.profiles["gained_count"].gt(0).sum()
            ),
            "blend_profiles_with_losses": int(
                audit.profiles["lost_count"].gt(0).sum()
            ),
            "blend_lost_safety_anchor_pairs": int(
                audit.profiles["lost_safety_anchor_count"].sum()
            ),
            "blend_pair_set_hash_semantics": (
                "sha256_of_sorted_profile_ids_and_per_profile_sorted_content_set_digests"
            ),
            **{
                f"{name}_pair_set_hash": _profile_pair_set_hash(values)
                for name, values in blend_pair_digests.items()
            },
        }
    )
    return output, metrics, audit


from recsys_vnpt.retrieval_metrics import (  # re-exported for existing callers
    candidate_gain_recall_by_profile,
    candidate_gain_recall_curve,
    candidate_recall_by_profile,
    candidate_recall_curve,
    positive_hit_source_counts,
    ranking_map_incremental_metrics,
    ranking_map_recall_metrics,
    retrieval_truth_attribution,
)
