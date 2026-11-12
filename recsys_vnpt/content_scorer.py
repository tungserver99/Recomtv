"""Learned ordering of content within a rule.

Two gradient-boosted heads (scikit-learn or XGBoost, see ``gbdt``) on the graded gains, fitted on
strictly pre-cutoff windows and RRF-fused with the incumbent hand-summed score
(``popularity + tag + co-watch``) rather than replacing it.

Negatives are drawn from the popular head of the positive's own rule, because
that is the distribution retrieval actually surfaces at serving time. Matching
the positive's popularity band instead produces a harder training set that
ranks worse: the lever is distribution match, not negative hardness.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import sparse

from recsys_vnpt.gbdt import head_device, make_head
from recsys_vnpt.orderings import reciprocal_rank, rrf_fuse
from recsys_vnpt.temporal import load_temporal_windows


GAIN_COLUMNS = ("completion_gain", "watch_time_gain")

CATEGORICAL_FEATURES = (
    "type_index",
    "family_index",
    "category_index",
    "country_index",
)

FEATURE_COLUMNS = (
    "pop_all_log",
    "pop_30d_log",
    "content_duration_log",
    "content_publish_year",
    "total_episode_log",
    "content_age_days",
    *CATEGORICAL_FEATURES,
    "tag_affinity",
    "category_affinity",
    "family_affinity",
    "profile_events_log",
    "profile_unique_contents_log",
)

# HistGradientBoosting rejects a categorical with more than 255 levels, and
# content_category_name alone has 905. Rare levels collapse into one bucket.
MAX_CATEGORY_LEVELS = 250

# Two adjacent bands either side is the widest a "popularity-matched" negative
# may drift before it stops being matched at all.
BAND_WIDENING_STEPS = 2


@dataclass(frozen=True)
class ContentScope:
    """Servable catalog at one cutoff, plus the sparse tag side tables."""

    contents: pd.DataFrame
    index: dict[str, int]
    contents_by_rule: dict[str, np.ndarray]
    rules_by_content: dict[str, tuple[str, ...]]
    tag_matrix: sparse.csr_matrix
    tag_idf: np.ndarray
    tag_norm: np.ndarray

    @property
    def size(self) -> int:
        return len(self.contents)


def _folded_codes(
    values: pd.Series, levels: dict[str, int] | None = None
) -> tuple[np.ndarray, dict[str, int]]:
    """Integer codes for a string column, rare levels folded into one bucket.

    The level map is built once over the whole catalog and reused for every
    window, so a code means the same thing at training and at serving time.
    """
    text = values.astype("string").fillna("(null)").astype(str)
    if levels is None:
        top = text.value_counts().head(MAX_CATEGORY_LEVELS).index
        levels = {value: index for index, value in enumerate(sorted(top))}
    other = len(levels)
    codes = text.map(levels).fillna(other).to_numpy(dtype=np.float32)
    return codes, levels


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def content_scope_as_of(
    canonical: Path,
    cutoff: pd.Timestamp,
    membership: pd.DataFrame,
    levels: dict[str, dict[str, int]] | None = None,
) -> tuple[ContentScope, dict[str, dict[str, int]], dict[str, Any]]:
    """Build the servable catalog and its static attributes at ``cutoff``.

    Contents outside every rule are dropped: 31% of the catalog belongs to no
    rule, so the system cannot serve them and they only dilute the pools.
    """
    columns = [
        "content_id",
        "content_duration",
        "content_publish_year",
        "total_episode",
        "type_id",
        "content_family",
        "content_category_name",
        "content_country",
        "createdate",
        "publish_date",
    ]
    catalog = pd.read_parquet(canonical / "contents.parquet", columns=columns)
    catalog["content_id"] = catalog["content_id"].astype(str)

    published = pd.to_datetime(catalog["publish_date"], errors="coerce")
    created = pd.to_datetime(catalog["createdate"], errors="coerce")
    # A content the catalog says appeared after the cutoff cannot be a
    # candidate at that cutoff; an unknown date keeps the content rather than
    # silently shrinking the pool.
    available = ~(published.gt(cutoff).fillna(False) & created.gt(cutoff).fillna(False))

    servable = set(membership["content_id"].astype(str))
    keep = available & catalog["content_id"].isin(servable)
    scoped = catalog[keep].reset_index(drop=True)
    if scoped.empty:
        raise ValueError(f"no servable content remains at cutoff {cutoff}")

    scoped_created = pd.to_datetime(scoped["createdate"], errors="coerce")
    frame = pd.DataFrame({"content_id": scoped["content_id"]})
    frame["content_duration_log"] = np.log1p(
        _numeric(scoped["content_duration"]).clip(lower=0).fillna(0.0)
    ).astype(np.float32)
    frame["content_publish_year"] = (
        _numeric(scoped["content_publish_year"]).astype(np.float32)
    )
    frame["total_episode_log"] = np.log1p(
        _numeric(scoped["total_episode"]).clip(lower=0).fillna(0.0)
    ).astype(np.float32)
    frame["content_age_days"] = (
        (cutoff - scoped_created).dt.total_seconds() / 86400.0
    ).astype(np.float32)

    level_maps = dict(levels or {})
    for name, column in (
        ("type_index", "type_id"),
        ("family_index", "content_family"),
        ("category_index", "content_category_name"),
        ("country_index", "content_country"),
    ):
        codes, level_maps[name] = _folded_codes(
            scoped[column], level_maps.get(name)
        )
        frame[name] = codes

    index = {value: position for position, value in enumerate(frame["content_id"])}

    normalised = membership[membership["content_id"].astype(str).isin(index)]
    contents_by_rule = {
        str(rule_id): np.asarray(
            sorted(set(group["content_id"].astype(str))), dtype=object
        )
        for rule_id, group in normalised.groupby("rule_id")
    }
    rules_by_content = {
        str(content_id): tuple(sorted(set(group["rule_id"].astype(str))))
        for content_id, group in normalised.groupby("content_id")
    }

    tag_matrix, tag_idf, tag_norm = _tag_tables(canonical, index)
    metrics = {
        "content_scope_contents": int(len(frame)),
        "content_scope_rules": int(len(contents_by_rule)),
        "content_scope_dropped_unservable": int(len(catalog) - len(frame)),
    }
    scope = ContentScope(
        contents=frame,
        index=index,
        contents_by_rule=contents_by_rule,
        rules_by_content=rules_by_content,
        tag_matrix=tag_matrix,
        tag_idf=tag_idf,
        tag_norm=tag_norm,
    )
    return scope, level_maps, metrics


def _tag_tables(
    canonical: Path, index: dict[str, int]
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Content-by-tag matrix with idf weights and per-content L2 norms."""
    path = canonical / "content_tags.parquet"
    size = len(index)
    if not path.exists():
        empty = sparse.csr_matrix((size, 0), dtype=np.float32)
        return empty, np.zeros(0, dtype=np.float32), np.zeros(size, dtype=np.float32)
    tags = pd.read_parquet(path, columns=["content_id", "tag_id"])
    tags["content_id"] = tags["content_id"].astype(str)
    tags = tags[tags["content_id"].isin(index)]
    if tags.empty:
        empty = sparse.csr_matrix((size, 0), dtype=np.float32)
        return empty, np.zeros(0, dtype=np.float32), np.zeros(size, dtype=np.float32)
    tag_ids = {
        value: position
        for position, value in enumerate(sorted(tags["tag_id"].astype(str).unique()))
    }
    matrix = sparse.csr_matrix(
        (
            np.ones(len(tags), dtype=np.float32),
            (
                tags["content_id"].map(index).to_numpy(),
                tags["tag_id"].astype(str).map(tag_ids).to_numpy(),
            ),
        ),
        shape=(size, len(tag_ids)),
    )
    document_frequency = np.asarray(matrix.sum(axis=0)).ravel()
    idf = np.log(1.0 + size / np.maximum(document_frequency, 1.0)).astype(np.float32)
    weighted = matrix.multiply(idf).tocsr()
    norm = np.sqrt(
        np.asarray(weighted.multiply(weighted).sum(axis=1)).ravel()
    ).astype(np.float32)
    return weighted, idf, norm


def popularity_at(
    interactions: pd.DataFrame, cutoff: pd.Timestamp, scope: ContentScope
) -> tuple[np.ndarray, np.ndarray]:
    """Event counts per content over all history and the trailing 30 days."""
    history = interactions[
        pd.to_datetime(interactions["create_date"], errors="coerce").le(cutoff)
    ]
    positions = history["content_id"].astype(str).map(scope.index)
    valid = positions.notna()
    all_time = np.bincount(
        positions[valid].to_numpy(dtype=np.int64), minlength=scope.size
    ).astype(np.float32)
    recent_mask = valid & pd.to_datetime(
        history["create_date"], errors="coerce"
    ).gt(cutoff - pd.Timedelta(days=30))
    recent = np.bincount(
        positions[recent_mask].to_numpy(dtype=np.int64), minlength=scope.size
    ).astype(np.float32)
    return all_time, recent


def popularity_bands(popularity: np.ndarray, bands: int = 8) -> np.ndarray:
    """Band index per content, cut on log popularity *value* not on rank.

    Rank quartiles do not work here: 90% of the catalog sits at popularity
    zero, so three of four rank quartiles would hold nothing but unwatched
    content and every "matched" negative would come from that region.
    """
    scores = np.log1p(np.maximum(popularity, 0.0))
    watched = scores > 0
    output = np.zeros(len(scores), dtype=np.int16)
    if not watched.any():
        return output
    quantiles = np.linspace(0.0, 1.0, bands + 1)[1:-1]
    edges = np.unique(np.quantile(scores[watched], quantiles))
    output[watched] = np.digitize(scores[watched], edges) + 1
    return output


def sample_training_pairs(
    positives: pd.DataFrame,
    scope: ContentScope,
    bands: np.ndarray,
    consumed: dict[str, set[str]],
    ratio: int = 4,
    seed: int = 42,
    strategy: str = "rule_head",
    popularity: np.ndarray | None = None,
    head_size: int = 200,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Draw negatives that look like the pool the scorer will actually rank.

    ``rule_head`` restricts each draw to the most popular contents of the
    positive's own rule, which is what retrieval surfaces and therefore the
    closest cheap match to what the scorer sees at serving time.
    ``band_matched`` instead matches the positive's log-popularity band
    anywhere in the rule. It builds a harder training set but ranks worse,
    because the model never learns to reject the unwatched tail that dominates
    the real pool. Kept only so that comparison stays reproducible.

    Widening is counted rather than hidden: if most draws fall back to "any
    servable content", the intended matching is not happening at all.
    """
    if ratio < 1:
        raise ValueError("content scorer ratio must be at least 1")
    if strategy not in {"rule_head", "band_matched"}:
        raise ValueError(f"unsupported content scorer sampling strategy: {strategy}")
    if strategy == "rule_head" and popularity is None:
        raise ValueError("rule_head sampling requires content popularity")
    counters = {
        "same_rule_head": 0,
        "same_rule_same_band": 0,
        "same_rule_near_band": 0,
        "same_rule_any_band": 0,
        "any_servable": 0,
        "exhausted": 0,
    }
    head_by_rule: dict[str, np.ndarray] = {}
    if strategy == "rule_head":
        for rule_id, pool in scope.contents_by_rule.items():
            order = np.argsort(
                -np.asarray(
                    [popularity[scope.index[str(value)]] for value in pool],
                    dtype=np.float64,
                ),
                kind="stable",
            )
            head_by_rule[rule_id] = pool[order[:head_size]]
    band_by_content = {
        content_id: int(bands[position])
        for content_id, position in scope.index.items()
    }
    servable = np.asarray(list(scope.index), dtype=object)
    positive_by_profile = {
        str(profile): set(group["content_id"].astype(str))
        for profile, group in positives.groupby("profile_id")
    }

    rows: list[dict[str, object]] = []
    for raw_profile, group in positives.groupby("profile_id", sort=True):
        profile_id = str(raw_profile)
        blocked = set(positive_by_profile[profile_id]) | consumed.get(
            profile_id, set()
        )
        selected: set[str] = set()
        profile_seed = int(
            hashlib.sha256(f"{seed}|{profile_id}".encode("utf-8")).hexdigest()[:8],
            16,
        )
        rng = np.random.default_rng(profile_seed)

        def draw(pool: np.ndarray, wanted: int, bucket: str) -> int:
            if wanted <= 0 or len(pool) == 0:
                return 0
            taken = 0
            probe = rng.choice(
                len(pool), size=min(len(pool), max(32, wanted * 8)), replace=False
            )
            for position in probe:
                content_id = str(pool[int(position)])
                if content_id in blocked or content_id in selected:
                    continue
                selected.add(content_id)
                rows.append(
                    {
                        "profile_id": profile_id,
                        "content_id": content_id,
                        "sample_type": bucket,
                    }
                )
                counters[bucket] += 1
                taken += 1
                if taken >= wanted:
                    break
            return taken

        for positive_id in sorted(group["content_id"].astype(str)):
            rules = scope.rules_by_content.get(positive_id, ())
            if not rules:
                continue
            band = band_by_content.get(positive_id, 0)
            rule_id = rules[int(rng.integers(0, len(rules)))]
            pool = scope.contents_by_rule.get(rule_id, np.asarray([], dtype=object))
            pool_bands = (
                np.asarray(
                    [band_by_content.get(str(value), 0) for value in pool],
                    dtype=np.int16,
                )
                if strategy == "band_matched"
                else np.empty(0, dtype=np.int16)
            )
            remaining = ratio
            if strategy == "rule_head":
                remaining -= draw(
                    head_by_rule.get(rule_id, pool), remaining, "same_rule_head"
                )
            else:
                remaining -= draw(
                    pool[pool_bands == band], remaining, "same_rule_same_band"
                )
                if remaining > 0:
                    near = np.abs(pool_bands - band) <= BAND_WIDENING_STEPS
                    remaining -= draw(pool[near], remaining, "same_rule_near_band")
            if remaining > 0:
                remaining -= draw(pool, remaining, "same_rule_any_band")
            if remaining > 0:
                remaining -= draw(servable, remaining, "any_servable")
            counters["exhausted"] += max(remaining, 0)

    negatives = pd.DataFrame(rows, columns=["profile_id", "content_id", "sample_type"])
    for gain in GAIN_COLUMNS:
        negatives[gain] = np.float32(0.0)
    observed = positives[["profile_id", "content_id", *GAIN_COLUMNS]].copy()
    observed["profile_id"] = observed["profile_id"].astype(str)
    observed["content_id"] = observed["content_id"].astype(str)
    observed["sample_type"] = "observed_engagement"
    combined = pd.concat([observed, negatives], ignore_index=True, sort=False)
    return combined, counters


def _l2_normalise(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    matrix = matrix.tocsr()
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    inverse = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    return (sparse.diags(inverse.astype(np.float32)) @ matrix).tocsr()


def _bucket_affinity(
    watched: sparse.csr_matrix, codes: np.ndarray
) -> sparse.csr_matrix:
    """Share of each profile's history that falls in each categorical bucket."""
    buckets = int(codes.max()) + 1 if len(codes) else 1
    onehot = sparse.csr_matrix(
        (np.ones(len(codes), dtype=np.float32), (np.arange(len(codes)), codes)),
        shape=(len(codes), buckets),
    )
    counts = (watched @ onehot).tocsr()
    totals = np.asarray(counts.sum(axis=1)).ravel()
    inverse = np.divide(1.0, totals, out=np.zeros_like(totals), where=totals > 0)
    return (sparse.diags(inverse.astype(np.float32)) @ counts).tocsr()


def _gather(
    matrix: sparse.csr_matrix, rows: np.ndarray, columns: np.ndarray
) -> np.ndarray:
    if matrix.shape[1] == 0:
        return np.zeros(len(rows), dtype=np.float32)
    clipped = np.clip(columns, 0, matrix.shape[1] - 1)
    return np.asarray(matrix[rows, clipped]).ravel().astype(np.float32)


@dataclass(frozen=True)
class HistoryContext:
    """Pre-cutoff viewing history reduced to the per-profile side tables.

    Built once and reused across chunks: the sparse products below cost more
    than the gather that follows them, and the serving pool runs to tens of
    millions of pairs.
    """

    profile_index: dict[str, int]
    user_tags: sparse.csr_matrix
    family_affinity: sparse.csr_matrix
    category_affinity: sparse.csr_matrix
    family_codes: np.ndarray
    category_codes: np.ndarray
    events: np.ndarray
    unique_contents: np.ndarray


def build_history_context(
    profiles: pd.Series,
    interactions: pd.DataFrame,
    cutoff: pd.Timestamp,
    scope: ContentScope,
) -> HistoryContext:
    """Reduce pre-cutoff history to the tables the pair features gather from."""
    profile_index = {
        value: position
        for position, value in enumerate(
            sorted(profiles.astype(str).drop_duplicates())
        )
    }
    history = interactions[
        pd.to_datetime(interactions["create_date"], errors="coerce").le(cutoff)
    ]
    rows = history["profile_id"].astype(str).map(profile_index)
    columns = history["content_id"].astype(str).map(scope.index)
    keep = (rows.notna() & columns.notna()).to_numpy()
    watched = sparse.csr_matrix(
        (
            np.ones(int(keep.sum()), dtype=np.float32),
            (
                rows.to_numpy()[keep].astype(np.int64),
                columns.to_numpy()[keep].astype(np.int64),
            ),
        ),
        shape=(len(profile_index), scope.size),
    )
    family_codes = scope.contents["family_index"].to_numpy(dtype=np.int64)
    category_codes = scope.contents["category_index"].to_numpy(dtype=np.int64)
    return HistoryContext(
        profile_index=profile_index,
        user_tags=_l2_normalise(watched @ scope.tag_matrix),
        family_affinity=_bucket_affinity(watched, family_codes),
        category_affinity=_bucket_affinity(watched, category_codes),
        family_codes=family_codes,
        category_codes=category_codes,
        events=np.asarray(watched.sum(axis=1)).ravel().astype(np.float32),
        unique_contents=np.asarray((watched > 0).sum(axis=1))
        .ravel()
        .astype(np.float32),
    )


def gather_pair_features(
    pairs: pd.DataFrame,
    context: HistoryContext,
    scope: ContentScope,
    popularity_all: np.ndarray,
    popularity_recent: np.ndarray,
) -> pd.DataFrame:
    """Per-(profile, content) features, including the user-by-item crosses.

    Every column here varies across a profile's candidates. The production
    feature builder produces none that do, which is why a model fed by it
    cannot rank inside a profile at all.
    """
    pair_rows = (
        pairs["profile_id"].astype(str).map(context.profile_index).to_numpy(dtype=np.int64)
    )
    pair_columns = (
        pairs["content_id"].astype(str).map(scope.index).to_numpy(dtype=np.int64)
    )
    tag_affinity = (
        np.asarray(
            context.user_tags[pair_rows]
            .multiply(scope.tag_matrix[pair_columns])
            .sum(axis=1)
        )
        .ravel()
        .astype(np.float32)
    )
    tag_affinity /= np.maximum(scope.tag_norm[pair_columns], 1e-6)

    static = scope.contents.iloc[pair_columns]
    output = pd.DataFrame(index=pairs.index)
    output["pop_all_log"] = np.log1p(popularity_all[pair_columns])
    output["pop_30d_log"] = np.log1p(popularity_recent[pair_columns])
    for column in (
        "content_duration_log",
        "content_publish_year",
        "total_episode_log",
        "content_age_days",
        *CATEGORICAL_FEATURES,
    ):
        output[column] = static[column].to_numpy()
    output["tag_affinity"] = tag_affinity
    output["category_affinity"] = _gather(
        context.category_affinity, pair_rows, context.category_codes[pair_columns]
    )
    output["family_affinity"] = _gather(
        context.family_affinity, pair_rows, context.family_codes[pair_columns]
    )
    output["profile_events_log"] = np.log1p(context.events[pair_rows])
    output["profile_unique_contents_log"] = np.log1p(
        context.unique_contents[pair_rows]
    )
    return output[list(FEATURE_COLUMNS)].astype(np.float32)


def pair_features(
    pairs: pd.DataFrame,
    interactions: pd.DataFrame,
    cutoff: pd.Timestamp,
    scope: ContentScope,
    popularity_all: np.ndarray,
    popularity_recent: np.ndarray,
) -> pd.DataFrame:
    """Single-shot feature build, for training frames that fit in memory."""
    context = build_history_context(
        pairs["profile_id"], interactions, cutoff, scope
    )
    return gather_pair_features(
        pairs, context, scope, popularity_all, popularity_recent
    )


def _window_positives(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        path, columns=["profile_id", "content_id", "sample_type", *GAIN_COLUMNS]
    )
    observed = frame[frame["sample_type"].eq("observed_engagement")].copy()
    observed = observed.astype({"profile_id": str, "content_id": str})
    for gain in GAIN_COLUMNS:
        observed[gain] = (
            pd.to_numeric(observed[gain], errors="coerce").fillna(0.0).clip(lower=0.0)
        )
    return observed[["profile_id", "content_id", *GAIN_COLUMNS]]


def _consumed_before(
    interactions: pd.DataFrame, cutoff: pd.Timestamp
) -> dict[str, set[str]]:
    history = interactions[
        pd.to_datetime(interactions["create_date"], errors="coerce").le(cutoff)
    ]
    return {
        str(profile): set(group["content_id"].astype(str))
        for profile, group in history.groupby("profile_id")
    }


INTERACTION_COLUMNS = ["profile_id", "content_id", "create_date"]


@dataclass(frozen=True)
class ContentScorerModel:
    """The two fitted heads and the category coding they were trained with."""

    heads: dict[str, Any]
    level_maps: dict[str, dict[str, int]]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class ContentServingScope:
    """Everything the scorer reads at one serving cutoff, shared across shards."""

    cutoff: pd.Timestamp
    scope: ContentScope
    metrics: dict[str, Any]
    interactions: pd.DataFrame
    popularity_all: np.ndarray
    popularity_recent: np.ndarray


def _normal_membership(canonical: Path) -> pd.DataFrame:
    from recsys_vnpt.rule_assembly import SPECIAL_RULES

    membership = (
        pd.read_parquet(
            canonical / "content_rule_membership.parquet",
            columns=["content_id", "rule_id"],
        )
        .drop_duplicates()
        .astype({"content_id": str, "rule_id": str})
    )
    return membership[~membership["rule_id"].isin(SPECIAL_RULES)]


def _training_windows(canonical: Path, evaluation_split: str) -> list[Any]:
    """Strictly pre-cutoff windows: the first three for a validation-split
    evaluation, plus the validation window for a test-split evaluation and for
    serving (the locked untouched-test refit protocol)."""
    if evaluation_split not in {"validation", "test"}:
        raise ValueError("content scorer split must be validation or test")
    windows = load_temporal_windows(canonical / "temporal")
    return list(windows[: 4 if evaluation_split == "test" else 3])


def fit_content_scorer(
    canonical: Path,
    interactions: pd.DataFrame,
    windows: list[Any],
    window_scopes: dict[str, ContentScope],
    level_maps: dict[str, dict[str, int]],
    seed: int = 42,
    negative_ratio: int = 4,
    negative_strategy: str = "rule_head",
    negative_head_size: int = 200,
    observed_weight: float = 3.0,
    learning_rate: float = 0.06,
    # 400 is a deliberate cap, not an unreached ceiling: early stopping wants
    # to run further, and letting it does measurably worse. The cap acts as
    # regularisation.
    max_iter: int = 400,
    max_leaf_nodes: int = 31,
    min_samples_leaf: int = 50,
    l2_regularization: float = 1.0,
    backend: str = "sklearn",
    device: str = "auto",
) -> ContentScorerModel:
    """Fit both gain heads on sampled pairs from the temporal windows."""
    event_times = pd.to_datetime(interactions["create_date"], errors="coerce")
    minimum_event, maximum_event = event_times.min(), event_times.max()
    if (
        pd.isna(maximum_event)
        or windows[0].label_start > maximum_event
        or windows[-1].label_end < minimum_event
    ):
        raise ValueError(
            "content scorer temporal windows "
            f"({windows[0].label_start:%Y-%m-%d}.."
            f"{windows[-1].label_end:%Y-%m-%d}) do not overlap the "
            f"interaction span ({minimum_event}..{maximum_event}); rebuild "
            "temporal artifacts (pipeline_cli build-temporal --window-mode relative)"
        )

    train_parts: list[pd.DataFrame] = []
    sampler_counters: dict[str, int] = {}
    for window in windows:
        scope = window_scopes[window.name]
        positives = _window_positives(canonical / "temporal" / f"{window.name}.parquet")
        positives = positives[positives["content_id"].isin(scope.index)]
        if positives.empty:
            continue
        popularity_all, popularity_recent = popularity_at(
            interactions, window.history_end, scope
        )
        bands = popularity_bands(popularity_all)
        pairs, counters = sample_training_pairs(
            positives,
            scope,
            bands,
            _consumed_before(interactions, window.history_end),
            ratio=negative_ratio,
            seed=seed,
            strategy=negative_strategy,
            popularity=popularity_all,
            head_size=negative_head_size,
        )
        for key, value in counters.items():
            sampler_counters[key] = sampler_counters.get(key, 0) + value
        features = pair_features(
            pairs, interactions, window.history_end, scope, popularity_all, popularity_recent
        )
        part = pd.concat(
            [pairs[["profile_id", *GAIN_COLUMNS]].reset_index(drop=True), features],
            axis=1,
        )
        part["window"] = window.name
        train_parts.append(part)

    if not train_parts:
        raise ValueError(
            "content scorer has no positive training rows in windows "
            f"{[w.name for w in windows]}; rebuild temporal artifacts "
            "(pipeline_cli build-temporal --window-mode relative)"
        )
    training = pd.concat(train_parts, ignore_index=True)
    if not any((training[gain] > 0).any() for gain in GAIN_COLUMNS):
        raise ValueError("content scorer training rows carry no positive gain")
    if float(training["tag_affinity"].abs().sum()) == 0 and float(
        training["profile_events_log"].sum()
    ) == 0:
        # Without any user-by-item signal the model degenerates to a global
        # popularity curve and RRF would de-personalize every profile.
        raise ValueError(
            "content scorer training features are degenerate (no pre-window "
            "history overlaps the temporal windows)"
        )

    x_train = training[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    heads: dict[str, Any] = {}
    fit_metrics: dict[str, Any] = {}
    for gain in GAIN_COLUMNS:
        denominators = (
            training.groupby(["window", "profile_id"], sort=False)[gain]
            .transform("sum")
            .clip(lower=1e-12)
        )
        target = (training[gain] / denominators).to_numpy(dtype=np.float32)
        weights = np.where(target > 0, observed_weight, 1.0).astype(np.float32)
        head = make_head(
            backend,
            device,
            [column in CATEGORICAL_FEATURES for column in FEATURE_COLUMNS],
            seed,
            learning_rate,
            max_iter,
            max_leaf_nodes,
            min_samples_leaf,
            l2_regularization,
        )
        head.fit(x_train, target, sample_weight=weights)
        heads[gain] = head
        fit_metrics[gain] = {
            "iterations": int(head.n_iter_),
            "target_positive_rate": float(np.mean(target > 0)),
        }

    sampled = sum(sampler_counters.values()) - sampler_counters.get("exhausted", 0)
    return ContentScorerModel(
        heads=heads,
        level_maps=level_maps,
        metrics={
            "content_scorer_backend": backend,
            "content_scorer_device": head_device(next(iter(heads.values()))),
            "content_scorer_training_windows": [w.name for w in windows],
            "content_scorer_training_rows": int(len(training)),
            "content_scorer_feature_columns": list(FEATURE_COLUMNS),
            "content_scorer_fit": fit_metrics,
            "content_scorer_negative_ratio": int(negative_ratio),
            "content_scorer_negative_strategy": negative_strategy,
            "content_scorer_negative_head_size": int(negative_head_size),
            "content_scorer_sampler": dict(sampler_counters),
            "content_scorer_matched_rate": (
                (
                    sampler_counters.get("same_rule_head", 0)
                    + sampler_counters.get("same_rule_same_band", 0)
                )
                / sampled
                if sampled
                else 0.0
            ),
        },
    )


def _window_scopes(
    canonical: Path,
    windows: list[Any],
    membership: pd.DataFrame,
    level_maps: dict[str, dict[str, int]],
) -> dict[str, ContentScope]:
    return {
        window.name: content_scope_as_of(
            canonical, window.history_end, membership, level_maps
        )[0]
        for window in windows
    }


def train_content_scorer(
    canonical: Path,
    cutoff: pd.Timestamp,
    interactions_loader: Callable[[set[str], list[str]], pd.DataFrame],
    evaluation_split: str = "test",
    **fit_parameters: Any,
) -> ContentScorerModel:
    """Fit a scorer from the canonical artifacts alone, for persisting.

    ``cutoff`` is the serving cutoff of the same data: the category level maps
    are built over the catalog servable there, exactly as a scorer fitted at
    serving time would build them.
    """
    membership = _normal_membership(canonical)
    _, level_maps, _ = content_scope_as_of(canonical, cutoff, membership)
    windows = _training_windows(canonical, evaluation_split)
    window_scopes = _window_scopes(canonical, windows, membership, level_maps)
    content_ids = set().union(*(scope.index for scope in window_scopes.values()))
    interactions = interactions_loader(content_ids, INTERACTION_COLUMNS)
    return fit_content_scorer(
        canonical, interactions, windows, window_scopes, level_maps, **fit_parameters
    )


def content_serving_scope(
    canonical: Path,
    cutoff: pd.Timestamp,
    model: ContentScorerModel,
    interactions_loader: Callable[[set[str], list[str]], pd.DataFrame],
) -> ContentServingScope:
    """Servable catalog, history and popularity at a serving cutoff, coded
    with the level maps the model was trained on."""
    scope, _, metrics = content_scope_as_of(
        canonical, cutoff, _normal_membership(canonical), model.level_maps
    )
    interactions = interactions_loader(set(scope.index), INTERACTION_COLUMNS)
    popularity_all, popularity_recent = popularity_at(interactions, cutoff, scope)
    return ContentServingScope(
        cutoff=cutoff,
        scope=scope,
        metrics=metrics,
        interactions=interactions,
        popularity_all=popularity_all,
        popularity_recent=popularity_recent,
    )


def _scorable(candidate_pairs: pd.DataFrame) -> pd.DataFrame:
    required = {"profile_id", "content_id", "candidate_rule_id", "retrieval_score", "rule_score"}
    missing = sorted(required - set(candidate_pairs.columns))
    if missing:
        raise ValueError(f"content scorer candidates are missing columns: {missing}")
    scored = candidate_pairs.copy()
    scored["profile_id"] = scored["profile_id"].astype(str)
    scored["content_id"] = scored["content_id"].astype(str)
    return scored


def content_scorer_signal(
    model: ContentScorerModel,
    serving: ContentServingScope,
    candidate_pairs: pd.DataFrame,
    rank_scope: str = "rule",
    chunk_size: int = 250_000,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """The learned signal fused into the content order: the reciprocal ranks
    of both predicted gains, inside each rule block or inside the profile.

    A rule block is scored only when every one of its contents can be
    featurised; mixing a learned rank with an unscored fallback inside one
    block would put two different scales on the same ladder. Uncovered rows
    get no learned credit.
    """
    scored = _scorable(candidate_pairs)
    serving_scope = serving.scope
    in_scope = scored["content_id"].isin(serving_scope.index).to_numpy()
    block_covered = (
        pd.Series(in_scope, index=scored.index)
        .groupby([scored["profile_id"], scored["candidate_rule_id"]])
        .transform("all")
        .to_numpy()
    )
    target_rows = scored[block_covered]
    predictions = {gain: np.full(len(scored), -np.inf) for gain in GAIN_COLUMNS}
    if len(target_rows):
        context = build_history_context(
            target_rows["profile_id"], serving.interactions, serving.cutoff, serving_scope
        )
        positions = np.flatnonzero(block_covered)
        for start in range(0, len(target_rows), chunk_size):
            chunk = target_rows.iloc[start : start + chunk_size]
            features = gather_pair_features(
                chunk,
                context,
                serving_scope,
                serving.popularity_all,
                serving.popularity_recent,
            ).to_numpy(dtype=np.float32)
            for gain, head in model.heads.items():
                predictions[gain][positions[start : start + len(chunk)]] = head.predict(
                    features
                )

    learned = sum(
        reciprocal_rank(scored, predictions[gain], rank_scope) for gain in GAIN_COLUMNS
    )
    metrics = {
        **model.metrics,
        "content_scorer_pairs_scored": int(block_covered.sum()),
        "content_scorer_pairs_out_of_scope_rate": float(1.0 - in_scope.mean())
        if len(scored)
        else 0.0,
        "content_scorer_blocks_skipped_rate": float(1.0 - block_covered.mean())
        if len(scored)
        else 0.0,
        **serving.metrics,
    }
    return np.where(block_covered, learned, 0.0), block_covered, metrics


def apply_content_scorer(
    model: ContentScorerModel,
    serving: ContentServingScope,
    candidate_pairs: pd.DataFrame,
    fusion_weight: float = 5.0,
    chunk_size: int = 250_000,
    rank_scope: str = "rule",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Re-rank candidates by RRF fusion of the hand sum and the learned gains.

    ``rank_scope="rule"`` ranks inside each (profile, rule) block and leaves an
    uncovered block on its hand sum. ``"profile"`` ranks over all of the
    profile's candidates, so a content the scorer rates highly keeps that lead
    when the page compares contents across rows.
    """
    if fusion_weight <= 0:
        raise ValueError("content scorer fusion weight must be positive")
    scored = _scorable(candidate_pairs)
    signal, block_covered, metrics = content_scorer_signal(
        model, serving, scored, rank_scope, chunk_size
    )
    fused = rrf_fuse(scored, [signal], rank_scope, fusion_weight)
    scored["retrieval_score"] = (
        np.where(block_covered, fused, scored["retrieval_score"].astype(float))
        if rank_scope == "rule"
        else fused
    )
    return scored, {
        "content_ordering": "learned_rrf",
        "content_scorer_fusion_weight": float(fusion_weight),
        "content_rank_scope": rank_scope,
        **metrics,
    }
