from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class HybridRetrievalConfig:
    embedding_dim: int = 64
    epochs: int = 5
    batch_size: int = 2048
    sampled_items: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    temperature: float = 0.07
    long_history_length: int = 100
    recent_history_length: int = 20
    temporal_holdout_days: int = 10
    item_identity_policy: str = "target_only"
    scoring_catalog_policy: str = "all_input"
    seed: int = 42

    def __post_init__(self) -> None:
        if min(
            self.embedding_dim,
            self.epochs,
            self.batch_size,
            self.sampled_items,
            self.long_history_length,
            self.recent_history_length,
        ) < 1:
            raise ValueError("hybrid retrieval dimensions must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("hybrid retrieval optimizer settings are invalid")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.temporal_holdout_days < 0:
            raise ValueError("temporal holdout days cannot be negative")
        if self.item_identity_policy not in {"target_only", "context_and_target"}:
            raise ValueError(
                "hybrid item identity policy must be target_only or context_and_target"
            )
        if self.scoring_catalog_policy not in {"all_input", "eligible_only"}:
            raise ValueError(
                "hybrid scoring catalog policy must be all_input or eligible_only"
            )


@dataclass
class HybridRetrievalResult:
    rankings: dict[str, list[tuple[str, float]]]
    metrics: dict[str, Any]
    artifact_path: Path | None = None


def _tokens(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return sorted(
        {
            token.strip().casefold()
            for token in str(value).split(",")
            if token.strip()
        }
    )


def _vocabulary(values: list[str]) -> dict[str, int]:
    return {value: index + 1 for index, value in enumerate(sorted(set(values)))}


def _content_id_set_hash(content_ids: set[str] | list[str]) -> str:
    payload = "\n".join(sorted({str(content_id) for content_id in content_ids}))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _categorical_indices(
    values: pd.Series,
    vocabulary: dict[str, int],
) -> np.ndarray:
    return (
        values.astype("string")
        .fillna("__MISSING__")
        .astype(str)
        .map(vocabulary)
        .fillna(0)
        .to_numpy(dtype=np.int64)
    )


def _padded_sequences(
    sequences: list[list[int]],
    length: int,
) -> np.ndarray:
    output = np.zeros((len(sequences), length), dtype=np.int64)
    for index, sequence in enumerate(sequences):
        values = sequence[-length:]
        if values:
            output[index, -len(values) :] = values
    return output


class HybridTwoTower(torch.nn.Module):
    def __init__(
        self,
        user_count: int,
        item_count: int,
        family_count: int,
        type_count: int,
        tag_count: int,
        profile_vocab_sizes: tuple[int, ...],
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.user_embedding = torch.nn.Embedding(
            user_count + 1, embedding_dim, padding_idx=0
        )
        self.item_embedding = torch.nn.Embedding(
            item_count + 1, embedding_dim, padding_idx=0
        )
        self.family_embedding = torch.nn.Embedding(
            family_count + 1, embedding_dim, padding_idx=0
        )
        self.type_embedding = torch.nn.Embedding(
            type_count + 1, embedding_dim, padding_idx=0
        )
        self.tag_embedding = torch.nn.Embedding(
            tag_count + 1, embedding_dim, padding_idx=0
        )
        self.profile_embeddings = torch.nn.ModuleList(
            torch.nn.Embedding(size + 1, embedding_dim, padding_idx=0)
            for size in profile_vocab_sizes
        )
        self.profile_projection = torch.nn.Sequential(
            torch.nn.Linear(4 * embedding_dim, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(embedding_dim),
        )
        self.item_projection = torch.nn.Sequential(
            torch.nn.Linear(4 * embedding_dim, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(embedding_dim),
        )
        for embedding in (
            self.user_embedding,
            self.item_embedding,
            self.family_embedding,
            self.type_embedding,
            self.tag_embedding,
            *self.profile_embeddings,
        ):
            torch.nn.init.normal_(embedding.weight, std=0.02)
            with torch.no_grad():
                embedding.weight[0].zero_()

    @staticmethod
    def _masked_mean(embedding: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        mask = ids.ne(0).unsqueeze(-1)
        return (embedding * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    def item_vectors(
        self,
        item_ids: torch.Tensor,
        family_ids: torch.Tensor,
        type_ids: torch.Tensor,
        tag_ids: torch.Tensor,
    ) -> torch.Tensor:
        item = self.item_embedding(item_ids)
        family = self.family_embedding(family_ids)
        type_value = self.type_embedding(type_ids)
        tags = self._masked_mean(self.tag_embedding(tag_ids), tag_ids)
        return F.normalize(
            self.item_projection(torch.cat([item, family, type_value, tags], dim=-1)),
            dim=-1,
        )

    def profile_vectors(
        self,
        user_ids: torch.Tensor,
        profile_features: torch.Tensor,
        long_history: torch.Tensor,
        recent_history: torch.Tensor,
    ) -> torch.Tensor:
        identity = self.user_embedding(user_ids)
        metadata_parts = [
            embedding(profile_features[:, index])
            for index, embedding in enumerate(self.profile_embeddings)
        ]
        metadata = (
            torch.stack(metadata_parts, dim=0).mean(dim=0)
            if metadata_parts
            else torch.zeros_like(identity)
        )
        long_term = self._masked_mean(
            self.item_embedding(long_history), long_history
        )
        recent = self._masked_mean(
            self.item_embedding(recent_history), recent_history
        )
        return F.normalize(
            self.profile_projection(
                torch.cat([identity, metadata, long_term, recent], dim=-1)
            ),
            dim=-1,
        )


def _prepare_item_features(
    contents: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, int],
    dict[str, dict[str, int]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    items = contents.drop_duplicates("content_id", keep="last").copy()
    items["content_id"] = items["content_id"].astype(str)
    items = items.sort_values("content_id", kind="stable").reset_index(drop=True)
    item_map = {
        content_id: index + 1
        for index, content_id in enumerate(items["content_id"])
    }
    family = (
        items["content_family"].astype("string").fillna("__MISSING__").astype(str)
        if "content_family" in items
        else pd.Series("__MISSING__", index=items.index)
    )
    type_values = (
        items["type_id"].astype("string").fillna("__MISSING__").astype(str)
        if "type_id" in items
        else pd.Series("__MISSING__", index=items.index)
    )
    family_vocab = _vocabulary(family.tolist())
    type_vocab = _vocabulary(type_values.tolist())
    tag_values = items.get("tag_names", pd.Series("", index=items.index))
    rule_values = items.get("rule_ids", pd.Series("", index=items.index))
    tag_lists = [
        sorted(
            set(_tokens(tag_value))
            | {f"rule:{rule_id}" for rule_id in _tokens(rule_value)}
        )
        for tag_value, rule_value in zip(tag_values, rule_values)
    ]
    tag_vocab = _vocabulary([tag for values in tag_lists for tag in values])
    maximum_tags = max((len(values) for values in tag_lists), default=1)
    maximum_tags = min(max(maximum_tags, 1), 32)
    tag_ids = np.zeros((len(items), maximum_tags), dtype=np.int64)
    for index, values in enumerate(tag_lists):
        encoded = [tag_vocab[tag] for tag in values[:maximum_tags]]
        if encoded:
            tag_ids[index, : len(encoded)] = encoded
    vocabs = {
        "family": family_vocab,
        "type": type_vocab,
        "tag": tag_vocab,
    }
    return (
        items,
        item_map,
        vocabs,
        _categorical_indices(family, family_vocab),
        _categorical_indices(type_values, type_vocab),
        tag_ids,
    )


PROFILE_METADATA_COLUMNS = ("package_code", "sex", "province")


def _prepare_profile_features(
    history: pd.DataFrame,
    profile_metadata: pd.DataFrame | None,
    item_map: dict[str, int],
    config: HybridRetrievalConfig,
) -> tuple[
    dict[str, int],
    dict[str, dict[str, int]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    profile_ids = sorted(history["profile_id"].astype(str).unique())
    user_map = {profile_id: index + 1 for index, profile_id in enumerate(profile_ids)}
    metadata = pd.DataFrame({"profile_id": profile_ids})
    if profile_metadata is not None and not profile_metadata.empty:
        normalized = profile_metadata.drop_duplicates("profile_id", keep="last").copy()
        normalized["profile_id"] = normalized["profile_id"].astype(str)
        available = [
            column for column in PROFILE_METADATA_COLUMNS if column in normalized
        ]
        metadata = metadata.merge(
            normalized[["profile_id", *available]], on="profile_id", how="left"
        )
    profile_vocabs: dict[str, dict[str, int]] = {}
    metadata_columns: list[np.ndarray] = []
    for column in PROFILE_METADATA_COLUMNS:
        values = (
            metadata[column].astype("string").fillna("__MISSING__").astype(str)
            if column in metadata
            else pd.Series("__MISSING__", index=metadata.index)
        )
        vocabulary = _vocabulary(values.tolist())
        profile_vocabs[column] = vocabulary
        metadata_columns.append(_categorical_indices(values, vocabulary))
    profile_features = np.column_stack(metadata_columns).astype(np.int64)

    ordered = history.sort_values(
        ["profile_id", "event_end", "content_id"], kind="stable"
    )
    sequences = {
        str(profile_id): [
            item_map.get(str(content_id), 0)
            for content_id in group["content_id"]
        ]
        for profile_id, group in ordered.groupby("profile_id", sort=True)
    }
    long_history = _padded_sequences(
        [sequences.get(profile_id, []) for profile_id in profile_ids],
        config.long_history_length,
    )
    recent_history = _padded_sequences(
        [sequences.get(profile_id, []) for profile_id in profile_ids],
        config.recent_history_length,
    )
    return user_map, profile_vocabs, profile_features, long_history, recent_history


def _profile_batch_arrays(
    profile_ids: list[str],
    user_map: dict[str, int],
    profile_vocabs: dict[str, dict[str, int]],
    known_features: np.ndarray,
    known_long: np.ndarray,
    known_recent: np.ndarray,
    profile_metadata: pd.DataFrame | None,
    config: HybridRetrievalConfig,
    serving_history: pd.DataFrame | None = None,
    item_map: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reverse_user = {profile_id: index for index, profile_id in enumerate(user_map)}
    metadata_map: dict[str, pd.Series] = {}
    if profile_metadata is not None and not profile_metadata.empty:
        normalized = profile_metadata.drop_duplicates("profile_id", keep="last").copy()
        normalized["profile_id"] = normalized["profile_id"].astype(str)
        metadata_map = {
            str(row["profile_id"]): row for _, row in normalized.iterrows()
        }
    user_ids = np.asarray([user_map.get(profile_id, 0) for profile_id in profile_ids])
    features = np.zeros(
        (len(profile_ids), len(PROFILE_METADATA_COLUMNS)), dtype=np.int64
    )
    long_history = np.zeros(
        (len(profile_ids), config.long_history_length), dtype=np.int64
    )
    recent_history = np.zeros(
        (len(profile_ids), config.recent_history_length), dtype=np.int64
    )
    for row_index, profile_id in enumerate(profile_ids):
        known_index = reverse_user.get(profile_id)
        if known_index is not None:
            features[row_index] = known_features[known_index]
            long_history[row_index] = known_long[known_index]
            recent_history[row_index] = known_recent[known_index]
            continue
        metadata_row = metadata_map.get(profile_id)
        if metadata_row is None:
            continue
        for column_index, column in enumerate(PROFILE_METADATA_COLUMNS):
            raw_value = metadata_row.get(column, "__MISSING__")
            value = "__MISSING__" if pd.isna(raw_value) else str(raw_value)
            features[row_index, column_index] = profile_vocabs[column].get(value, 0)
    if serving_history is not None and item_map is not None and not serving_history.empty:
        ordered = serving_history.sort_values(
            ["profile_id", "event_end", "content_id"], kind="stable"
        )
        sequences = {
            str(profile_id): [
                item_map.get(str(content_id), 0)
                for content_id in group["content_id"]
            ]
            for profile_id, group in ordered.groupby("profile_id", sort=True)
        }
        long_history = _padded_sequences(
            [sequences.get(profile_id, []) for profile_id in profile_ids],
            config.long_history_length,
        )
        recent_history = _padded_sequences(
            [sequences.get(profile_id, []) for profile_id in profile_ids],
            config.recent_history_length,
        )
    return user_ids, features, long_history, recent_history


def _sample_unobserved(
    user_ids: np.ndarray,
    item_count: int,
    consumed: dict[int, set[int]],
    sample_count: int,
    probabilities: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    output = np.empty((len(user_ids), sample_count), dtype=np.int64)
    values = np.arange(1, item_count + 1)
    for row, user_id in enumerate(user_ids):
        blocked = consumed.get(int(user_id), set())
        available = item_count - len(blocked)
        if available <= 0:
            output[row] = rng.choice(
                values, size=sample_count, replace=True, p=probabilities
            )
            continue
        selected: list[int] = []
        while len(selected) < sample_count:
            proposals = rng.choice(
                values,
                size=max(sample_count * 2, 16),
                replace=True,
                p=probabilities,
            )
            selected.extend(
                int(value) for value in proposals if int(value) not in blocked
            )
        output[row] = selected[:sample_count]
    return output


def _leave_one_out_history(
    history: np.ndarray,
    positive_items: np.ndarray,
) -> np.ndarray:
    """Remove the target item from each training profile history."""
    output = history.copy()
    output[output == positive_items[:, None]] = 0
    return output


def _identity_item_indices(
    context_history: pd.DataFrame,
    target_item_indices: np.ndarray,
    item_map: dict[str, int],
    policy: str,
) -> set[int]:
    indices = set(map(int, np.unique(target_item_indices)))
    if policy == "context_and_target":
        indices.update(
            context_history["content_id"]
            .map(item_map)
            .dropna()
            .astype(int)
            .tolist()
        )
    return indices


def _deterministic_topk(
    scores: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select exact-score ties at the K boundary by ascending item index."""
    if scores.ndim != 1:
        raise ValueError("deterministic top-k expects a one-dimensional score row")
    if k < 1 or k > scores.numel():
        raise ValueError("deterministic top-k K is out of range")
    top_values, _ = torch.topk(scores, k=k)
    threshold = top_values.min()
    greater = torch.nonzero(scores > threshold, as_tuple=False).flatten()
    equal = torch.sort(
        torch.nonzero(scores == threshold, as_tuple=False).flatten()
    ).values
    needed = k - int(greater.numel())
    selected = torch.cat([greater, equal[:needed]])
    selected_values = scores[selected]
    values = selected_values.detach().cpu().numpy()
    indices = selected.detach().cpu().numpy()
    order = np.lexsort((indices, -values))
    ordered = selected[
        torch.as_tensor(order, dtype=torch.long, device=selected.device)
    ]
    return scores[ordered], ordered


def train_and_rank_hybrid_retriever(
    playback_events: pd.DataFrame,
    contents: pd.DataFrame,
    profiles: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    candidate_budget: int = 1000,
    profile_metadata: pd.DataFrame | None = None,
    config: HybridRetrievalConfig = HybridRetrievalConfig(),
    output_dir: str | Path | None = None,
    device: str | None = None,
    eligible_content_ids: set[str] | None = None,
) -> HybridRetrievalResult:
    if candidate_budget < 1:
        raise ValueError("candidate_budget must be positive")
    required_events = {
        "profile_id",
        "content_id",
        "event_end",
        "watch_seconds",
    }
    if not required_events.issubset(playback_events):
        raise ValueError(
            f"hybrid retrieval is missing {sorted(required_events - set(playback_events))}"
        )
    cutoff_value = pd.Timestamp(cutoff)
    history = playback_events.copy()
    history["event_end"] = pd.to_datetime(history["event_end"], errors="coerce")
    history = history[history["event_end"].le(cutoff_value)].copy()
    history["profile_id"] = history["profile_id"].astype(str)
    history["content_id"] = history["content_id"].astype(str)
    history["watch_seconds"] = pd.to_numeric(
        history["watch_seconds"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    history = history[history["watch_seconds"].gt(0)]
    if history.empty:
        raise ValueError("hybrid retrieval has no pre-cutoff playback history")

    (
        items,
        item_map,
        item_vocabs,
        family_ids,
        type_ids,
        tag_ids,
    ) = _prepare_item_features(contents)
    normalized_eligible_ids = (
        {str(content_id) for content_id in eligible_content_ids} & set(item_map)
        if eligible_content_ids is not None
        else set(item_map)
    )
    if (
        config.scoring_catalog_policy == "eligible_only"
        and eligible_content_ids is None
    ):
        raise ValueError(
            "eligible-only hybrid scoring requires eligible_content_ids"
        )
    scoring_items = (
        items[items["content_id"].isin(normalized_eligible_ids)].copy()
        if config.scoring_catalog_policy == "eligible_only"
        else items.copy()
    )
    if scoring_items.empty:
        raise ValueError("hybrid scoring catalog is empty")
    scoring_item_indices = scoring_items["content_id"].map(item_map).to_numpy(
        dtype=np.int64
    )
    scoring_content_ids = set(scoring_items["content_id"].astype(str))
    scoring_catalog_hash = _content_id_set_hash(scoring_content_ids)
    eligible_catalog_hash = _content_id_set_hash(normalized_eligible_ids)
    history = history[history["content_id"].isin(item_map)]
    query_cutoff = cutoff_value - pd.Timedelta(
        config.temporal_holdout_days, unit="D"
    )
    if config.temporal_holdout_days > 0:
        context_history = history[history["event_end"].le(query_cutoff)].copy()
        target_history = history[history["event_end"].gt(query_cutoff)].copy()
    else:
        context_history = history.copy()
        target_history = history.copy()
    if context_history.empty or target_history.empty:
        raise ValueError(
            "hybrid temporal objective requires non-empty context and target history"
        )
    (
        user_map,
        profile_vocabs,
        profile_features,
        long_history,
        recent_history,
    ) = _prepare_profile_features(
        context_history, profile_metadata, item_map, config
    )

    edges = target_history.groupby(
        ["profile_id", "content_id"], as_index=False, sort=True
    ).agg(
        total_watch_seconds=("watch_seconds", "sum"),
        max_episode_watch_seconds=("watch_seconds", "max"),
    )
    edges = edges[edges["profile_id"].isin(user_map)].copy()
    context_consumed_keys = set(
        context_history[["profile_id", "content_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if config.temporal_holdout_days > 0:
        novel_mask = [
            (profile_id, content_id) not in context_consumed_keys
            for profile_id, content_id in edges[
                ["profile_id", "content_id"]
            ].itertuples(index=False, name=None)
        ]
        edges = edges.loc[novel_mask].copy()
    if edges.empty:
        raise ValueError("hybrid temporal objective has no novel target edges")
    edges["user_index"] = edges["profile_id"].map(user_map).astype(int)
    edges["item_index"] = edges["content_id"].map(item_map).astype(int)
    context_pair_watch = context_history.groupby(
        ["profile_id", "content_id"], sort=False
    )["watch_seconds"].sum()
    training_p95_watch = max(
        float(context_pair_watch.quantile(0.95)), 1.0
    )
    watch_time_gain = (
        np.log1p(
            np.minimum(
                edges["total_watch_seconds"].to_numpy(dtype=float),
                training_p95_watch,
            )
        )
        / np.log1p(training_p95_watch)
    )
    atomic_duration = pd.to_numeric(
        contents.set_index(contents["content_id"].astype(str)).get(
            "atomic_duration_seconds",
            pd.Series(dtype=float),
        ),
        errors="coerce",
    )
    edge_atomic_duration = edges["content_id"].map(atomic_duration)
    completion_gain = (
        edges["max_episode_watch_seconds"]
        / edge_atomic_duration.where(edge_atomic_duration.gt(0))
    ).fillna(0.0).clip(lower=0.0, upper=1.0).to_numpy(dtype=float)
    edge_gain = np.clip(
        0.5 * completion_gain + 0.5 * watch_time_gain,
        1e-6,
        1.0,
    ).astype(np.float32)
    frequency = np.bincount(
        edges["item_index"].to_numpy(dtype=int), minlength=len(item_map) + 1
    )[1:].astype(np.float64)
    sampling_probabilities = np.power(frequency + 1e-12, 0.75)
    sampling_probabilities /= sampling_probabilities.sum()
    consumed_history = history[history["profile_id"].isin(user_map)].copy()
    consumed_history["user_index"] = consumed_history["profile_id"].map(user_map)
    consumed_history["item_index"] = consumed_history["content_id"].map(item_map)
    consumed = {
        int(user_id): set(group["item_index"].dropna().astype(int))
        for user_id, group in consumed_history.groupby("user_index")
    }

    target = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model = HybridTwoTower(
        user_count=len(user_map),
        item_count=len(item_map),
        family_count=len(item_vocabs["family"]),
        type_count=len(item_vocabs["type"]),
        tag_count=len(item_vocabs["tag"]),
        profile_vocab_sizes=tuple(len(profile_vocabs[column]) for column in PROFILE_METADATA_COLUMNS),
        embedding_dim=config.embedding_dim,
    ).to(target)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    rng = np.random.default_rng(config.seed)
    losses: list[float] = []
    train_started = time.perf_counter()
    edge_users = edges["user_index"].to_numpy(dtype=np.int64)
    edge_items = edges["item_index"].to_numpy(dtype=np.int64)
    item_family = np.concatenate([[0], family_ids])
    item_type = np.concatenate([[0], type_ids])
    item_tags = np.vstack([np.zeros((1, tag_ids.shape[1]), dtype=np.int64), tag_ids])

    for _ in range(config.epochs):
        permutation = rng.permutation(len(edges))
        running = 0.0
        batches = 0
        for start in range(0, len(permutation), config.batch_size):
            batch = permutation[start : start + config.batch_size]
            users = edge_users[batch]
            positives = edge_items[batch]
            negatives = _sample_unobserved(
                users,
                len(item_map),
                consumed,
                config.sampled_items,
                sampling_probabilities,
                rng,
            )
            sampled = np.column_stack([positives, negatives])
            user_rows = users - 1
            batch_long_history = _leave_one_out_history(
                long_history[user_rows], positives
            )
            batch_recent_history = _leave_one_out_history(
                recent_history[user_rows], positives
            )
            user_vector = model.profile_vectors(
                torch.as_tensor(users, device=target),
                torch.as_tensor(profile_features[user_rows], device=target),
                torch.as_tensor(batch_long_history, device=target),
                torch.as_tensor(batch_recent_history, device=target),
            )
            flattened = sampled.reshape(-1)
            sampled_items = model.item_vectors(
                torch.as_tensor(flattened, device=target),
                torch.as_tensor(item_family[flattened], device=target),
                torch.as_tensor(item_type[flattened], device=target),
                torch.as_tensor(item_tags[flattened], device=target),
            ).reshape(len(batch), 1 + config.sampled_items, -1)
            logits = torch.einsum("bd,bkd->bk", user_vector, sampled_items)
            logits = logits / config.temperature
            q = np.concatenate([[0.0], sampling_probabilities])
            correction_values = np.zeros_like(sampled, dtype=np.float32)
            correction_values[:, 1:] = np.log(
                np.clip(config.sampled_items * q[negatives], 1e-12, None)
            )
            correction = torch.as_tensor(
                correction_values,
                dtype=logits.dtype,
                device=target,
            )
            corrected_logits = logits - correction
            per_row_loss = F.cross_entropy(
                corrected_logits,
                torch.zeros(len(batch), dtype=torch.long, device=target),
                reduction="none",
            )
            weights = torch.as_tensor(edge_gain[batch], device=target)
            loss = (per_row_loss * weights).sum() / weights.sum().clamp_min(1e-12)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu())
            batches += 1
        losses.append(running / max(batches, 1))
    training_seconds = time.perf_counter() - train_started

    requested_profiles = profiles["profile_id"].dropna().astype(str).drop_duplicates().tolist()
    (
        request_user_ids,
        request_features,
        request_long,
        request_recent,
    ) = _profile_batch_arrays(
        requested_profiles,
        user_map,
        profile_vocabs,
        profile_features,
        long_history,
        recent_history,
        profile_metadata,
        config,
        serving_history=history,
        item_map=item_map,
    )
    rank_started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        item_batches: list[torch.Tensor] = []
        all_item_ids = scoring_item_indices
        target_item_indices = set(map(int, np.unique(edge_items)))
        identity_item_indices = _identity_item_indices(
            context_history,
            edge_items,
            item_map,
            config.item_identity_policy,
        )
        scoring_item_ids = np.asarray(
            [
                item_index if item_index in identity_item_indices else 0
                for item_index in all_item_ids
            ],
            dtype=np.int64,
        )
        for start in range(0, len(all_item_ids), 8192):
            ids = all_item_ids[start : start + 8192]
            model_ids = scoring_item_ids[start : start + 8192]
            item_batches.append(
                model.item_vectors(
                    torch.as_tensor(model_ids, device=target),
                    torch.as_tensor(item_family[ids], device=target),
                    torch.as_tensor(item_type[ids], device=target),
                    torch.as_tensor(item_tags[ids], device=target),
                ).cpu()
            )
        all_item_vectors = torch.cat(item_batches, dim=0).to(target)
        content_ids = scoring_items["content_id"].astype(str).tolist()
        score_column_by_item_index = {
            int(item_index): column
            for column, item_index in enumerate(all_item_ids)
        }
        rankings: dict[str, list[tuple[str, float]]] = {}
        retrieved_content_ids: set[str] = set()
        score_sum = 0.0
        score_square_sum = 0.0
        score_count = 0
        top_score_sum = 0.0
        kth_score_sum = 0.0
        cold_item_recommendations = 0
        for start in range(0, len(requested_profiles), 256):
            stop = min(start + 256, len(requested_profiles))
            profile_vectors = model.profile_vectors(
                torch.as_tensor(request_user_ids[start:stop], device=target),
                torch.as_tensor(request_features[start:stop], device=target),
                torch.as_tensor(request_long[start:stop], device=target),
                torch.as_tensor(request_recent[start:stop], device=target),
            )
            scores = profile_vectors @ all_item_vectors.T
            for row, profile_id in enumerate(requested_profiles[start:stop]):
                blocked = consumed.get(user_map.get(profile_id, 0), set())
                blocked_columns = [
                    score_column_by_item_index[item_index]
                    for item_index in blocked
                    if item_index in score_column_by_item_index
                ]
                if blocked_columns:
                    blocked_tensor = torch.as_tensor(
                        blocked_columns,
                        device=target,
                    )
                    scores[row, blocked_tensor] = -torch.inf
                k = min(
                    candidate_budget,
                    len(content_ids) - len(blocked_columns),
                )
                if k <= 0:
                    rankings[profile_id] = []
                    continue
                values, indices = _deterministic_topk(scores[row], k=k)
                value_list = values.detach().cpu().tolist()
                index_list = indices.detach().cpu().tolist()
                rankings[profile_id] = [
                    (content_ids[int(index)], float(value))
                    for value, index in zip(
                        value_list,
                        index_list,
                    )
                ]
                retrieved_content_ids.update(
                    content_ids[int(index)] for index in index_list
                )
                score_sum += float(sum(value_list))
                score_square_sum += float(sum(value * value for value in value_list))
                score_count += len(value_list)
                top_score_sum += float(value_list[0])
                kth_score_sum += float(value_list[-1])
                cold_item_recommendations += sum(
                    int(
                        int(all_item_ids[int(index)])
                        not in identity_item_indices
                    )
                    for index in index_list
                )
    ranking_seconds = time.perf_counter() - rank_started

    history_hash = hashlib.sha256(
        pd.util.hash_pandas_object(
            edges[["profile_id", "content_id", "total_watch_seconds"]],
            index=False,
            categorize=True,
        ).to_numpy().tobytes()
    ).hexdigest()
    artifact_path: Path | None = None
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        artifact_path = directory / "hybrid_two_tower_v1.pth"
        torch.save(
            {
                "schema_version": 3,
                "config": asdict(config),
                "history_hash": history_hash,
                "eligible_catalog_hash": eligible_catalog_hash,
                "scoring_catalog_hash": scoring_catalog_hash,
                "model_state_dict": model.state_dict(),
                "user_map": user_map,
                "item_map": item_map,
                "item_vocabs": item_vocabs,
                "profile_vocabs": profile_vocabs,
            },
            artifact_path,
        )
        (directory / "hybrid_two_tower_v1.json").write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "config": asdict(config),
                    "history_hash": history_hash,
                    "eligible_catalog_hash": eligible_catalog_hash,
                    "scoring_catalog_hash": scoring_catalog_hash,
                    "target_semantics": "temporal novel engagement gain sampled softmax",
                    "unobserved_semantics": "contrastive sample, not a true negative",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    metrics = {
        "model_family": "hybrid_two_tower_v1",
        "target_semantics": "temporal_novel_engagement_gain_sampled_softmax",
        "classification_metrics_valid": False,
        "history_events": int(len(history)),
        "context_events": int(len(context_history)),
        "target_events": int(len(target_history)),
        "query_cutoff": str(query_cutoff),
        "serving_cutoff": str(cutoff_value),
        "temporal_holdout_days": config.temporal_holdout_days,
        "history_pairs": int(len(edges)),
        "history_profiles": int(len(user_map)),
        "catalog_contents": int(len(item_map)),
        "eligible_catalog_contents": int(len(normalized_eligible_ids)),
        "eligible_catalog_hash": eligible_catalog_hash,
        "scoring_catalog_policy": config.scoring_catalog_policy,
        "scored_catalog_contents": int(len(scoring_content_ids)),
        "scored_ineligible_contents": int(
            len(scoring_content_ids - normalized_eligible_ids)
        ),
        "scoring_catalog_hash": scoring_catalog_hash,
        "scoring_zero_identity_contents": int((scoring_item_ids == 0).sum()),
        "requested_profiles": int(len(requested_profiles)),
        "candidate_budget": int(candidate_budget),
        "training_time_seconds": training_seconds,
        "ranking_time_seconds": ranking_seconds,
        "epochs_completed": len(losses),
        "training_loss_history": losses,
        "history_hash": history_hash,
        "unobserved_semantics": "contrastive sample, not a true negative",
        "cold_profile_count": int(sum(user_id == 0 for user_id in request_user_ids)),
        "history_contents": int(len(target_item_indices)),
        "target_item_identity_contents": int(len(target_item_indices)),
        "item_identity_contents": int(len(identity_item_indices)),
        "cold_catalog_contents": int(len(item_map) - len(identity_item_indices)),
        "cold_item_identity_policy": config.item_identity_policy,
        "metadata_only_item_identity": "zero_id",
        "topk_tie_policy": "score_desc_item_index_asc_at_boundary",
        "leave_one_out_history": True,
        "sampling_correction": "negatives_only_log_sample_count_times_q_approximate",
        "training_p95_watch_seconds": training_p95_watch,
        "mean_target_completion_gain": float(np.mean(completion_gain)),
        "mean_target_watch_time_gain": float(np.mean(watch_time_gain)),
        "mean_target_combined_gain": float(np.mean(edge_gain)),
        "distinct_retrieved_contents": int(len(retrieved_content_ids)),
        "retrieved_catalog_coverage": len(retrieved_content_ids) / max(len(item_map), 1),
        "retrieved_scored_catalog_coverage": len(retrieved_content_ids)
        / max(len(scoring_content_ids), 1),
        "mean_retrieval_score": score_sum / max(score_count, 1),
        "retrieval_score_std": max(
            score_square_sum / max(score_count, 1)
            - (score_sum / max(score_count, 1)) ** 2,
            0.0,
        )
        ** 0.5,
        "mean_top_retrieval_score": top_score_sum / max(len(rankings), 1),
        "mean_kth_retrieval_score": kth_score_sum / max(len(rankings), 1),
        "cold_item_recommendation_rate": cold_item_recommendations
        / max(score_count, 1),
    }
    return HybridRetrievalResult(
        rankings=rankings,
        metrics=metrics,
        artifact_path=artifact_path,
    )
