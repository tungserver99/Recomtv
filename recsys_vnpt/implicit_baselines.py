from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from recsys_vnpt.retrieval import (
    fuse_content_scores_with_external_rank,
    per_profile_score_ranks,
)


IMPLICIT_BASELINE_FAMILIES = (
    "graded_global_popularity",
    "graded_per_rule_popularity",
    "session_item_knn",
    "bpr_mf",
    "bpr_per_rule_fusion",
    "knn_per_rule_fusion",
    "lightgcn",
)


@dataclass(frozen=True)
class ImplicitBaselineConfig:
    family: str
    embedding_dim: int = 32
    epochs: int = 10
    batch_size: int = 8192
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    lightgcn_layers: int = 2
    maximum_lightgcn_edges_per_epoch: int = 200_000
    seed: int = 42
    fusion_bpr_weight: float = 1.0
    fusion_rrf_constant: int = 60

    def __post_init__(self) -> None:
        if self.family not in IMPLICIT_BASELINE_FAMILIES:
            raise ValueError(f"unsupported implicit baseline: {self.family}")
        if self.fusion_bpr_weight < 0 or self.fusion_rrf_constant < 1:
            raise ValueError("implicit baseline fusion settings are invalid")
        if min(
            self.embedding_dim,
            self.epochs,
            self.batch_size,
            self.lightgcn_layers,
            self.maximum_lightgcn_edges_per_epoch,
        ) < 1:
            raise ValueError("implicit baseline limits must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("implicit baseline optimizer settings are invalid")


@dataclass
class ImplicitBaselineResult:
    scores: np.ndarray
    metrics: dict[str, Any]
    artifact_path: Path | None = None


def _history_before(
    playback_events: pd.DataFrame,
    cutoff: str | pd.Timestamp,
) -> pd.DataFrame:
    required = {"profile_id", "content_id", "event_end", "watch_seconds"}
    missing = sorted(required - set(playback_events))
    if missing:
        raise ValueError(f"implicit history is missing columns: {missing}")
    history = playback_events.copy()
    history["event_end"] = pd.to_datetime(history["event_end"], errors="coerce")
    history = history[history["event_end"].le(pd.Timestamp(cutoff))]
    history["profile_id"] = history["profile_id"].astype(str)
    history["content_id"] = history["content_id"].astype(str)
    history["watch_seconds"] = pd.to_numeric(
        history["watch_seconds"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    return history[history["watch_seconds"].gt(0)].reset_index(drop=True)


def _interaction_edges(history: pd.DataFrame) -> pd.DataFrame:
    edges = history.groupby(
        ["profile_id", "content_id"], as_index=False, sort=True
    ).agg(watch_seconds=("watch_seconds", "sum"))
    edges["gain_weight"] = np.log1p(edges["watch_seconds"])
    maximum = float(edges["gain_weight"].max()) if len(edges) else 0.0
    if maximum > 0:
        edges["gain_weight"] /= maximum
    return edges


def _popularity_scores(
    history: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, float]]:
    popularity = history.groupby("content_id")["watch_seconds"].sum()
    popularity = np.log1p(popularity)
    maximum = float(popularity.max()) if len(popularity) else 0.0
    if maximum > 0:
        popularity /= maximum
    scores = (
        candidates["content_id"].astype(str).map(popularity).fillna(0.0).to_numpy(
            dtype=np.float32
        )
    )
    return scores, {str(key): float(value) for key, value in popularity.items()}


def _session_item_knn_scores(
    history: pd.DataFrame,
    candidates: pd.DataFrame,
    fallback: np.ndarray,
) -> np.ndarray:
    if "session_id" not in history or history.empty:
        return fallback.copy()
    item_ids = sorted(
        set(history["content_id"].astype(str))
        | set(candidates["content_id"].astype(str))
    )
    session_ids = sorted(history["session_id"].astype(str).unique())
    item_index = {value: index for index, value in enumerate(item_ids)}
    session_index = {value: index for index, value in enumerate(session_ids)}
    events = history[["session_id", "content_id"]].drop_duplicates().copy()
    session_item = csr_matrix(
        (
            np.ones(len(events), dtype=np.float32),
            (
                events["session_id"].astype(str).map(session_index).to_numpy(),
                events["content_id"].astype(str).map(item_index).to_numpy(),
            ),
        ),
        shape=(len(session_ids), len(item_ids)),
    )
    item_session = normalize(session_item.T.tocsr(), norm="l2", axis=1)
    profile_history = {
        str(profile_id): [
            item_index[content_id]
            for content_id in group["content_id"].astype(str).drop_duplicates()
            if content_id in item_index
        ]
        for profile_id, group in history.groupby("profile_id", sort=False)
    }
    output = fallback.copy()
    for profile_id, indices in candidates.groupby(
        candidates["profile_id"].astype(str), sort=False
    ).groups.items():
        consumed_indices = profile_history.get(str(profile_id), [])
        if not consumed_indices:
            continue
        row_indices = np.asarray(list(indices), dtype=np.int64)
        candidate_indices = [
            item_index[str(content_id)]
            for content_id in candidates.loc[row_indices, "content_id"]
        ]
        similarities = item_session[candidate_indices] @ item_session[
            consumed_indices
        ].T
        collaborative = np.asarray(similarities.sum(axis=1)).reshape(-1)
        if collaborative.size and float(collaborative.max()) > 0:
            collaborative = collaborative / float(collaborative.max())
            output[row_indices] = (
                0.9 * collaborative.astype(np.float32)
                + 0.1 * fallback[row_indices]
            )
    return output


class _BPRMF(torch.nn.Module):
    def __init__(self, users: int, items: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_embedding = torch.nn.Embedding(users, embedding_dim)
        self.item_embedding = torch.nn.Embedding(items, embedding_dim)
        torch.nn.init.normal_(self.user_embedding.weight, std=0.05)
        torch.nn.init.normal_(self.item_embedding.weight, std=0.05)

    def triplet_scores(
        self,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user = self.user_embedding(users)
        return (
            (user * self.item_embedding(positives)).sum(dim=-1),
            (user * self.item_embedding(negatives)).sum(dim=-1),
        )


def _negative_items(
    user_ids: np.ndarray,
    item_count: int,
    consumed: dict[int, set[int]],
    rng: np.random.Generator,
) -> np.ndarray:
    output = rng.integers(0, item_count, size=len(user_ids), dtype=np.int64)
    for index, user_id in enumerate(user_ids):
        blocked = consumed[int(user_id)]
        if len(blocked) >= item_count:
            continue
        while int(output[index]) in blocked:
            output[index] = rng.integers(0, item_count)
    return output


def _maps(edges: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    users = {
        value: index
        for index, value in enumerate(sorted(edges["profile_id"].astype(str).unique()))
    }
    items = {
        value: index
        for index, value in enumerate(sorted(edges["content_id"].astype(str).unique()))
    }
    return users, items


def _train_bpr(
    edges: pd.DataFrame,
    config: ImplicitBaselineConfig,
    device: torch.device,
) -> tuple[_BPRMF, dict[str, int], dict[str, int], list[float]]:
    user_map, item_map = _maps(edges)
    users = edges["profile_id"].map(user_map).to_numpy(dtype=np.int64)
    items = edges["content_id"].map(item_map).to_numpy(dtype=np.int64)
    weights = edges["gain_weight"].to_numpy(dtype=np.float32)
    consumed = {
        int(user_id): set(map(int, group["content_id"].map(item_map)))
        for user_id, group in edges.groupby(edges["profile_id"].map(user_map))
    }
    model = _BPRMF(len(user_map), len(item_map), config.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    rng = np.random.default_rng(config.seed)
    losses: list[float] = []
    for _ in range(config.epochs):
        permutation = rng.permutation(len(edges))
        running = 0.0
        batches = 0
        for start in range(0, len(permutation), config.batch_size):
            batch = permutation[start : start + config.batch_size]
            batch_users = users[batch]
            negatives = _negative_items(
                batch_users, len(item_map), consumed, rng
            )
            user_tensor = torch.as_tensor(batch_users, device=device)
            positive_tensor = torch.as_tensor(items[batch], device=device)
            negative_tensor = torch.as_tensor(negatives, device=device)
            weight_tensor = torch.as_tensor(weights[batch], device=device)
            positive_score, negative_score = model.triplet_scores(
                user_tensor, positive_tensor, negative_tensor
            )
            loss = -(
                torch.nn.functional.logsigmoid(positive_score - negative_score)
                * weight_tensor
            ).sum() / weight_tensor.sum().clamp_min(1e-12)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu())
            batches += 1
        losses.append(running / max(batches, 1))
    return model, user_map, item_map, losses


class _LightGCN(_BPRMF):
    def __init__(
        self,
        users: int,
        items: int,
        embedding_dim: int,
        adjacency: torch.Tensor,
        layers: int,
    ) -> None:
        super().__init__(users, items, embedding_dim)
        self.register_buffer("adjacency", adjacency)
        self.layers = layers

    def propagated(self) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )
        states = [embeddings]
        for _ in range(self.layers):
            embeddings = torch.sparse.mm(self.adjacency, embeddings)
            states.append(embeddings)
        final = torch.stack(states, dim=0).mean(dim=0)
        return final[: self.user_embedding.num_embeddings], final[
            self.user_embedding.num_embeddings :
        ]


def _lightgcn_adjacency(
    users: np.ndarray,
    items: np.ndarray,
    user_count: int,
    item_count: int,
    device: torch.device,
) -> torch.Tensor:
    shifted_items = items + user_count
    rows = np.concatenate([users, shifted_items])
    columns = np.concatenate([shifted_items, users])
    degrees = np.bincount(
        np.concatenate([users, shifted_items]), minlength=user_count + item_count
    ).astype(np.float32)
    values = 1.0 / np.sqrt(
        np.maximum(degrees[rows], 1.0) * np.maximum(degrees[columns], 1.0)
    )
    indices = torch.as_tensor(np.vstack([rows, columns]), dtype=torch.long, device=device)
    value_tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        indices,
        value_tensor,
        (user_count + item_count, user_count + item_count),
        device=device,
    ).coalesce()


def _train_lightgcn(
    edges: pd.DataFrame,
    config: ImplicitBaselineConfig,
    device: torch.device,
) -> tuple[_LightGCN, dict[str, int], dict[str, int], list[float]]:
    user_map, item_map = _maps(edges)
    users = edges["profile_id"].map(user_map).to_numpy(dtype=np.int64)
    items = edges["content_id"].map(item_map).to_numpy(dtype=np.int64)
    weights = edges["gain_weight"].to_numpy(dtype=np.float32)
    consumed = {
        int(user_id): set(map(int, group["content_id"].map(item_map)))
        for user_id, group in edges.groupby(edges["profile_id"].map(user_map))
    }
    adjacency = _lightgcn_adjacency(
        users, items, len(user_map), len(item_map), device
    )
    model = _LightGCN(
        len(user_map),
        len(item_map),
        config.embedding_dim,
        adjacency,
        config.lightgcn_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    rng = np.random.default_rng(config.seed)
    losses: list[float] = []
    for _ in range(config.epochs):
        if len(edges) > config.maximum_lightgcn_edges_per_epoch:
            selected = rng.choice(
                len(edges),
                config.maximum_lightgcn_edges_per_epoch,
                replace=False,
            )
        else:
            selected = np.arange(len(edges))
        selected_users = users[selected]
        negatives = _negative_items(
            selected_users, len(item_map), consumed, rng
        )
        user_embeddings, item_embeddings = model.propagated()
        user_tensor = torch.as_tensor(selected_users, device=device)
        positive_tensor = torch.as_tensor(items[selected], device=device)
        negative_tensor = torch.as_tensor(negatives, device=device)
        weight_tensor = torch.as_tensor(weights[selected], device=device)
        positive_score = (
            user_embeddings[user_tensor] * item_embeddings[positive_tensor]
        ).sum(dim=-1)
        negative_score = (
            user_embeddings[user_tensor] * item_embeddings[negative_tensor]
        ).sum(dim=-1)
        loss = -(
            torch.nn.functional.logsigmoid(positive_score - negative_score)
            * weight_tensor
        ).sum() / weight_tensor.sum().clamp_min(1e-12)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return model, user_map, item_map, losses


def _embedding_scores(
    candidates: pd.DataFrame,
    user_map: dict[str, int],
    item_map: dict[str, int],
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    output = fallback.copy()
    for index, (profile_id, content_id) in enumerate(
        candidates[["profile_id", "content_id"]].astype(str).itertuples(
            index=False, name=None
        )
    ):
        user_index = user_map.get(profile_id)
        item_index = item_map.get(content_id)
        if user_index is None or item_index is None:
            continue
        output[index] = float(
            np.dot(user_embeddings[user_index], item_embeddings[item_index])
        )
    finite = np.isfinite(output)
    if finite.any():
        minimum = float(output[finite].min())
        maximum = float(output[finite].max())
        if maximum > minimum:
            output[finite] = (output[finite] - minimum) / (maximum - minimum)
    return output.astype(np.float32)


def _fused_per_rule_scores(
    candidates: pd.DataFrame,
    external_scores: np.ndarray,
    config: ImplicitBaselineConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """RRF-fuse serving retrieval_score rank with an external per-profile rank."""
    if "retrieval_score" not in candidates:
        raise ValueError(
            f"{config.family} requires serving retrieval_score"
        )
    retrieval_scores = pd.to_numeric(
        candidates["retrieval_score"], errors="coerce"
    )
    if retrieval_scores.isna().any() or not np.isfinite(
        retrieval_scores.to_numpy(dtype=float)
    ).all():
        raise ValueError("serving retrieval_score must be finite")
    fusion_frame = candidates[["profile_id", "content_id"]].copy()
    fusion_frame["external_rank"] = per_profile_score_ranks(
        fusion_frame, external_scores
    )
    return fuse_content_scores_with_external_rank(
        fusion_frame,
        retrieval_scores.to_numpy(dtype=float),
        external_weight=config.fusion_bpr_weight,
        rrf_constant=config.fusion_rrf_constant,
    )


def _history_hash(edges: pd.DataFrame) -> str:
    ordered = edges.sort_values(["profile_id", "content_id"], kind="stable")
    values = pd.util.hash_pandas_object(
        ordered, index=False, categorize=True
    ).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def score_implicit_baseline(
    playback_events: pd.DataFrame,
    candidates: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    config: ImplicitBaselineConfig,
    output_dir: str | Path | None = None,
    device: str | None = None,
) -> ImplicitBaselineResult:
    required_candidates = {"profile_id", "content_id"}
    if not required_candidates.issubset(candidates):
        raise ValueError("implicit baseline candidates require profile_id/content_id")
    started = time.perf_counter()
    history = _history_before(playback_events, cutoff)
    edges = _interaction_edges(history)
    if edges.empty:
        raise ValueError("implicit baseline has no pre-cutoff playback history")
    fallback: np.ndarray | None = None
    fallback_seconds = 0.0
    if config.family != "graded_per_rule_popularity":
        fallback_started = time.perf_counter()
        fallback, _ = _popularity_scores(history, candidates)
        fallback_seconds = time.perf_counter() - fallback_started
    training_seconds = 0.0
    scoring_seconds = 0.0
    artifact_payload: dict[str, Any] = {}
    losses: list[float] = []
    target = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    score_source = "model_rank_score"
    fusion_metrics: dict[str, Any] = {}
    if config.family == "graded_global_popularity":
        assert fallback is not None
        scores = fallback
        scoring_seconds = fallback_seconds
        target = torch.device("cpu")
        score_source = "global_watch_popularity"
    elif config.family == "graded_per_rule_popularity":
        if "retrieval_score" not in candidates:
            raise ValueError(
                "graded_per_rule_popularity requires serving retrieval_score; "
                "use graded_global_popularity for fixed candidates"
            )
        scoring_started = time.perf_counter()
        retrieval_scores = pd.to_numeric(
            candidates["retrieval_score"], errors="coerce"
        )
        if retrieval_scores.isna().any() or not np.isfinite(
            retrieval_scores.to_numpy(dtype=float)
        ).all():
            raise ValueError("serving retrieval_score must be finite")
        scores = retrieval_scores.to_numpy(dtype=np.float32)
        scoring_seconds = time.perf_counter() - scoring_started
        target = torch.device("cpu")
        score_source = "serving_personalized_retrieval_score"
    elif config.family == "session_item_knn":
        assert fallback is not None
        scoring_started = time.perf_counter()
        scores = _session_item_knn_scores(history, candidates, fallback)
        scoring_seconds = fallback_seconds + time.perf_counter() - scoring_started
        target = torch.device("cpu")
    elif config.family == "bpr_mf":
        assert fallback is not None
        training_started = time.perf_counter()
        model, user_map, item_map, losses = _train_bpr(edges, config, target)
        training_seconds = time.perf_counter() - training_started
        user_embeddings = model.user_embedding.weight.detach().cpu().numpy()
        item_embeddings = model.item_embedding.weight.detach().cpu().numpy()
        scoring_started = time.perf_counter()
        scores = _embedding_scores(
            candidates,
            user_map,
            item_map,
            user_embeddings,
            item_embeddings,
            fallback,
        )
        scoring_seconds = fallback_seconds + time.perf_counter() - scoring_started
        artifact_payload = {
            "model_state_dict": model.state_dict(),
            "user_map": user_map,
            "item_map": item_map,
        }
    elif config.family == "bpr_per_rule_fusion":
        assert fallback is not None
        training_started = time.perf_counter()
        model, user_map, item_map, losses = _train_bpr(edges, config, target)
        training_seconds = time.perf_counter() - training_started
        scoring_started = time.perf_counter()
        bpr_scores = _embedding_scores(
            candidates,
            user_map,
            item_map,
            model.user_embedding.weight.detach().cpu().numpy(),
            model.item_embedding.weight.detach().cpu().numpy(),
            fallback,
        )
        scores, fusion_metrics = _fused_per_rule_scores(
            candidates, bpr_scores, config
        )
        scoring_seconds = fallback_seconds + time.perf_counter() - scoring_started
        artifact_payload = {
            "model_state_dict": model.state_dict(),
            "user_map": user_map,
            "item_map": item_map,
        }
        score_source = "per_rule_retrieval_bpr_rank_rrf_fusion"
    elif config.family == "knn_per_rule_fusion":
        assert fallback is not None
        scoring_started = time.perf_counter()
        knn_scores = _session_item_knn_scores(history, candidates, fallback)
        scores, fusion_metrics = _fused_per_rule_scores(
            candidates, knn_scores, config
        )
        scoring_seconds = fallback_seconds + time.perf_counter() - scoring_started
        target = torch.device("cpu")
        score_source = "per_rule_retrieval_session_knn_rank_rrf_fusion"
    else:
        assert fallback is not None
        training_started = time.perf_counter()
        model, user_map, item_map, losses = _train_lightgcn(edges, config, target)
        training_seconds = time.perf_counter() - training_started
        with torch.no_grad():
            user_tensor, item_tensor = model.propagated()
        scoring_started = time.perf_counter()
        scores = _embedding_scores(
            candidates,
            user_map,
            item_map,
            user_tensor.detach().cpu().numpy(),
            item_tensor.detach().cpu().numpy(),
            fallback,
        )
        scoring_seconds = fallback_seconds + time.perf_counter() - scoring_started
        artifact_payload = {
            "model_state_dict": model.state_dict(),
            "user_map": user_map,
            "item_map": item_map,
        }
    artifact_path: Path | None = None
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        artifact_path = directory / f"{config.family}.pth"
        torch.save(
            {
                "schema_version": 1,
                "family": config.family,
                "config": asdict(config),
                "history_hash": _history_hash(edges),
                **artifact_payload,
            },
            artifact_path,
        )
    metrics = {
        "model_family": config.family,
        "score_source": score_source,
        "target_semantics": "implicit_engagement_ranking",
        "classification_metrics_valid": False,
        "history_events": int(len(history)),
        "history_pairs": int(len(edges)),
        "history_profiles": int(edges["profile_id"].nunique()),
        "history_contents": int(edges["content_id"].nunique()),
        "training_time_seconds": training_seconds,
        "scoring_time_seconds": scoring_seconds,
        "inference_pairs": int(len(candidates)),
        "candidate_input_columns": int(len(candidates.columns)),
        "candidate_frame_deep_copied": False,
        "inference_seconds_per_pair": scoring_seconds / max(len(candidates), 1),
        "end_to_end_seconds": time.perf_counter() - started,
        "device": str(target),
        "epochs_completed": len(losses),
        "training_loss_history": losses,
        "history_hash": _history_hash(edges),
        "unobserved_semantics": "sampled contrastive item, not a true negative",
    }
    if fusion_metrics:
        metrics["fusion_bpr_rank_coverage"] = float(
            fusion_metrics["content_external_rank_coverage"]
        )
        metrics["fusion_bpr_weight"] = float(config.fusion_bpr_weight)
        metrics["fusion_rrf_constant"] = int(config.fusion_rrf_constant)
    return ImplicitBaselineResult(
        scores=np.asarray(scores, dtype=np.float32),
        metrics=metrics,
        artifact_path=artifact_path,
    )
