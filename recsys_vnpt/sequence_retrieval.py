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
class SASRecLiteConfig:
    embedding_dim: int = 64
    heads: int = 2
    layers: int = 2
    maximum_sequence_length: int = 50
    sampled_items: int = 32
    epochs: int = 3
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    dropout: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        if min(
            self.embedding_dim,
            self.heads,
            self.layers,
            self.maximum_sequence_length,
            self.sampled_items,
            self.epochs,
            self.batch_size,
        ) < 1:
            raise ValueError("SASRec-lite dimensions must be positive")
        if self.embedding_dim % self.heads:
            raise ValueError("embedding_dim must be divisible by heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class SequenceRetrievalResult:
    rankings: dict[str, list[tuple[str, float]]]
    metrics: dict[str, Any]
    artifact_path: Path | None = None


class SASRecLite(torch.nn.Module):
    def __init__(self, item_count: int, config: SASRecLiteConfig) -> None:
        super().__init__()
        self.config = config
        self.item_embedding = torch.nn.Embedding(
            item_count + 1, config.embedding_dim, padding_idx=0
        )
        self.position_embedding = torch.nn.Embedding(
            config.maximum_sequence_length, config.embedding_dim
        )
        layer = torch.nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.heads,
            dim_feedforward=4 * config.embedding_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, config.layers)
        self.output_norm = torch.nn.LayerNorm(config.embedding_dim)
        torch.nn.init.normal_(self.item_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def encode(self, sequences: torch.Tensor) -> torch.Tensor:
        length = sequences.shape[1]
        positions = torch.arange(length, device=sequences.device).unsqueeze(0)
        values = self.item_embedding(sequences) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=sequences.device),
            diagonal=1,
        )
        encoded = self.encoder(
            values,
            mask=causal_mask,
            src_key_padding_mask=sequences.eq(0),
        )
        position_indices = torch.arange(length, device=sequences.device).unsqueeze(0)
        last_indices = torch.where(
            sequences.ne(0), position_indices, torch.zeros_like(position_indices)
        ).max(dim=1).values
        batch_indices = torch.arange(len(sequences), device=sequences.device)
        return F.normalize(
            self.output_norm(encoded[batch_indices, last_indices]), dim=-1
        )


def _collapse_consecutive(values: list[int]) -> list[int]:
    output: list[int] = []
    for value in values:
        if not output or output[-1] != value:
            output.append(value)
    return output


def _padded_prefixes(
    prefixes: list[list[int]],
    maximum_length: int,
) -> np.ndarray:
    output = np.zeros((len(prefixes), maximum_length), dtype=np.int64)
    for index, prefix in enumerate(prefixes):
        values = prefix[-maximum_length:]
        output[index, -len(values) :] = values
    return output


def _training_examples(
    history: pd.DataFrame,
    item_map: dict[str, int],
    maximum_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[int]], dict[str, set[int]], dict[str, int]]:
    prefixes: list[list[int]] = []
    targets: list[int] = []
    weights: list[float] = []
    profile_sequences: dict[str, list[int]] = {}
    consumed: dict[str, set[int]] = {}
    eligible_sessions = 0
    total_sessions = 0
    for session_id, group in history.groupby("session_id", sort=True):
        total_sessions += 1
        ordered = group.sort_values(["event_end", "content_id"], kind="stable")
        sequence = _collapse_consecutive(
            [item_map.get(str(value), 0) for value in ordered["content_id"]]
        )
        sequence = [value for value in sequence if value]
        if len(set(sequence)) < 2:
            continue
        eligible_sessions += 1
        watches = pd.to_numeric(
            ordered["watch_seconds"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        for index in range(1, len(sequence)):
            prefixes.append(sequence[:index])
            targets.append(sequence[index])
            watch_index = min(index, len(watches) - 1)
            weights.append(float(np.log1p(max(watches[watch_index], 0.0))))

    for profile_id, group in history.groupby("profile_id", sort=True):
        ordered = group.sort_values(["event_end", "content_id"], kind="stable")
        sequence = _collapse_consecutive(
            [item_map.get(str(value), 0) for value in ordered["content_id"]]
        )
        sequence = [value for value in sequence if value]
        profile_sequences[str(profile_id)] = sequence[-maximum_length:]
        consumed[str(profile_id)] = set(sequence)
    weight_array = np.asarray(weights, dtype=np.float32)
    if len(weight_array):
        weight_array /= max(float(np.quantile(weight_array, 0.95)), 1e-12)
        weight_array = np.clip(weight_array, 0.0, 1.0)
    return (
        _padded_prefixes(prefixes, maximum_length),
        np.asarray(targets, dtype=np.int64),
        weight_array,
        profile_sequences,
        consumed,
        {
            "total_sessions": total_sessions,
            "eligible_sequence_sessions": eligible_sessions,
        },
    )


def _sample_items(
    targets: np.ndarray,
    item_count: int,
    sample_count: int,
    probabilities: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.arange(1, item_count + 1)
    output = np.empty((len(targets), sample_count), dtype=np.int64)
    for row, target in enumerate(targets):
        selected: list[int] = []
        while len(selected) < sample_count:
            proposals = rng.choice(
                values,
                size=max(16, sample_count * 2),
                replace=True,
                p=probabilities,
            )
            selected.extend(int(value) for value in proposals if int(value) != int(target))
        output[row] = selected[:sample_count]
    return output


def train_and_rank_sasrec_lite(
    playback_events: pd.DataFrame,
    contents: pd.DataFrame,
    profiles: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    candidate_budget: int = 1000,
    config: SASRecLiteConfig = SASRecLiteConfig(),
    output_dir: str | Path | None = None,
    device: str | None = None,
) -> SequenceRetrievalResult:
    required = {
        "profile_id",
        "content_id",
        "session_id",
        "event_end",
        "watch_seconds",
    }
    if not required.issubset(playback_events):
        raise ValueError(
            f"SASRec-lite history is missing {sorted(required - set(playback_events))}"
        )
    items = contents.drop_duplicates("content_id", keep="last").copy()
    items["content_id"] = items["content_id"].astype(str)
    items = items.sort_values("content_id", kind="stable").reset_index(drop=True)
    item_map = {
        content_id: index + 1
        for index, content_id in enumerate(items["content_id"])
    }
    history = playback_events.copy()
    history["event_end"] = pd.to_datetime(history["event_end"], errors="coerce")
    history = history[history["event_end"].le(pd.Timestamp(cutoff))].copy()
    history["profile_id"] = history["profile_id"].astype(str)
    history["content_id"] = history["content_id"].astype(str)
    history = history[history["content_id"].isin(item_map)]
    (
        sequences,
        targets,
        weights,
        profile_sequences,
        consumed,
        session_metrics,
    ) = _training_examples(history, item_map, config.maximum_sequence_length)
    if len(targets) == 0:
        return SequenceRetrievalResult(
            rankings={},
            metrics={
                "model_family": "sasrec_lite",
                "target_semantics": "next observed item ranking",
                "classification_metrics_valid": False,
                "training_examples": 0,
                "profiles_with_sequence_rankings": 0,
                **session_metrics,
            },
        )

    target_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(config.seed)
    model = SASRecLite(len(item_map), config).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    frequency = np.bincount(targets, minlength=len(item_map) + 1)[1:].astype(float)
    probabilities = np.power(frequency + 1e-12, 0.75)
    probabilities /= probabilities.sum()
    correction_values = np.concatenate([[0.0], probabilities])
    rng = np.random.default_rng(config.seed)
    losses: list[float] = []
    started = time.perf_counter()
    for _ in range(config.epochs):
        permutation = rng.permutation(len(targets))
        running = 0.0
        batches = 0
        model.train()
        for start in range(0, len(permutation), config.batch_size):
            batch = permutation[start : start + config.batch_size]
            positives = targets[batch]
            negatives = _sample_items(
                positives,
                len(item_map),
                config.sampled_items,
                probabilities,
                rng,
            )
            sampled = np.column_stack([positives, negatives])
            vectors = model.encode(
                torch.as_tensor(sequences[batch], device=target_device)
            )
            item_vectors = F.normalize(
                model.item_embedding(torch.as_tensor(sampled, device=target_device)),
                dim=-1,
            )
            logits = torch.einsum("bd,bkd->bk", vectors, item_vectors)
            correction = torch.as_tensor(
                np.log(np.clip(correction_values[sampled], 1e-12, None)),
                dtype=logits.dtype,
                device=target_device,
            )
            per_row = F.cross_entropy(
                logits - correction,
                torch.zeros(len(batch), dtype=torch.long, device=target_device),
                reduction="none",
            )
            batch_weights = torch.as_tensor(weights[batch], device=target_device)
            loss = (per_row * batch_weights).sum() / batch_weights.sum().clamp_min(1e-12)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach().cpu())
            batches += 1
        losses.append(running / max(batches, 1))
    training_seconds = time.perf_counter() - started

    requested = profiles["profile_id"].dropna().astype(str).drop_duplicates().tolist()
    eligible_profiles = [
        profile_id
        for profile_id in requested
        if len(set(profile_sequences.get(profile_id, []))) >= 2
    ]
    rankings: dict[str, list[tuple[str, float]]] = {}
    ranking_started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        all_items = F.normalize(model.item_embedding.weight[1:], dim=-1)
        content_ids = items["content_id"].astype(str).tolist()
        for start in range(0, len(eligible_profiles), 256):
            profile_batch = eligible_profiles[start : start + 256]
            batch_sequences = _padded_prefixes(
                [profile_sequences[profile_id] for profile_id in profile_batch],
                config.maximum_sequence_length,
            )
            vectors = model.encode(
                torch.as_tensor(batch_sequences, device=target_device)
            )
            scores = vectors @ all_items.T
            for row, profile_id in enumerate(profile_batch):
                blocked = consumed.get(profile_id, set())
                if blocked:
                    scores[row, torch.as_tensor(
                        [item_id - 1 for item_id in blocked], device=target_device
                    )] = -torch.inf
                k = min(candidate_budget, len(content_ids) - len(blocked))
                if k <= 0:
                    rankings[profile_id] = []
                    continue
                values, indices = torch.topk(scores[row], k=k)
                rankings[profile_id] = [
                    (content_ids[int(index)], float(value))
                    for value, index in zip(
                        values.cpu().tolist(), indices.cpu().tolist()
                    )
                ]
    ranking_seconds = time.perf_counter() - ranking_started

    history_hash = hashlib.sha256(
        pd.util.hash_pandas_object(
            history[["profile_id", "content_id", "session_id", "event_end"]],
            index=False,
            categorize=True,
        ).to_numpy().tobytes()
    ).hexdigest()
    artifact_path: Path | None = None
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        artifact_path = directory / "sasrec_lite_v1.pth"
        torch.save(
            {
                "schema_version": 1,
                "config": asdict(config),
                "history_hash": history_hash,
                "item_map": item_map,
                "model_state_dict": model.state_dict(),
            },
            artifact_path,
        )
        (directory / "sasrec_lite_v1.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "config": asdict(config),
                    "history_hash": history_hash,
                    "minimum_distinct_session_contents": 2,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    metrics = {
        "model_family": "sasrec_lite",
        "target_semantics": "next_observed_item_ranking",
        "classification_metrics_valid": False,
        "training_examples": int(len(targets)),
        "training_time_seconds": training_seconds,
        "ranking_time_seconds": ranking_seconds,
        "epochs_completed": len(losses),
        "training_loss_history": losses,
        "requested_profiles": int(len(requested)),
        "profiles_with_sequence_rankings": int(len(rankings)),
        "sequence_profile_coverage": len(rankings) / max(len(requested), 1),
        "history_hash": history_hash,
        "unobserved_semantics": "contrastive sample, not a true negative",
        **session_metrics,
    }
    return SequenceRetrievalResult(rankings, metrics, artifact_path)
