"""Training objectives.

The graded targets need more than a plain BCE: rules and contents are scored in
groups, so the listwise and pairwise forms here operate per group rather than per
row.

`TrainingConfig` is only read for its three loss weights and is imported under
`TYPE_CHECKING`, so this module does not close a cycle with `model_runner`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from recsys_vnpt.model_runner import TrainingConfig


# Families whose heads are trained with the Tri-BCE objective.
TRI_BCE_FAMILIES = {
    "embedding_dcnv3",
    "legacy_embedding_dcnv3",
    "fcn_official",
    "ecn_official",
    "hierarchical_fcn",
}


def _weighted_bce(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor | None,
) -> torch.Tensor:
    if not torch.isfinite(predictions).all():
        raise FloatingPointError(
            "model produced non-finite probabilities; use BF16/FP32 or lower learning rate"
        )
    safe_predictions = predictions.clamp(1e-7, 1 - 1e-7)
    losses = torch.nn.functional.binary_cross_entropy(
        safe_predictions, labels, reduction="none"
    )
    if sample_weights is None:
        return losses.mean()
    weights = sample_weights.float().clamp_min(0)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-12)


def _grouped_listwise_loss(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    group_values: list[tuple[str, ...]],
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = torch.logit(probabilities.clamp(1e-6, 1 - 1e-6))
    grouped: dict[tuple[str, ...], list[int]] = {}
    for index, key in enumerate(group_values):
        grouped.setdefault(key, []).append(index)
    losses: list[torch.Tensor] = []
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        index_tensor = torch.tensor(indices, dtype=torch.long, device=logits.device)
        targets = labels[index_tensor]
        if sample_weights is not None:
            targets = targets * sample_weights[index_tensor].clamp_min(0.0)
        positive_mass = targets.sum()
        if positive_mass <= 0:
            continue
        target_distribution = targets / positive_mass
        losses.append(
            -(target_distribution * torch.log_softmax(logits[index_tensor], dim=0)).sum()
        )
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def _rule_listwise_loss(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    profiles: list[str],
    rules: list[str],
    windows: list[str] | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = torch.logit(probabilities.clamp(1e-6, 1 - 1e-6))
    temporal_windows = windows or ["__NO_WINDOW__"] * len(profiles)
    by_context: dict[tuple[str, str], dict[str, list[int]]] = {}
    for index, (window, profile, rule) in enumerate(
        zip(temporal_windows, profiles, rules)
    ):
        by_context.setdefault((window, profile), {}).setdefault(rule, []).append(
            index
        )
    losses: list[torch.Tensor] = []
    for rule_rows in by_context.values():
        if len(rule_rows) < 2:
            continue
        rule_logits: list[torch.Tensor] = []
        rule_targets: list[torch.Tensor] = []
        for indices in rule_rows.values():
            index_tensor = torch.tensor(
                indices, dtype=torch.long, device=logits.device
            )
            if sample_weights is None:
                row_weights = torch.ones_like(logits[index_tensor])
            else:
                row_weights = sample_weights[index_tensor].clamp_min(0.0)
            denominator = row_weights.sum().clamp_min(1e-12)
            rule_logits.append(
                (logits[index_tensor] * row_weights).sum() / denominator
            )
            rule_targets.append(labels[index_tensor].sum())
        stacked_targets = torch.stack(rule_targets)
        if stacked_targets.sum() <= 0:
            continue
        target_distribution = stacked_targets / stacked_targets.sum()
        losses.append(
            -(target_distribution * torch.log_softmax(torch.stack(rule_logits), dim=0)).sum()
        )
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def _grouped_pairwise_loss(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    group_values: list[tuple[str, ...]],
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pair rows with unequal gain inside a serving-equivalent group."""
    logits = torch.logit(probabilities.clamp(1e-6, 1 - 1e-6))
    grouped: dict[tuple[str, ...], list[int]] = {}
    for index, key in enumerate(group_values):
        grouped.setdefault(key, []).append(index)
    losses: list[torch.Tensor] = []
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        index_tensor = torch.tensor(indices, dtype=torch.long, device=logits.device)
        group_logits = logits[index_tensor]
        group_targets = labels[index_tensor]
        left, right = torch.triu_indices(
            len(indices), len(indices), offset=1, device=logits.device
        )
        target_delta = group_targets[left] - group_targets[right]
        unequal = target_delta.abs().gt(1e-8)
        if not unequal.any():
            continue
        logit_delta = group_logits[left] - group_logits[right]
        pair_loss = torch.nn.functional.softplus(
            -target_delta[unequal].sign() * logit_delta[unequal]
        ) * target_delta[unequal].abs()
        if sample_weights is not None:
            group_weights = sample_weights[index_tensor].clamp_min(0.0)
            pair_weights = torch.sqrt(
                group_weights[left][unequal] * group_weights[right][unequal]
            )
            pair_loss = pair_loss * pair_weights
        losses.append(pair_loss.mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def _single_target_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    family: str,
    sample_weights: torch.Tensor | None,
    batch_frame: pd.DataFrame | None,
    config: TrainingConfig | None,
) -> torch.Tensor:
    fused_bce = _weighted_bce(outputs["y_pred"], labels, sample_weights)
    tri_bce = fused_bce
    if family in TRI_BCE_FAMILIES:
        deep = _weighted_bce(outputs["y_d"], labels, sample_weights)
        shallow = _weighted_bce(outputs["y_s"], labels, sample_weights)
        deep_weight = torch.relu(deep - fused_bce)
        shallow_weight = torch.relu(shallow - fused_bce)
        tri_bce = fused_bce + deep * deep_weight + shallow * shallow_weight
    if family != "hierarchical_fcn" or batch_frame is None or config is None:
        return tri_bce

    profiles = batch_frame["profile_id"].astype(str).tolist()
    windows = (
        batch_frame["temporal_window"].astype(str).tolist()
        if "temporal_window" in batch_frame
        else ["__NO_WINDOW__"] * len(batch_frame)
    )
    rules = (
        batch_frame["rule_id"].astype(str).tolist()
        if "rule_id" in batch_frame
        else ["__NO_RULE__"] * len(batch_frame)
    )
    content_groups = list(zip(windows, profiles, rules))
    content_probabilities = outputs.get("content_score", outputs["y_pred"])
    content_loss = _grouped_listwise_loss(
        content_probabilities,
        labels,
        content_groups,
        sample_weights=sample_weights,
    )
    pairwise_loss = _grouped_pairwise_loss(
        content_probabilities,
        labels,
        content_groups,
        sample_weights=sample_weights,
    )
    if "rule_id" in batch_frame:
        rule_loss = _rule_listwise_loss(
            outputs["rule_score"],
            labels,
            profiles,
            rules,
            windows=windows,
            sample_weights=sample_weights,
        )
    else:
        rule_loss = outputs["rule_score"].sum() * 0.0
    return (
        config.rule_loss_weight * rule_loss
        + config.content_loss_weight * content_loss
        + config.auxiliary_loss_weight * pairwise_loss
    )


def _loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    family: str,
    sample_weights: torch.Tensor | None = None,
    batch_frame: pd.DataFrame | None = None,
    config: TrainingConfig | None = None,
) -> torch.Tensor:
    if labels.ndim == 1:
        return _single_target_loss(
            outputs,
            labels,
            family,
            sample_weights,
            batch_frame,
            config,
        )
    if labels.ndim != 2 or labels.shape[1] < 1:
        raise ValueError("labels must have shape [rows] or [rows, targets]")
    losses = [
        _single_target_loss(
            outputs,
            labels[:, index],
            family,
            sample_weights,
            batch_frame,
            config,
        )
        for index in range(labels.shape[1])
    ]
    return torch.stack(losses).mean()


def _soft_target_log_loss(
    targets: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    truth = np.asarray(targets, dtype=np.float64)
    predictions = np.clip(np.asarray(scores, dtype=np.float64), 1e-7, 1 - 1e-7)
    losses = -(
        truth * np.log(predictions) + (1.0 - truth) * np.log(1.0 - predictions)
    )
    if weights is None:
        return float(losses.mean()) if losses.size else 0.0
    normalized_weights = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    denominator = float(normalized_weights.sum())
    return (
        float(np.sum(losses * normalized_weights) / denominator)
        if denominator > 0
        else 0.0
    )
