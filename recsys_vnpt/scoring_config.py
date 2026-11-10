"""Validated, backward-compatible content scorer composition settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


CONTENT_BACKENDS = ("heuristic", "gbdt", "fcn")
RANK_SCOPES = ("per_rule", "profile")
PRESETS = {"legacy": "legacy", "gxgp": "GxGp", "gxep": "GxEp", "custom": "custom"}
DEFAULT_WEIGHTS = {"heuristic": 1.0, "gbdt": 5.0, "fcn": 1.0}


@dataclass(frozen=True)
class ScoringConfig:
    """The content ranking components and the rank domain used by their RRF."""

    preset: str
    content_backends: tuple[str, ...]
    rank_scope: str
    weights: dict[str, float]
    rrf_constant: float = 60.0

    @property
    def uses_gbdt(self) -> bool:
        return "gbdt" in self.content_backends

    @property
    def uses_fcn(self) -> bool:
        return "fcn" in self.content_backends


def scoring_config_payload(config: ScoringConfig) -> dict[str, object]:
    """JSON-safe identity stored in bundles, metrics, and delivery manifests."""
    payload = asdict(config)
    payload["content_backends"] = list(config.content_backends)
    return payload


def resolve_scoring_config(
    preset: str = "legacy",
    content_backends: Sequence[str] | None = None,
    rank_scope: str | None = None,
    weights: Mapping[str, float] | None = None,
    legacy_content_ordering: str = "learned_rrf",
    legacy_fusion_weight: float = 5.0,
    rrf_constant: float = 60.0,
) -> ScoringConfig:
    """Resolve a preset or explicit composition while retaining legacy defaults.

    GxGp/GxEp names describe component selections only. Their weights are
    configurable defaults pending verification against the original benchmark.
    """
    normalized_preset = PRESETS.get(str(preset).lower())
    if normalized_preset is None:
        raise ValueError(f"unsupported scoring preset: {preset}")

    if content_backends is None:
        if normalized_preset == "legacy":
            if legacy_content_ordering not in {"hand_sum", "learned_rrf"}:
                raise ValueError(
                    "legacy content ordering must be hand_sum or learned_rrf"
                )
            selected = (
                ("heuristic", "gbdt")
                if legacy_content_ordering == "learned_rrf"
                else ("heuristic",)
            )
        elif normalized_preset == "GxGp":
            selected = ("heuristic", "gbdt")
        elif normalized_preset == "GxEp":
            selected = ("heuristic", "gbdt", "fcn")
        else:
            raise ValueError("custom preset requires explicit content_backends")
    else:
        selected = tuple(dict.fromkeys(map(str, content_backends)))

    if not selected:
        raise ValueError("at least one content backend must be enabled")
    unknown = sorted(set(selected) - set(CONTENT_BACKENDS))
    if unknown:
        raise ValueError(f"unsupported content backend(s): {unknown}")

    if rank_scope is None:
        scope = "profile" if normalized_preset in {"GxGp", "GxEp"} else "per_rule"
    else:
        scope = str(rank_scope)
    if scope not in RANK_SCOPES:
        raise ValueError(f"unsupported content rank scope: {scope}")

    resolved_weights = {**DEFAULT_WEIGHTS, **dict(weights or {})}
    unknown_weights = sorted(set(resolved_weights) - set(CONTENT_BACKENDS))
    if unknown_weights:
        raise ValueError(f"unsupported content fusion weight(s): {unknown_weights}")
    if weights is not None:
        unknown_weights = sorted(set(weights) - set(CONTENT_BACKENDS))
        if unknown_weights:
            raise ValueError(f"unsupported content fusion weight(s): {unknown_weights}")
    if normalized_preset == "legacy" and weights is None:
        resolved_weights["gbdt"] = float(legacy_fusion_weight)
    for backend in selected:
        value = float(resolved_weights[backend])
        if value <= 0:
            raise ValueError(f"active content fusion weight for {backend} must be positive")
        resolved_weights[backend] = value
    if rrf_constant <= 0:
        raise ValueError("RRF constant must be positive")

    return ScoringConfig(
        preset=normalized_preset,
        content_backends=selected,
        rank_scope=scope,
        weights=resolved_weights,
        rrf_constant=float(rrf_constant),
    )
