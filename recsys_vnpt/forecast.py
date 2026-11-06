from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class RuntimeObservation:
    profiles: int
    scored_pairs: int
    seconds: float
    hardware: str = "unknown"
    run_id: str = "unknown"

    def __post_init__(self) -> None:
        if self.profiles < 1 or self.scored_pairs < 1 or self.seconds <= 0:
            raise ValueError("runtime observations require positive profiles, pairs, and seconds")


def observation_from_metrics(
    run_id: str,
    inference: dict[str, Any],
    hardware: str,
) -> RuntimeObservation:
    return RuntimeObservation(
        profiles=int(inference["profiles_scored"]),
        scored_pairs=int(inference["inference_pairs"]),
        seconds=float(inference.get("p50_seconds", inference["inference_time_seconds"]))
        + float(inference.get("serialization_seconds", 0.0)),
        hardware=hardware,
        run_id=run_id,
    )


def _fit(observations: list[RuntimeObservation]) -> tuple[np.ndarray, str, int]:
    full = np.asarray(
        [[1.0, row.scored_pairs, row.profiles] for row in observations], dtype=float
    )
    seconds = np.asarray([row.seconds for row in observations], dtype=float)
    rank = int(np.linalg.matrix_rank(full))
    if rank == 3:
        coefficients, *_ = np.linalg.lstsq(full, seconds, rcond=None)
        return coefficients, "pairs_and_profiles", rank
    reduced = np.asarray([[1.0, row.profiles] for row in observations], dtype=float)
    reduced_rank = int(np.linalg.matrix_rank(reduced))
    if reduced_rank < 2:
        raise ValueError("workloads do not vary enough to fit a runtime forecast")
    coefficients, *_ = np.linalg.lstsq(reduced, seconds, rcond=None)
    return coefficients, "constant_candidates_per_profile", rank


def _predict(
    coefficients: np.ndarray,
    model_type: str,
    profiles: int,
    pairs: int,
) -> float:
    if model_type == "pairs_and_profiles":
        value = coefficients[0] + coefficients[1] * pairs + coefficients[2] * profiles
    else:
        value = coefficients[0] + coefficients[1] * profiles
    return max(float(value), 0.0)


def fit_runtime_forecast(
    observations: Iterable[RuntimeObservation],
    target_profiles: Iterable[int] = (1_000_000, 2_000_000),
    pairs_per_profile: int | None = None,
) -> dict[str, Any]:
    rows = sorted(list(observations), key=lambda row: (row.profiles, row.scored_pairs))
    targets = list(target_profiles)
    if len(rows) < 3:
        raise ValueError("at least three workloads are required for train/holdout forecasting")
    hardware = {row.hardware for row in rows}
    if len(hardware) != 1:
        raise ValueError("fit each hardware type separately; mixed hardware observations are invalid")
    train, holdout = rows[:-1], rows[-1]
    coefficients, model_type, design_rank = _fit(train)
    holdout_prediction = _predict(
        coefficients, model_type, holdout.profiles, holdout.scored_pairs
    )
    holdout_error = abs(holdout_prediction - holdout.seconds) / holdout.seconds

    fitted = [
        _predict(coefficients, model_type, row.profiles, row.scored_pairs) for row in train
    ]
    residuals = np.asarray([row.seconds - value for row, value in zip(train, fitted)])
    parameter_count = len(coefficients)
    uncertainty_available = len(train) > parameter_count
    residual_std = (
        float(np.sqrt(np.sum(np.square(residuals)) / (len(train) - parameter_count)))
        if uncertainty_available
        else None
    )
    design = np.asarray(
        (
            [[1.0, row.scored_pairs, row.profiles] for row in train]
            if model_type == "pairs_and_profiles"
            else [[1.0, row.profiles] for row in train]
        ),
        dtype=float,
    )
    information_inverse = np.linalg.pinv(design.T @ design)
    pair_ratio = pairs_per_profile
    if pair_ratio is None:
        ratios = [row.scored_pairs / row.profiles for row in rows]
        if max(ratios) - min(ratios) > 1e-9 and model_type != "pairs_and_profiles":
            raise ValueError("pairs_per_profile is required when candidate counts vary")
        pair_ratio = int(round(float(np.median(ratios))))

    forecasts: dict[str, Any] = {}
    maximum_observed_profiles = max(row.profiles for row in rows)
    for profiles in targets:
        if profiles < 1:
            raise ValueError("target profile counts must be positive")
        pairs = int(profiles * pair_ratio)
        median = _predict(coefficients, model_type, profiles, pairs)
        target_design = np.asarray(
            [1.0, pairs, profiles]
            if model_type == "pairs_and_profiles"
            else [1.0, profiles],
            dtype=float,
        )
        prediction_std = (
            residual_std
            * float(
                np.sqrt(
                    1.0
                    + target_design @ information_inverse @ target_design
                )
            )
            if residual_std is not None
            else None
        )
        margin = 1.96 * prediction_std if prediction_std is not None else None
        forecasts[str(profiles)] = {
            "profiles": int(profiles),
            "scored_pairs": pairs,
            "median_seconds": median,
            "median_hours": median / 3600,
            "uncertainty_low_seconds": max(0.0, median - margin) if margin is not None else None,
            "uncertainty_high_seconds": median + margin if margin is not None else None,
            "prediction_std_seconds": prediction_std,
            "extrapolation_ratio": profiles / maximum_observed_profiles,
            "within_measured_ramp": profiles <= maximum_observed_profiles,
        }
    hardware_name = next(iter(hardware))
    all_targets_measured = all(int(target) <= maximum_observed_profiles for target in targets)
    measured_on_a100 = "A100" in hardware_name.upper()
    return {
        "hardware": hardware_name,
        "model_type": model_type,
        "design_rank": design_rank,
        "coefficients": coefficients.tolist(),
        "pairs_per_profile": pair_ratio,
        "maximum_observed_profiles": maximum_observed_profiles,
        "timing_scope": "steady_state_inference_plus_serialization; excludes one-time load/retrieval",
        "holdout": {
            "actual_seconds": holdout.seconds,
            "predicted_seconds": holdout_prediction,
            "forecast_error_rate": holdout_error,
            "gate_passed": holdout_error <= 0.10,
        },
        "residual_std_seconds": residual_std,
        "uncertainty_available": uncertainty_available,
        "uncertainty_scope": "linear-fit residual only; excludes hardware, data-volume, and pipeline drift",
        "forecasts": forecasts,
        "observations": [asdict(row) for row in rows],
        "a100_projection_status": (
            "measured_on_a100" if measured_on_a100 else "requires_a100_shadow_observations"
        ),
        "production_projection_qualified": bool(
            holdout_error <= 0.10 and measured_on_a100 and all_targets_measured
        ),
    }
