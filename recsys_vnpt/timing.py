from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


def measure_seconds(func: Callable[[], T], clock: Callable[[], float] | None = None) -> tuple[T, float]:
    timer = time.perf_counter if clock is None else clock
    start = timer()
    result = func()
    elapsed = timer() - start
    _require_nonnegative("elapsed_seconds", elapsed)
    return result, elapsed


def runtime_metrics(
    model_family: str,
    training_time_seconds: float | None = None,
    inference_time_seconds: float | None = None,
    inference_pairs: int | None = None,
) -> dict[str, float | int | str]:
    metrics: dict[str, float | int | str] = {"model_family": model_family}
    if training_time_seconds is not None:
        _require_nonnegative("training_time_seconds", training_time_seconds)
        metrics["training_time_seconds"] = float(training_time_seconds)
    if inference_time_seconds is not None:
        _require_nonnegative("inference_time_seconds", inference_time_seconds)
        metrics["inference_time_seconds"] = float(inference_time_seconds)
    if inference_pairs is not None:
        if inference_pairs < 0:
            raise ValueError("inference_pairs must be nonnegative")
        metrics["inference_pairs"] = int(inference_pairs)
        if inference_time_seconds is not None and inference_pairs > 0:
            metrics["inference_seconds_per_pair"] = float(inference_time_seconds) / inference_pairs
    return metrics


def _require_nonnegative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
