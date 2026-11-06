from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_model_run_metrics(
    model_family: str,
    quality_metrics: dict[str, float],
    runtime_metrics: dict[str, float | int | str],
    rule_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "model_family": model_family,
        "quality": dict(quality_metrics),
        "runtime": dict(runtime_metrics),
        "rule": dict(rule_metrics or {}),
    }


def write_metrics_json(path: str | Path, metrics: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return output


def read_metrics_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_model_runs(
    runs: list[dict[str, Any]],
    primary_metric: str,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    if not runs:
        raise ValueError("compare_model_runs requires at least one run")

    def score(run: dict[str, Any]) -> float:
        quality = run.get("quality", {})
        if primary_metric not in quality:
            raise ValueError(f"Run {run.get('model_family', '<unknown>')} is missing {primary_metric}")
        return float(quality[primary_metric])

    best = max(runs, key=score) if higher_is_better else min(runs, key=score)
    return {
        "primary_metric": primary_metric,
        "higher_is_better": higher_is_better,
        "best_model_family": best["model_family"],
        "runs": list(runs),
    }
