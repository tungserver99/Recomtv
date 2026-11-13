from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recsys_vnpt.batch_contracts import BatchConfig, load_config
from recsys_vnpt.runtime import atomic_write_text

from production.cutoff import validate_run_times


@dataclass(frozen=True)
class ProductionConfig:
    config_path: Path
    project_root: Path
    schema_version: int
    timezone: str
    schedule_enabled: bool
    run_times: tuple[str, ...]
    retrain_run_times: tuple[str, ...]
    inference_run_times: tuple[str, ...]
    data_root: Path
    batch_values: dict[str, Any]

    @property
    def source_root(self) -> Path:
        """Backward-compatible alias used by the existing staging code."""
        return self.data_root

    def resolve_data_root(self, value: str | Path) -> Path:
        candidate = _resolve_path(str(value), self.project_root)
        if not candidate.is_dir():
            raise ValueError(f"data_root must be an existing directory: {candidate}")
        roots = [
            self.incoming_root,
            self.work_root,
            self.delivery_root,
            self.model_registry,
            self.state_root,
        ]
        for root in roots:
            if _overlap(candidate, root):
                raise ValueError(f"data_root must not overlap production storage: {candidate} and {root}")
        return candidate

    @property
    def batch(self) -> BatchConfig:
        runtime_path = self.state_root / f"runtime_batch_config.{self.config_path.stem}.json"
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(runtime_path, json.dumps(self.batch_values, indent=2, sort_keys=True))
        return load_config(runtime_path)

    @property
    def incoming_root(self) -> Path:
        return Path(self.batch_values["incoming_root"])

    @property
    def work_root(self) -> Path:
        return Path(self.batch_values["work_root"])

    @property
    def delivery_root(self) -> Path:
        return Path(self.batch_values["delivery_root"])

    @property
    def model_registry(self) -> Path:
        return Path(self.batch_values["model_registry"])

    @property
    def state_root(self) -> Path:
        return self.work_root.parent / "state"

    @property
    def database_path(self) -> Path:
        return self.state_root / "production.db"


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


def _overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def load_production_config(path: str | Path) -> ProductionConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read production config {config_path}: {error}") from error
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("production config schema_version must be 1")
    project_root = config_path.parent.parent.resolve()
    schedule = raw.get("schedule")
    batch = raw.get("batch")
    if not isinstance(schedule, dict) or not isinstance(batch, dict):
        raise ValueError("production config requires schedule and batch objects")
    timezone = str(raw.get("timezone", ""))
    # ZoneInfo validation lives with the scheduling functions.
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise ValueError(f"unknown timezone {timezone!r}") from error
    try:
        run_times = validate_run_times(schedule.get("run_times"))
    except (TypeError, ValueError) as error:
        raise ValueError(str(error)) from error
    try:
        retrain_values = schedule.get("retrain_run_times", ())
        retrain_run_times = validate_run_times(retrain_values) if retrain_values else ()
        inference_values = schedule.get("inference_run_times", run_times)
        inference_run_times = validate_run_times(inference_values) if inference_values else ()
    except (TypeError, ValueError) as error:
        raise ValueError(str(error)) from error
    configured_data_root = raw.get("data_root", raw.get("source_root"))
    if not isinstance(configured_data_root, str) or not configured_data_root.strip():
        raise ValueError("production config requires a non-empty data_root")
    data_root = _resolve_path(configured_data_root, project_root)
    resolved_batch = dict(batch)
    path_keys = ("incoming_root", "work_root", "delivery_root", "model_registry")
    for key in path_keys:
        if key not in resolved_batch:
            raise ValueError(f"batch config is missing {key}")
        resolved_batch[key] = str(_resolve_path(str(resolved_batch[key]), project_root))
    resolved_batch["combination"] = "GxEp"
    state_root = Path(resolved_batch["work_root"]).parent / "state"
    roots = [data_root, *(Path(resolved_batch[key]) for key in path_keys), state_root]
    for index, root in enumerate(roots):
        for other in roots[index + 1 :]:
            if _overlap(root, other):
                raise ValueError(f"production roots must not overlap: {root} and {other}")
    # Validate against the existing batch schema now, not only when a run starts.
    temp = state_root / f"runtime_batch_config.{config_path.stem}.json"
    temp.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(temp, json.dumps(resolved_batch, indent=2, sort_keys=True))
    load_config(temp)
    return ProductionConfig(
        config_path=config_path,
        project_root=project_root,
        schema_version=1,
        timezone=timezone,
        schedule_enabled=bool(schedule.get("enabled", False)),
        run_times=run_times,
        retrain_run_times=retrain_run_times,
        inference_run_times=inference_run_times,
        data_root=data_root,
        batch_values=resolved_batch,
    )

