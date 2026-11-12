"""Versioned manifests for canonical and temporal batch preparation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from recsys_vnpt.batch_contracts import BatchConfig, BatchError, ReadyManifest
from recsys_vnpt.runtime import atomic_write_text
from recsys_vnpt.source_manifest import stable_hash


PREPARED_DATASET_SCHEMA_VERSION = 1
PREPARATION_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class PreparedDatasetPaths:
    artifact_root: Path
    manifest_path: Path
    config_hash: str
    artifact_key: str


@dataclass(frozen=True)
class PreparedDataset:
    artifact_root: Path
    manifest_path: Path
    manifest: dict[str, Any]


def preparation_config(config: BatchConfig, batch_started_at: datetime) -> dict[str, Any]:
    """Only settings that change prepared canonical or temporal artifacts."""
    return {
        "contract_version": PREPARATION_CONTRACT_VERSION,
        "domain": config.domain,
        "include_behavior": config.include_behavior,
        "user_reference_date": batch_started_at.date().isoformat(),
        "window_mode": "relative",
        "time_train": config.time_train,
        "time_valid": config.time_valid,
        "time_test": config.time_test,
        "sequence_mode": "historical",
    }


def prepared_dataset_paths(config: BatchConfig, input_manifest: ReadyManifest) -> PreparedDatasetPaths:
    prepare_config = preparation_config(config, input_manifest.batch_started_at)
    config_hash = stable_hash(prepare_config)
    artifact_key = stable_hash(
        {
            "input_fingerprint": input_manifest.fingerprint,
            "preparation_config_hash": config_hash,
        }
    )
    work_root = config.root("work_root")
    return PreparedDatasetPaths(
        # Keep the on-disk path short for Windows/MAX_PATH; the manifest still
        # records and validates the full 256-bit artifact key.
        artifact_root=work_root / "prepared_artifacts" / artifact_key[:24],
        manifest_path=(
            work_root
            / "prepared_datasets"
            / input_manifest.batch_id
            / f"{artifact_key[:32]}.json"
        ),
        config_hash=config_hash,
        artifact_key=artifact_key,
    )


def _required_artifacts(artifact_root: Path, domain: str) -> list[Path]:
    canonical = artifact_root / "data" / "data" / "canonical" / domain
    temporal = canonical / "temporal"
    return [
        canonical / "users.parquet",
        canonical / "interactions.parquet",
        canonical / "profile_mapping.parquet",
        canonical / "contents.parquet",
        canonical / "content_versions.parquet",
        canonical / "content_rule_membership.parquet",
        canonical / "playback_events.parquet",
        canonical / "label_contract_v4.json",
        temporal / "windows.json",
        temporal / "train_1.parquet",
        temporal / "train_2.parquet",
        temporal / "train_3.parquet",
        temporal / "validation.parquet",
        temporal / "test.parquet",
    ]


def _artifact_inventory(artifact_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(artifact_root).as_posix(),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(artifact_root.rglob("*"))
        if path.is_file()
    ]


def _has_playback_date_partitions(artifact_root: Path) -> bool:
    canonical = artifact_root / "data" / "data" / "canonical"
    return any(
        canonical.glob("*/playback_events_by_date/event_date=*/part-*.parquet")
    )


def write_prepared_dataset_manifest(
    config: BatchConfig,
    input_manifest: ReadyManifest,
    *,
    status: str,
    timings: dict[str, float] | None = None,
    error: str | None = None,
) -> Path:
    """Atomically record progress or completion for this exact input/config pair."""
    if status not in {"PREPARING", "READY", "FAILED"}:
        raise ValueError(f"unsupported prepared dataset status: {status}")
    paths = prepared_dataset_paths(config, input_manifest)
    payload: dict[str, Any] = {
        "schema_version": PREPARED_DATASET_SCHEMA_VERSION,
        "status": status,
        "batch_id": input_manifest.batch_id,
        "batch_started_at": input_manifest.batch_started_at.isoformat(),
        "input_fingerprint": input_manifest.fingerprint,
        "preparation_config": preparation_config(config, input_manifest.batch_started_at),
        "preparation_config_hash": paths.config_hash,
        "artifact_key": paths.artifact_key,
        "artifact_root": paths.artifact_root.relative_to(config.root("work_root")).as_posix(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "timings_seconds": timings or {},
    }
    if error:
        payload["error"] = error
    if status == "READY":
        payload["prepared_at"] = datetime.now(timezone.utc).isoformat()
        missing = [str(path) for path in _required_artifacts(paths.artifact_root, config.domain) if not path.exists()]
        if missing:
            raise BatchError(
                "PREPARED_DATA_UNAVAILABLE",
                f"preparation did not produce required artifacts: {missing[:5]}",
            )
        if not _has_playback_date_partitions(paths.artifact_root):
            raise BatchError(
                "PREPARED_DATA_UNAVAILABLE",
                "preparation did not produce playback event date partitions",
            )
        windows_path = paths.artifact_root / "data" / "data" / "canonical" / config.domain / "temporal" / "windows.json"
        try:
            windows_payload = json.loads(windows_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BatchError("PREPARED_DATA_UNAVAILABLE", f"cannot read temporal windows: {error}") from error
        payload["temporal_windows"] = windows_payload.get("windows", [])
        payload["artifacts"] = _artifact_inventory(paths.artifact_root)
        input_root = (config.root("incoming_root") / input_manifest.batch_id).resolve()
        input_files = []
        for row in input_manifest.files:
            source = (input_root / Path(*PurePosixPath(row["path"]).parts)).resolve(strict=True)
            if input_root not in source.parents or not source.is_file():
                raise BatchError("INPUT_INTEGRITY", f"invalid prepared input path: {row['path']}")
            stat = source.stat()
            input_files.append(
                {
                    "path": row["path"],
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        payload["input_files"] = input_files
    paths.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(paths.manifest_path, json.dumps(payload, indent=2, sort_keys=True, default=str))
    return paths.manifest_path


def _unavailable(config: BatchConfig, batch_id: str, reason: str) -> BatchError:
    return BatchError(
        "PREPARED_DATA_UNAVAILABLE",
        f"{reason} for batch {batch_id}. Run `python -m recsys_vnpt.batch_cli prepare-data "
        f"--config <same-config.json> --batch-id {batch_id}` first.",
    )


def load_prepared_dataset(config: BatchConfig, input_manifest: ReadyManifest) -> PreparedDataset:
    """Resolve only the prepared version pinned to this batch and config."""
    paths = prepared_dataset_paths(config, input_manifest)
    try:
        payload = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        batch_dir = paths.manifest_path.parent
        if batch_dir.is_dir():
            for other in batch_dir.glob("*.json"):
                try:
                    candidate = json.loads(other.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if candidate.get("input_fingerprint") == input_manifest.fingerprint:
                    raise _unavailable(config, input_manifest.batch_id, "prepared dataset does not match current preparation settings") from error
        raise _unavailable(config, input_manifest.batch_id, "prepared dataset is missing or unreadable") from error

    expected = {
        "schema_version": PREPARED_DATASET_SCHEMA_VERSION,
        "status": "READY",
        "batch_id": input_manifest.batch_id,
        "batch_started_at": input_manifest.batch_started_at.isoformat(),
        "input_fingerprint": input_manifest.fingerprint,
        "preparation_config_hash": paths.config_hash,
        "artifact_key": paths.artifact_key,
    }
    mismatches = [name for name, value in expected.items() if payload.get(name) != value]
    if mismatches:
        raise _unavailable(config, input_manifest.batch_id, f"prepared dataset manifest mismatch: {mismatches}")

    recorded_inputs = payload.get("input_files")
    if not isinstance(recorded_inputs, list) or len(recorded_inputs) != len(input_manifest.files):
        raise _unavailable(config, input_manifest.batch_id, "prepared dataset input file inventory is missing")
    input_root = (config.root("incoming_root") / input_manifest.batch_id).resolve()
    current_rows = {row["path"]: row for row in input_manifest.files}
    for item in recorded_inputs:
        try:
            relative = PurePosixPath(str(item["path"]))
            if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
                raise ValueError("unsafe input path")
            source = (input_root / Path(*relative.parts)).resolve(strict=True)
            if input_root not in source.parents or not source.is_file():
                raise ValueError("input file is outside the committed batch")
            stat = source.stat()
            committed = current_rows[str(item["path"])]
            if (
                stat.st_size != item["size_bytes"]
                or stat.st_mtime_ns != item["mtime_ns"]
                or stat.st_size != committed["size_bytes"]
            ):
                raise ValueError(f"raw input changed since preparation: {item['path']}")
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise _unavailable(config, input_manifest.batch_id, f"prepared dataset input is invalid: {error}") from error

    work_root = config.root("work_root").resolve()
    try:
        relative = PurePosixPath(str(payload["artifact_root"]))
        if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
            raise ValueError("unsafe artifact root")
        artifact_root = (work_root / Path(*relative.parts)).resolve(strict=True)
        if work_root not in artifact_root.parents or artifact_root != paths.artifact_root.resolve():
            raise ValueError("artifact root does not match the requested dataset")
    except (KeyError, OSError, ValueError) as error:
        raise _unavailable(config, input_manifest.batch_id, f"prepared dataset artifact root is invalid: {error}") from error

    recorded = payload.get("artifacts")
    if not isinstance(recorded, list) or not recorded:
        raise _unavailable(config, input_manifest.batch_id, "prepared dataset artifact inventory is missing")
    for item in recorded:
        try:
            relative = PurePosixPath(str(item["path"]))
            if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
                raise ValueError("unsafe artifact path")
            artifact = (artifact_root / Path(*relative.parts)).resolve(strict=True)
            if artifact_root not in artifact.parents or not artifact.is_file():
                raise ValueError("artifact is not a regular file inside the dataset")
            if artifact.stat().st_size != item["size_bytes"]:
                raise ValueError(f"artifact size changed: {item['path']}")
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise _unavailable(config, input_manifest.batch_id, f"prepared dataset artifact is invalid: {error}") from error

    missing = [str(path) for path in _required_artifacts(artifact_root, config.domain) if not path.exists()]
    if missing:
        raise _unavailable(config, input_manifest.batch_id, f"prepared dataset is missing required artifacts: {missing[:5]}")
    if not _has_playback_date_partitions(artifact_root):
        raise _unavailable(config, input_manifest.batch_id, "prepared dataset is missing playback event date partitions")
    windows_path = artifact_root / "data" / "data" / "canonical" / config.domain / "temporal" / "windows.json"
    try:
        windows_payload = json.loads(windows_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise _unavailable(config, input_manifest.batch_id, f"temporal windows are unreadable: {error}") from error
    if windows_payload.get("windows", []) != payload.get("temporal_windows"):
        raise _unavailable(config, input_manifest.batch_id, "temporal windows do not match the prepared manifest")
    return PreparedDataset(artifact_root, paths.manifest_path, payload)
