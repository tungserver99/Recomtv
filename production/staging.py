from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from production.config import ProductionConfig
from recsys_vnpt.batch_contracts import (
    BatchError,
    REQUIRED_SOURCES,
    load_and_validate_ready,
    write_ready_manifest,
)
from recsys_vnpt.runtime import atomic_write_text
from recsys_vnpt.source_manifest import sha256_file, stable_hash


_REQUIRED_DURATION_COLUMNS = {"profile_id", "username", "content_id", "duration", "create_date"}


@dataclass(frozen=True)
class StagingResult:
    batch_dir: Path
    fingerprint: str
    source_fingerprint: str
    reused: bool = False


def _source_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise BatchError("INPUT_INTEGRITY", f"source_root does not exist: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for required in REQUIRED_SOURCES:
        if not (root / required).exists():
            raise BatchError("INPUT_INTEGRITY", f"required source is missing: {required}")
    return files


def _notify(callback, event: str, **fields: Any) -> None:
    if callback is not None:
        callback(event, **fields)


def _source_fingerprint(files: list[Path], root: Path, progress_callback=None) -> str:
    artifacts = []
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(root).as_posix()
        _notify(progress_callback, "source_hash", status="started", file=relative, index=index, total=len(files))
        artifacts.append((relative, sha256_file(path)))
        _notify(progress_callback, "source_hash", status="completed", file=relative, index=index, total=len(files))
    return stable_hash(artifacts)


def _validate_duration_csv(source: Path) -> None:
    import pandas as pd

    columns = set(pd.read_csv(source, nrows=0).columns)
    missing = sorted(_REQUIRED_DURATION_COLUMNS - columns)
    if missing:
        raise BatchError("SCHEMA_DRIFT", f"duration CSV {source} is missing columns: {missing}")


def _validate_duration_parquet(source: Path) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise BatchError("CONFIG_INVALID", "Parquet duration staging requires the configured pyarrow dependency") from error
    missing = sorted(_REQUIRED_DURATION_COLUMNS - set(pq.ParquetFile(source).schema_arrow.names))
    if missing:
        raise BatchError("SCHEMA_DRIFT", f"duration Parquet {source} is missing columns: {missing}")


def _copy_duration_sources(source: Path, target: Path, progress_callback=None) -> None:
    if not source.is_dir():
        raise BatchError("INPUT_INTEGRITY", "required source is missing: duration")
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        _notify(progress_callback, "duration_copy", status="started", file=path.name)
        if path.suffix.lower() == ".csv":
            _validate_duration_csv(path)
        elif path.suffix.lower() == ".parquet":
            _validate_duration_parquet(path)
        shutil.copy2(path, target / path.name)
        _notify(progress_callback, "duration_copy", status="completed", file=path.name, bytes=path.stat().st_size)


def _copy_non_duration_sources(source: Path, target: Path) -> None:
    for child in source.iterdir():
        if child.name == "duration":
            continue
        destination = target / child.name
        if child.is_dir():
            shutil.copytree(child, destination)
        elif child.is_file():
            shutil.copy2(child, destination)


def _existing_batch(target: Path, batch_id: str, batch_started_at: datetime, source_fingerprint: str):
    ready = target / "_READY.json"
    if ready.is_file():
        try:
            manifest = load_and_validate_ready(target, batch_id)
            metadata = json.loads((target / "_PRODUCTION_SOURCE.json").read_text(encoding="utf-8"))
        except Exception as error:
            raise BatchError("INPUT_INTEGRITY", f"existing slot snapshot is invalid: {error}") from error
        if manifest.batch_started_at != batch_started_at:
            raise BatchError("INPUT_INTEGRITY", "existing batch has a different batch_started_at")
        if metadata.get("source_fingerprint") != source_fingerprint:
            raise BatchError("INPUT_INTEGRITY", "source snapshot changed after this slot was committed")
        return StagingResult(target, manifest.fingerprint, source_fingerprint, True)
    if target.exists():
        shutil.rmtree(target)
    return None


def build_input_snapshot(
    config: ProductionConfig,
    batch_id: str,
    batch_started_at: datetime,
    progress_callback: Callable[..., None] | None = None,
) -> StagingResult:
    if batch_started_at.tzinfo is None or batch_started_at.utcoffset() is None:
        raise ValueError("batch_started_at must be timezone-aware")
    files = _source_files(config.source_root)
    _notify(progress_callback, "source_fingerprint", status="started", files=len(files))
    source_fingerprint = _source_fingerprint(files, config.source_root, progress_callback)
    _notify(progress_callback, "source_fingerprint", status="completed", files=len(files), fingerprint=source_fingerprint)
    target = config.incoming_root / batch_id
    existing = _existing_batch(target, batch_id, batch_started_at, source_fingerprint)
    if existing:
        _notify(progress_callback, "input_snapshot", status="reused", batch_id=batch_id)
        return existing

    config.incoming_root.mkdir(parents=True, exist_ok=True)
    stage_parent = config.incoming_root / f".{batch_id}.{uuid.uuid4().hex}.stage"
    stage = stage_parent / batch_id
    stage.mkdir(parents=True)
    try:
        _notify(progress_callback, "metadata_copy", status="started")
        _copy_non_duration_sources(config.source_root, stage)
        _notify(progress_callback, "metadata_copy", status="completed")
        _notify(progress_callback, "duration_copy", status="started")
        _copy_duration_sources(config.source_root / "duration", stage / "duration", progress_callback)
        _notify(progress_callback, "duration_copy", status="completed")
        _notify(progress_callback, "source_snapshot_check", status="started")
        current_source_fingerprint = _source_fingerprint(_source_files(config.source_root), config.source_root, progress_callback)
        if current_source_fingerprint != source_fingerprint:
            raise BatchError("INPUT_INTEGRITY", "source snapshot changed while this batch was being staged")
        duration_source = config.source_root / "duration"
        if not any(p.suffix.lower() in {".csv", ".parquet"} for p in duration_source.iterdir() if p.is_file()):
            raise BatchError("INPUT_INTEGRITY", "duration directory has no CSV or Parquet events")
        atomic_write_text(
            stage / "_PRODUCTION_SOURCE.json",
            json.dumps(
                {
                    "source_fingerprint": source_fingerprint,
                    "batch_started_at": batch_started_at.isoformat(),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
                sort_keys=True,
            ),
        )
        write_ready_manifest(stage, batch_id=batch_id, batch_started_at=batch_started_at)
        load_and_validate_ready(stage, batch_id)
        stage.replace(target)
        _notify(progress_callback, "ready_manifest", status="committed")
        manifest = load_and_validate_ready(target, batch_id)
        return StagingResult(target, manifest.fingerprint, source_fingerprint)
    except Exception:
        shutil.rmtree(stage_parent, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage_parent, ignore_errors=True)

