"""Batch configuration, the input commit protocol, and stable failure codes.

Upstream writes an input batch into ``<incoming_root>/<batch_id>/`` and writes
``_READY.json`` last. Only then is the batch complete: the pipeline never infers
completeness from modification times, and every file it will read must be
listed in the manifest with its size and SHA-256.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from recsys_vnpt.gbdt import GBDT_BACKENDS, GBDT_DEVICES
from recsys_vnpt.orderings import OrderingConfig, expand_combination
from recsys_vnpt.rule_assembly import PAGE_DEDUP_POLICIES, RuleAssemblyConfig
from recsys_vnpt.runtime import atomic_write_text
from recsys_vnpt.source_manifest import sha256_file, stable_hash


READY_FILE = "_READY.json"
READY_SCHEMA_VERSION = 2
LEGACY_READY_SCHEMA_VERSION = 1
CONFIG_SCHEMA_VERSION = 1

# Stable failure codes and the process exit code each one maps to.
EXIT_CODES = {
    "INPUT_TIMEOUT": 10,
    "INPUT_INTEGRITY": 11,
    "SCHEMA_DRIFT": 12,
    "BATCH_LOCKED": 13,
    "CONFIG_INVALID": 14,
    "PREPARED_DATA_UNAVAILABLE": 15,
    "MODEL_UNAVAILABLE": 20,
    "RETRAIN_FAILED": 21,
    "QUALITY_GATE_FAILED": 22,
    "INFERENCE_FAILED": 30,
    "CAPACITY": 31,
    "OUTPUT_INVALID": 40,
    "PUBLISH_FAILED": 50,
}

# Raw inputs a batch must carry, by top-level name. Behavior logs are optional
# (config ``include_behavior``) and rule metadata falls back to the repository
# copy when the batch does not ship a newer one.
REQUIRED_SOURCES = (
    "duration",
    "mytv_vmp_content",
    "tags",
    "content_tags",
    "month_mytv_info.parquet",
)


class BatchError(RuntimeError):
    def __init__(self, code: str, message: str):
        if code not in EXIT_CODES:
            raise ValueError(f"unknown batch error code: {code}")
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message

    def __reduce__(self):
        # Worker processes send failures back pickled; keep both fields.
        return (BatchError, (self.code, self.message))

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.code]


@dataclass(frozen=True)
class BatchConfig:
    """One deployment's static configuration; the batch id comes per run."""

    incoming_root: str
    work_root: str
    delivery_root: str
    model_registry: str
    schema_version: int = CONFIG_SCHEMA_VERSION
    domain: str = "all"
    time_train: int = 24
    time_valid: int = 24
    time_test: int = 24
    input_wait_seconds: int = 3600
    poll_seconds: int = 30
    min_free_bytes: int = 10 * 1024**3
    include_behavior: bool = False
    num_shards: int = 16
    workers: int = 1
    shard_retries: int = 1
    candidate_budget: int = 1000
    recency_candidates_per_rule: int = 8
    recency_reserved_per_rule: int = 8
    recency_max_age_days: int = 30
    candidate_rules: int = 30
    reserved_per_rule: int = 33
    source_candidates_per_rule: int = 33
    rule_ordering: str = "learned_rrf"
    content_ordering: str = "learned_rrf"
    rule_scorer_fusion_weight: float = 1.25
    content_scorer_fusion_weight: float = 5.0
    content_rank_scope: str = "profile"
    content_fcn_family: str = "fcn_official"
    routed_fallback: str = "learned_rrf"
    gbdt_backend: str = "xgboost"
    gbdt_device: str = "auto"
    fcn_epochs: int = 2
    fcn_schedule_epochs: int = 30
    contents_per_rule_max: int = 30
    page_content_budget: int = 600
    page_dedup: str = "balanced_regret"
    gid: int = 3
    gate_max_regression: float = 0.02
    gate_evaluation_profiles: int = 5_000
    # Inference serves this many profiles (a stable hash sample) instead of
    # all of them; 0 serves all. For trial runs only: the delivery then
    # covers part of the base.
    max_profiles: int = 0
    sla_seconds: int = 6 * 3600
    qualification_workloads: tuple[int, ...] = (10_000, 100_000, 500_000, 1_000_000, 2_000_000)
    qualification_repetitions: int = 2

    def __post_init__(self) -> None:
        problems = []
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            problems.append(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
        if self.domain not in {"all", "clip", "movie", "music"}:
            problems.append("domain must be all, clip, movie or music")
        for name in ("time_train", "time_valid", "time_test"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                problems.append(f"{name} must be a positive integer")
        if self.num_shards < 1 or self.workers < 1:
            problems.append("num_shards and workers must be positive")
        if self.shard_retries < 0:
            problems.append("shard_retries cannot be negative")
        if self.candidate_budget < 200:
            problems.append("candidate_budget must be at least 200")
        if self.candidate_rules < 20 or min(self.reserved_per_rule, self.source_candidates_per_rule) < 1:
            problems.append("candidate_rules must be at least 20 and per-rule depths positive")
        if self.candidate_rules * self.reserved_per_rule > self.candidate_budget:
            problems.append("candidate_rules * reserved_per_rule exceeds candidate_budget")
        if not 10 <= self.contents_per_rule_max <= 30:
            problems.append("contents_per_rule_max must be within 10-30")
        if not 200 <= self.page_content_budget <= 600:
            problems.append("page_content_budget must be within 200-600")
        if self.page_dedup not in PAGE_DEDUP_POLICIES:
            problems.append(f"page_dedup must be one of {PAGE_DEDUP_POLICIES}")
        try:
            self.ordering
        except ValueError as error:
            problems.append(str(error))
        if not 1 <= self.fcn_epochs <= self.fcn_schedule_epochs:
            problems.append("fcn_epochs must be within 1..fcn_schedule_epochs")
        if self.gbdt_backend not in GBDT_BACKENDS:
            problems.append(f"gbdt_backend must be one of {GBDT_BACKENDS}")
        if self.gbdt_device not in GBDT_DEVICES:
            problems.append(f"gbdt_device must be one of {GBDT_DEVICES}")
        if not 0 <= self.gate_max_regression < 1:
            problems.append("gate_max_regression must be within [0, 1)")
        if self.max_profiles < 0:
            problems.append("max_profiles cannot be negative")
        if self.gate_evaluation_profiles < 1:
            problems.append("gate_evaluation_profiles must be positive")
        if self.input_wait_seconds < 0 or self.poll_seconds < 1:
            problems.append("input_wait_seconds must be >= 0 and poll_seconds >= 1")
        if not self.qualification_workloads or min(self.qualification_workloads) < 1:
            problems.append("qualification_workloads must be positive")
        if self.qualification_repetitions < 1:
            problems.append("qualification_repetitions must be positive")
        if problems:
            raise BatchError("CONFIG_INVALID", "; ".join(problems))

    @property
    def ordering(self) -> OrderingConfig:
        return OrderingConfig(
            rule_ordering=self.rule_ordering,
            content_ordering=self.content_ordering,
            content_rank_scope=self.content_rank_scope,
            content_fcn_family=self.content_fcn_family,
            routed_fallback=self.routed_fallback,
            rule_scorer_fusion_weight=self.rule_scorer_fusion_weight,
            content_scorer_fusion_weight=self.content_scorer_fusion_weight,
        )

    @property
    def uses_learned_scorers(self) -> bool:
        return self.ordering.needs_models

    @property
    def assembly(self) -> RuleAssemblyConfig:
        return RuleAssemblyConfig(
            contents_per_rule_max=self.contents_per_rule_max,
            page_content_budget=self.page_content_budget,
            page_dedup=self.page_dedup,
        )

    def page_fingerprint(self) -> str:
        """Hash of every setting that changes a served page; shards computed
        under a different one are never reused."""
        operational = {
            "incoming_root", "work_root", "delivery_root", "model_registry",
            "time_train", "time_valid", "time_test",
            "input_wait_seconds", "poll_seconds", "min_free_bytes", "workers", "shard_retries",
            "num_shards", "gate_max_regression", "gate_evaluation_profiles", "sla_seconds",
            "qualification_workloads", "qualification_repetitions",
        }
        return stable_hash({k: v for k, v in asdict(self).items() if k not in operational})

    def root(self, name: str) -> Path:
        return Path(getattr(self, name)).expanduser()


def load_config(path: str | Path) -> BatchConfig:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BatchError("CONFIG_INVALID", f"cannot read {path}: {error}") from error
    if not isinstance(raw, dict):
        raise BatchError("CONFIG_INVALID", "config must be a JSON object")
    # A named combination (orderings.COMBINATIONS) fills in the ordering keys;
    # setting one of them differently as well is a contradiction, not an override.
    combination = raw.pop("combination", None)
    if combination is not None:
        try:
            preset = expand_combination(combination)
        except ValueError as error:
            raise BatchError("CONFIG_INVALID", str(error)) from error
        clashes = sorted(name for name, value in preset.items() if name in raw and raw[name] != value)
        if clashes:
            raise BatchError(
                "CONFIG_INVALID", f"combination {combination} sets {clashes}, which the config sets differently"
            )
        raw = {**raw, **preset}
    known = {item.name for item in fields(BatchConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise BatchError("CONFIG_INVALID", f"unknown config keys: {unknown}")
    if "qualification_workloads" in raw:
        raw["qualification_workloads"] = tuple(raw["qualification_workloads"])
    try:
        return BatchConfig(**raw)
    except TypeError as error:
        raise BatchError("CONFIG_INVALID", str(error)) from error


@dataclass(frozen=True)
class ReadyManifest:
    batch_id: str
    batch_started_at: datetime
    created_at: datetime
    files: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def fingerprint(self) -> str:
        """Identity of the input content, independent of its batch id."""
        return stable_hash(sorted((row["path"], row["sha256"]) for row in self.files))


def _aware(value: Any, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as error:
        raise BatchError("INPUT_INTEGRITY", f"{name} is not ISO-8601: {value!r}") from error
    if parsed.tzinfo is None:
        raise BatchError("INPUT_INTEGRITY", f"{name} must carry a timezone offset")
    return parsed


def batch_started_at_from_metadata(metadata: dict[str, Any]) -> datetime:
    """Read new timestamps or map a legacy data_cutoff for old records."""
    value = metadata.get("batch_started_at")
    if value is None:
        value = metadata.get("data_cutoff")
    return _aware(value, "batch_started_at")


def _listed_files(batch_dir: Path) -> set[str]:
    return {
        path.relative_to(batch_dir).as_posix()
        for path in batch_dir.rglob("*")
        if path.is_file() and path.name != READY_FILE
    }


def load_and_validate_ready(
    batch_dir: str | Path,
    batch_id: str,
    *,
    verify_checksums: bool = True,
) -> ReadyManifest:
    """Parse ``_READY.json`` and optionally verify every listed file byte for byte.

    Consumers of an already prepared immutable batch may set
    ``verify_checksums=False``. That still checks file presence, size, safe paths,
    required sources, and the exact listed-file set without reading the raw bytes.
    """
    batch_dir = Path(batch_dir)
    ready_path = batch_dir / READY_FILE
    try:
        raw = json.loads(ready_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BatchError("INPUT_INTEGRITY", f"unreadable {ready_path}: {error}") from error
    schema_version = raw.get("schema_version")
    if schema_version not in {LEGACY_READY_SCHEMA_VERSION, READY_SCHEMA_VERSION}:
        raise BatchError("INPUT_INTEGRITY", f"unsupported ready schema: {raw.get('schema_version')}")
    if raw.get("batch_id") != batch_id or batch_dir.name != batch_id:
        raise BatchError(
            "INPUT_INTEGRITY",
            f"batch id mismatch: manifest={raw.get('batch_id')!r}, "
            f"directory={batch_dir.name!r}, requested={batch_id!r}",
        )
    entries = raw.get("files")
    if not isinstance(entries, list) or not entries:
        raise BatchError("INPUT_INTEGRITY", "manifest lists no files")

    root = batch_dir.resolve()
    seen: set[str] = set()
    files = []
    for entry in entries:
        relative = str(entry.get("path", ""))
        # Manifest paths are POSIX paths relative to the batch, on any platform.
        parts = PurePosixPath(relative).parts
        if (
            not relative
            or PurePosixPath(relative).is_absolute()
            or ".." in parts
            or "\\" in relative
            or re.match(r"^[A-Za-z]:", relative)
        ):
            raise BatchError("INPUT_INTEGRITY", f"unsafe manifest path: {relative!r}")
        if relative in seen:
            raise BatchError("INPUT_INTEGRITY", f"duplicate manifest path: {relative}")
        seen.add(relative)
        try:
            resolved = (batch_dir / relative).resolve(strict=True)
        except OSError as error:
            raise BatchError("INPUT_INTEGRITY", f"listed file is missing: {relative}") from error
        if root not in resolved.parents:
            raise BatchError("INPUT_INTEGRITY", f"path escapes the batch: {relative}")
        digest = str(entry.get("sha256", ""))
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise BatchError("INPUT_INTEGRITY", f"invalid sha256 for {relative}")
        if resolved.stat().st_size != entry.get("size_bytes"):
            raise BatchError("INPUT_INTEGRITY", f"size mismatch: {relative}")
        if verify_checksums and sha256_file(resolved) != digest:
            raise BatchError("INPUT_INTEGRITY", f"checksum mismatch: {relative}")
        files.append({"path": relative, "size_bytes": entry["size_bytes"], "sha256": digest})

    unlisted = sorted(_listed_files(batch_dir) - seen)
    if unlisted:
        raise BatchError(
            "INPUT_INTEGRITY",
            f"{len(unlisted)} files are not in the manifest, e.g. {unlisted[:3]}",
        )
    present = {Path(row["path"]).parts[0] for row in files}
    missing = [name for name in REQUIRED_SOURCES if name not in present]
    if missing:
        raise BatchError("INPUT_INTEGRITY", f"required sources are missing: {missing}")
    return ReadyManifest(
        batch_id=batch_id,
        batch_started_at=batch_started_at_from_metadata(raw),
        created_at=_aware(raw.get("created_at"), "created_at"),
        files=tuple(sorted(files, key=lambda row: row["path"])),
    )


def load_ready_manifest_identity(batch_dir: str | Path, batch_id: str) -> ReadyManifest:
    """Quickly validate the committed manifest without rereading large sources."""
    return load_and_validate_ready(batch_dir, batch_id, verify_checksums=False)


def wait_for_ready(
    batch_dir: str | Path,
    timeout_seconds: float,
    poll_seconds: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Block until ``_READY.json`` exists, or fail with INPUT_TIMEOUT."""
    ready = Path(batch_dir) / READY_FILE
    deadline = clock() + timeout_seconds
    while not ready.is_file():
        if clock() >= deadline:
            raise BatchError(
                "INPUT_TIMEOUT", f"{ready} did not appear within {timeout_seconds}s"
            )
        sleep(poll_seconds)
    return ready


def write_ready_manifest(
    batch_dir: str | Path,
    batch_id: str | None = None,
    batch_started_at: datetime | None = None,
) -> Path:
    """Hash every file of a finished batch and write ``_READY.json`` last.

    The reference implementation of the upstream side of the protocol.
    """
    batch_dir = Path(batch_dir)
    if batch_started_at is not None and (
        batch_started_at.tzinfo is None or batch_started_at.utcoffset() is None
    ):
        raise ValueError("batch_started_at must carry a timezone")
    existing = batch_dir / READY_FILE
    if existing.is_file():
        manifest = load_and_validate_ready(batch_dir, batch_id or batch_dir.name)
        if batch_started_at is not None and manifest.batch_started_at != batch_started_at:
            raise BatchError("INPUT_INTEGRITY", "batch_started_at conflicts with the committed READY manifest")
        return existing
    batch_started_at = batch_started_at or datetime.now(timezone.utc)
    files = [
        {
            "path": relative,
            "size_bytes": (batch_dir / relative).stat().st_size,
            "sha256": sha256_file(batch_dir / relative),
        }
        for relative in sorted(_listed_files(batch_dir))
    ]
    payload = {
        "schema_version": READY_SCHEMA_VERSION,
        "batch_id": batch_id or batch_dir.name,
        "batch_started_at": batch_started_at.isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    path = batch_dir / READY_FILE
    atomic_write_text(path, json.dumps(payload, indent=2))
    return path
