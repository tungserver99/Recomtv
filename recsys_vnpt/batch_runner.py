"""Production batch coordinator: inference, retraining, hardware qualification.

Inference turns one committed input batch into one committed delivery
generation. Profiles are split into shards by a stable hash of ``profile_id``;
each shard is computed, validated and written with a success record last, so a
rerun after a failure recomputes only the shards that are missing or do not
verify. Shards are merged by streaming them in profile order, which makes the
three delivered files byte-identical whatever the shard or worker count.

Retraining fits both learned scorers, and the FCN models the configured
ordering uses, on the batch; gates the new bundle against the currently
promoted bundle on the temporal test window; and moves the promotion pointer
only when the gate passes. Inference never fits anything: it serves the
promoted bundle or fails.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import multiprocessing
import os
import platform
import re
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from recsys_vnpt.batch_contracts import (
    BatchConfig,
    BatchError,
    ReadyManifest,
    batch_started_at_from_metadata,
    load_and_validate_ready,
    wait_for_ready,
)
from recsys_vnpt.batch_state import BatchLock, BatchStatus
from recsys_vnpt.export import (
    JsonArrayWriter,
    build_profile_grouping,
    commit_delivery_generation,
    production_output_names,
    read_current_delivery,
    serialize_assembled_output,
)
from recsys_vnpt.gbdt import serve_on
from recsys_vnpt.output_validation import validate_client_output
from recsys_vnpt.paths import data_artifact_path
from recsys_vnpt.prepared_dataset import (
    load_prepared_dataset,
    prepared_dataset_paths,
    write_prepared_dataset_manifest,
)
from recsys_vnpt.runtime import atomic_write_text
from recsys_vnpt.scorer_bundle import (
    BundleError,
    bundle_identity,
    load_bundle,
    load_bundle_fcn,
    promote,
    promoted_bundle,
    save_bundle,
)
from recsys_vnpt.source_manifest import sha256_file, stable_hash


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PART_FILES = ("homepage.jsonl", "rule_content.txt", "grouping.jsonl")
SHARD_SUCCESS = "_SHARD_SUCCESS.json"
GATE_METRICS = ("rule_ndcg@20", "macro_content_ndcg@10")

# Output metrics that must be exact for a delivery to be published.
HARD_OUTPUT_RATES = (
    "profile_output_coverage",
    "exact_20_normal_rule_rate",
    "content_count_within_range_rate",
    "special_rule_complete_rate",
    "cross_file_consistency_rate",
    "special_no_content_rate",
)
HARD_OUTPUT_COUNTS = (
    "duplicate_within_rule_count",
    "invalid_content_count",
    "unmatched_rule_count",
)

# Which failure code an unexpected exception maps to, by the phase it hit.
PHASE_ERROR_CODES = {
    "PREPARING_DATA": "SCHEMA_DRIFT",
    "LOADING_PREPARED_DATA": "PREPARED_DATA_UNAVAILABLE",
    "LOADING_MODEL": "MODEL_UNAVAILABLE",
    "RUNNING_SHARDS": "INFERENCE_FAILED",
    "MEASURING": "INFERENCE_FAILED",
    "MERGING": "OUTPUT_INVALID",
    "VALIDATING_OUTPUT": "OUTPUT_INVALID",
    "PUBLISHING": "PUBLISH_FAILED",
    "TRAINING": "RETRAIN_FAILED",
    "EVALUATING": "RETRAIN_FAILED",
    "GATING": "RETRAIN_FAILED",
}


def code_version() -> str:
    """The commit being run, or RECSYS_VNPT_CODE_VERSION for a deployment
    without git metadata."""
    configured = os.environ.get("RECSYS_VNPT_CODE_VERSION")
    if configured:
        return configured
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _check_batch_id(batch_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}", batch_id):
        raise BatchError(
            "CONFIG_INVALID",
            f"batch id {batch_id!r} must be letters, digits and . _ + - only",
        )


@contextmanager
def _roots(
    source: Path,
    data: Path,
    temporal: Path | None = None,
    data_end_at=None,
) -> Iterator[None]:
    """Point the raw readers at the batch and generated artifacts at ``data``."""
    names = (
        "RECSYS_VNPT_SOURCE_ROOT", "RECSYS_VNPT_DATA_ROOT",
        "RECSYS_VNPT_TEMPORAL_ROOT", "RECSYS_VNPT_DATA_END_AT",
    )
    saved = {name: os.environ.get(name) for name in names}
    os.environ["RECSYS_VNPT_SOURCE_ROOT"] = str(source)
    os.environ["RECSYS_VNPT_DATA_ROOT"] = str(data)
    if temporal is not None:
        os.environ["RECSYS_VNPT_TEMPORAL_ROOT"] = str(temporal)
    else:
        os.environ.pop("RECSYS_VNPT_TEMPORAL_ROOT", None)
    if data_end_at is not None:
        os.environ["RECSYS_VNPT_DATA_END_AT"] = str(data_end_at)
    else:
        os.environ.pop("RECSYS_VNPT_DATA_END_AT", None)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def _waiting_lock(locks: Path, name: str, timeout: float, poll: float) -> Iterator[None]:
    """Like BatchLock, but waits: inference and retraining of one input batch
    share its prepared artifacts and must not build them concurrently."""
    deadline = time.monotonic() + timeout
    lock = BatchLock(locks, name)
    while True:
        try:
            lock.__enter__()
            break
        except BatchError as error:
            if error.code != "BATCH_LOCKED" or time.monotonic() >= deadline:
                raise
            time.sleep(poll)
    try:
        yield
    finally:
        lock.__exit__(None, None, None)


# --- input and data preparation ------------------------------------------------


def _validate_input(
    config: BatchConfig,
    batch_id: str,
    *,
    verify_checksums: bool = True,
) -> ReadyManifest:
    manifest = load_and_validate_ready(
        config.root("incoming_root") / batch_id,
        batch_id,
        verify_checksums=verify_checksums,
    )
    work = config.root("work_root")
    work.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(work).free
    if free < config.min_free_bytes:
        raise BatchError("CAPACITY", f"{free} bytes free under {work}, need {config.min_free_bytes}")
    return manifest


def _prepare(
    config: BatchConfig,
    manifest: ReadyManifest,
    temporal: bool = True,
    progress=None,
    data_root: Path | None = None,
    timings: dict[str, float] | None = None,
) -> Path:
    """Build canonical and optional temporal artifacts for an explicit prepare job."""
    from recsys_vnpt.pipeline_v2 import build_temporal_project, prepare_canonical_project

    data_root = data_root or prepared_dataset_paths(config, manifest).artifact_root
    source = config.root("incoming_root") / manifest.batch_id
    with _roots(source, data_root):
        try:
            if progress:
                progress("canonical_prepare", status="started", temporal=temporal)
            step_started = time.perf_counter()
            prepare_canonical_project(
                PROJECT_ROOT,
                config.domain,
                include_behavior=config.include_behavior,
                user_reference_date=pd.Timestamp(manifest.batch_started_at.date()),
                require_precise_create_date=temporal,
                progress_callback=progress,
            )
            if timings is not None:
                timings["canonical_seconds"] = time.perf_counter() - step_started
            if progress:
                progress("prepare_step_timing", step="canonical", seconds=time.perf_counter() - step_started)
            if progress:
                progress("canonical_prepare", status="completed", temporal=temporal)
            if temporal:
                if progress:
                    progress("temporal_build", status="started")
                step_started = time.perf_counter()
                build_temporal_project(
                    PROJECT_ROOT,
                    config.domain,
                    window_mode="relative",
                    time_train=config.time_train,
                    time_valid=config.time_valid,
                    time_test=config.time_test,
                    progress_callback=progress,
                )
                if timings is not None:
                    timings["temporal_seconds"] = time.perf_counter() - step_started
                if progress:
                    progress("prepare_step_timing", step="temporal", seconds=time.perf_counter() - step_started)
                if progress:
                    progress("temporal_build", status="completed")
        except MemoryError as error:
            raise BatchError("CAPACITY", "out of memory while preparing data") from error
        except BatchError:
            raise
        except Exception as error:
            raise BatchError("SCHEMA_DRIFT", f"{type(error).__name__}: {error}") from error
    return data_root


def run_prepare_data_batch(config: BatchConfig, batch_id: str) -> dict[str, Any]:
    """Validate input and materialize the reusable canonical/temporal dataset."""
    def body(work: Path, status: BatchStatus) -> dict[str, Any]:
        wait_for_ready(config.root("incoming_root") / batch_id, config.input_wait_seconds, config.poll_seconds)
        status.transition("VALIDATING_INPUT")
        validation_started = time.perf_counter()
        manifest = _validate_input(config, batch_id, verify_checksums=False)
        validation_seconds = time.perf_counter() - validation_started
        paths = prepared_dataset_paths(config, manifest)
        with _waiting_lock(
            config.root("work_root") / "locks",
            f"prepare-{paths.artifact_key}",
            config.input_wait_seconds,
            config.poll_seconds,
        ):
            try:
                dataset = load_prepared_dataset(config, manifest)
            except BatchError as error:
                if error.code != "PREPARED_DATA_UNAVAILABLE":
                    raise
                status.log("prepared_dataset_cache", status="miss", reason=error.message)
            else:
                status.transition(
                    "PREPARING_DATA",
                    input_fingerprint=manifest.fingerprint,
                    prepared_dataset_key=paths.artifact_key,
                    cache="hit",
                )
                status.transition("PREPARED", prepared_dataset=str(dataset.manifest_path), reused=True)
                return {
                    "status": "prepared",
                    "reused": True,
                    "batch_id": batch_id,
                    "input_fingerprint": manifest.fingerprint,
                    "preparation_config_hash": paths.config_hash,
                    "prepared_dataset": str(dataset.manifest_path),
                    "artifact_root": str(dataset.artifact_root),
                    "validation_seconds": validation_seconds,
                    "prepare_seconds": 0.0,
                }

            full_validation_started = time.perf_counter()
            verified_manifest = _validate_input(config, batch_id, verify_checksums=True)
            if verified_manifest.fingerprint != manifest.fingerprint:
                raise BatchError(
                    "INPUT_INTEGRITY",
                    "input manifest changed while prepare-data was resolving its cache; retry the job",
                )
            manifest = verified_manifest
            validation_seconds = time.perf_counter() - validation_started
            status.log(
                "input_validation_timing",
                quick_seconds=full_validation_started - validation_started,
                checksum_seconds=time.perf_counter() - full_validation_started,
            )

            status.transition(
                "PREPARING_DATA",
                input_fingerprint=manifest.fingerprint,
                prepared_dataset_key=paths.artifact_key,
                cache="miss",
            )
            write_prepared_dataset_manifest(config, manifest, status="PREPARING")
            prepare_started = time.perf_counter()
            timings: dict[str, float] = {"validation_seconds": validation_seconds}
            try:
                artifact_root = _prepare(
                    config,
                    manifest,
                    temporal=True,
                    progress=status.log,
                    data_root=paths.artifact_root,
                    timings=timings,
                )
                timings["prepare_seconds"] = time.perf_counter() - prepare_started
                manifest_path = write_prepared_dataset_manifest(
                    config, manifest, status="READY", timings=timings
                )
            except Exception as error:
                timings["prepare_seconds"] = time.perf_counter() - prepare_started
                try:
                    write_prepared_dataset_manifest(
                        config,
                        manifest,
                        status="FAILED",
                        timings=timings,
                        error=f"{type(error).__name__}: {error}",
                    )
                except Exception:
                    pass
                if isinstance(error, BatchError):
                    raise
                if isinstance(error, MemoryError):
                    raise BatchError("CAPACITY", "out of memory while preparing data") from error
                raise BatchError("SCHEMA_DRIFT", f"{type(error).__name__}: {error}") from error

            status.transition("PREPARED", prepared_dataset=str(manifest_path), reused=False)
            return {
                "status": "prepared",
                "reused": False,
                "batch_id": batch_id,
                "input_fingerprint": manifest.fingerprint,
                "preparation_config_hash": paths.config_hash,
                "prepared_dataset": str(manifest_path),
                "artifact_root": str(artifact_root),
                **timings,
            }

    return _run(config, batch_id, "prepare", body, allow_already_done=False)


def sample_profiles(profile_ids: list[str], limit: int) -> list[str]:
    """The first ``limit`` profiles by SHA-256 of the id, so every run and
    machine serves the same sample; 0 keeps every profile."""
    if not limit:
        return profile_ids
    return sorted(profile_ids, key=lambda pid: hashlib.sha256(pid.encode("utf-8")).hexdigest())[:limit]


def _profile_ids(domain: str, data_end_at=None, progress_callback=None) -> list[str]:
    if data_end_at is not None:
        canonical = data_artifact_path("canonical", domain, project_root=PROJECT_ROOT)
        from recsys_vnpt.prepare import _read_canonical_interactions

        interactions = _read_canonical_interactions(
            canonical / "interactions.parquet",
            columns=["profile_id"],
            data_end_at=data_end_at,
            progress_callback=progress_callback,
        )
        return interactions["profile_id"].astype(str).drop_duplicates().tolist()
    mapping = pd.read_parquet(
        data_artifact_path("canonical", domain, "profile_mapping.parquet", project_root=PROJECT_ROOT),
        columns=["profile_id"],
    )
    return mapping["profile_id"].astype(str).drop_duplicates().tolist()


def _promoted(
    config: BatchConfig,
    *,
    model_path: str | Path | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    """Return a verified explicit bundle, or the promoted bundle by default."""
    if not config.uses_learned_scorers:
        if model_path is not None:
            raise BatchError("MODEL_UNAVAILABLE", "the configured ordering does not use a learned model bundle")
        return None, {"bundle_id": None, "model_sha256": None}
    if model_path is not None:
        try:
            bundle = Path(model_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise BatchError("MODEL_UNAVAILABLE", f"model bundle path is unavailable: {error}") from error
        if not bundle.is_dir():
            raise BatchError("MODEL_UNAVAILABLE", f"model bundle path is not a directory: {bundle}")
        model_source = "explicit"
    else:
        bundle = promoted_bundle(config.root("model_registry"))
        if bundle is None:
            raise BatchError(
                "MODEL_UNAVAILABLE",
                "no promoted scorer bundle; run run-retrain first, or configure a "
                "combination that uses no learned model (HxH, HsegxHseg)",
            )
        bundle = bundle.resolve()
        model_source = "promoted"
    try:
        _, _, manifest = load_bundle(bundle)
        load_bundle_fcn(bundle, config.ordering.fcn_families)
    except (BundleError, OSError, ValueError, KeyError) as error:
        raise BatchError("MODEL_UNAVAILABLE", str(error)) from error
    return bundle, {
        "bundle_id": manifest["bundle_id"],
        "model_sha256": bundle_identity(manifest),
        "model_data_end_at": manifest.get("data_end_at"),
        "model_serving_cutoff": manifest.get("serving_cutoff"),
        "model_source": model_source,
        "model_path": str(bundle),
    }


# --- shards ----------------------------------------------------------------------


def shard_of(profile_id: str, num_shards: int) -> int:
    digest = hashlib.sha256(str(profile_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def assign_shards(profile_ids: list[str], num_shards: int) -> list[list[str]]:
    shards: list[list[str]] = [[] for _ in range(num_shards)]
    for profile_id in sorted(set(map(str, profile_ids))):
        shards[shard_of(profile_id, num_shards)].append(profile_id)
    return shards


def _serving_context(config: BatchConfig, bundle: str | None, data_end_at=None, progress_callback=None):
    from recsys_vnpt.serving import (
        _member_id_map,
        load_serving_context,
        serving_retrieval_config,
        with_scorers,
    )

    context = load_serving_context(
        PROJECT_ROOT,
        config.domain,
        serving_retrieval_config(
            config.candidate_budget,
            False,
            config.recency_candidates_per_rule,
            config.recency_reserved_per_rule,
            config.recency_max_age_days,
            config.candidate_rules,
            config.reserved_per_rule,
            config.source_candidates_per_rule,
        ),
        config.assembly,
        data_end_at=data_end_at,
        progress_callback=progress_callback,
    )
    rule_model = content_model = None
    fcn = {}
    if bundle is not None:
        rule_model, content_model, _ = load_bundle(bundle)
        serve_on(rule_model, config.gbdt_device)
        serve_on(content_model, config.gbdt_device)
        fcn = load_bundle_fcn(bundle, config.ordering.fcn_families)
    context = with_scorers(context, config.ordering, rule_model, content_model, fcn)
    return context, _member_id_map(context.profiles)


def _verified_shard(directory: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    """A previous shard result, if it was computed for exactly this identity
    and every file still matches its recorded hash."""
    success_path = directory / SHARD_SUCCESS
    if not success_path.is_file():
        return None
    try:
        success = json.loads(success_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if success.get("identity") != identity:
        return None
    for name in PART_FILES:
        path = directory / name
        if not path.is_file() or sha256_file(path) != success["files"].get(name):
            return None
    return success


def _run_shard(
    context: Any,
    member_ids: dict[str, str],
    config: BatchConfig,
    directory: Path,
    profile_ids: list[str],
    identity: dict[str, Any],
) -> dict[str, Any]:
    from recsys_vnpt.serving import recommend_profiles

    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / SHARD_SUCCESS).unlink(missing_ok=True)
    pages: dict[str, dict[str, Any]] = {}
    if profile_ids:
        pages, _ = recommend_profiles(context, profile_ids)
    homepage, rule_content = serialize_assembled_output(pages, config.assembly)
    validation = validate_client_output(
        homepage,
        rule_content,
        set(profile_ids),
        context.catalog,
        context.valid_membership,
        config.assembly,
    )
    if not validation.passed:
        raise BatchError(
            "OUTPUT_INVALID",
            f"shard {identity['shard']}: " + "; ".join(validation.errors[:5]),
        )
    grouping = build_profile_grouping(homepage, config.gid, member_ids)
    lines = {
        "homepage.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in homepage),
        "rule_content.txt": rule_content,
        "grouping.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in grouping),
    }
    for name, text in lines.items():
        atomic_write_text(directory / name, text)
    success = {
        "identity": identity,
        "profiles": len(profile_ids),
        "serving_cutoff": str(context.cutoff),
        "files": {name: sha256_file(directory / name) for name in PART_FILES},
        "validation": validation.metrics,
        "seconds": time.perf_counter() - started,
    }
    atomic_write_text(directory / SHARD_SUCCESS, json.dumps(success, indent=2, sort_keys=True))
    return success


_WORKER: dict[str, Any] = {}


def _init_worker(
    config_fields: dict[str, Any], bundle: str | None, roots: dict[str, str],
    data_end_at=None, status_path: str | None = None, batch_id: str | None = None,
    run_id: str | None = None,
) -> None:
    os.environ.update(roots)
    config = BatchConfig(**config_fields)
    if data_end_at is not None:
        os.environ["RECSYS_VNPT_DATA_END_AT"] = str(data_end_at)
    else:
        os.environ.pop("RECSYS_VNPT_DATA_END_AT", None)
    _WORKER["config"] = config
    log = BatchStatus(Path(status_path), batch_id, "inference", run_id).log if status_path and batch_id else None
    _WORKER["context"], _WORKER["member_ids"] = _serving_context(config, bundle, data_end_at, log)


def _worker_shard(directory: str, profile_ids: list[str], identity: dict[str, Any]) -> dict[str, Any]:
    return _run_shard(
        _WORKER["context"],
        _WORKER["member_ids"],
        _WORKER["config"],
        Path(directory),
        profile_ids,
        identity,
    )


def _compute_shards(
    config: BatchConfig,
    pending: list[tuple[int, Path, list[str], dict[str, Any]]],
    bundle_path: str | None,
    results: dict[int, dict[str, Any]],
    log,
    data_end_at=None,
) -> list[tuple[int, BaseException]]:
    """One pass over ``pending``; returns the shards that failed for reasons a
    retry can fix. A contract failure (BatchError) is never retried."""
    failures: list[tuple[int, BaseException]] = []
    if config.workers == 1:
        try:
            context, member_ids = _serving_context(config, bundle_path, data_end_at, log)
        except (BatchError, MemoryError):
            raise
        except Exception as error:
            return [(index, error) for index, *_ in pending]
        for index, directory, ids, shard_identity in pending:
            try:
                results[index] = _run_shard(context, member_ids, config, directory, ids, shard_identity)
            except (BatchError, MemoryError):
                raise
            except Exception as error:
                failures.append((index, error))
                continue
            if log:
                log("shard_done", shard=index, profiles=len(ids), seconds=results[index]["seconds"])
        return failures
    roots = {
        name: os.environ[name]
        for name in ("RECSYS_VNPT_SOURCE_ROOT", "RECSYS_VNPT_DATA_ROOT", "RECSYS_VNPT_TEMPORAL_ROOT", "RECSYS_VNPT_DATA_END_AT")
        if name in os.environ
    }
    # Spawn, not fork: forking while another thread (the qualification memory
    # sampler) holds a lock can deadlock the child. A worker that dies breaks
    # the whole pool, so every shard it leaves unfinished is retried together.
    with ProcessPoolExecutor(
        max_workers=min(config.workers, len(pending)),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(
            asdict(config), bundle_path, roots, data_end_at,
            str(getattr(getattr(log, "__self__", None), "path", "")) or None,
            getattr(getattr(log, "__self__", None), "batch_id", None),
            getattr(getattr(log, "__self__", None), "run_id", None),
        ),
    ) as pool:
        futures = {
            pool.submit(_worker_shard, str(directory), ids, shard_identity): index
            for index, directory, ids, shard_identity in pending
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except BatchError:
                raise
            except Exception as error:
                failures.append((index, error))
                continue
            if log:
                log("shard_done", shard=index, seconds=results[index]["seconds"])
    return failures


def run_shards(
    config: BatchConfig,
    profile_ids: list[str],
    shard_root: Path,
    identity: dict[str, Any],
    bundle: Path | None,
    reuse: bool,
    log=None,
    data_end_at=None,
) -> list[dict[str, Any]]:
    """Compute every shard not already verified under ``shard_root``.

    With ``workers > 1`` each worker process loads its own serving context, so
    memory grows with the worker count; qualification picks the count. Shards
    that fail for a transient reason (a dead worker, an I/O error) are retried
    ``shard_retries`` times before the run fails.
    """
    shards = assign_shards(profile_ids, config.num_shards)
    results: dict[int, dict[str, Any]] = {}
    pending = []
    for index, ids in enumerate(shards):
        directory = shard_root / f"shard_{index:05d}"
        shard_identity = {
            **identity,
            "shard": index,
            "num_shards": config.num_shards,
            "profiles_sha256": stable_hash(ids),
        }
        previous = _verified_shard(directory, shard_identity) if reuse else None
        if previous is not None:
            results[index] = {**previous, "reused": True}
        else:
            pending.append((index, directory, ids, shard_identity))
    if log:
        log("shards", total=len(shards), reused=len(results), pending=len(pending))

    bundle_path = str(bundle) if bundle is not None else None
    for attempt in range(config.shard_retries + 1):
        if not pending:
            break
        failures = _compute_shards(config, pending, bundle_path, results, log, data_end_at)
        if not failures:
            break
        if attempt == config.shard_retries:
            index, error = failures[0]
            raise BatchError(
                "INFERENCE_FAILED",
                f"{len(failures)} shards failed after {attempt + 1} attempts; "
                f"shard {index}: {type(error).__name__}: {error}",
            ) from error
        failed = {index for index, _ in failures}
        if log:
            log(
                "shard_retry",
                shards=sorted(failed),
                errors=sorted({f"{type(error).__name__}: {error}"[:200] for _, error in failures}),
            )
        pending = [item for item in pending if item[0] in failed]
    return [results[index] for index in range(len(shards))]


# --- merge -------------------------------------------------------------------------


def _json_pid(line: str) -> str:
    return str(json.loads(line)["pid"])


def _content_pid(line: str) -> str:
    return line.split("|", 1)[0]


def merge_shards(
    shard_dirs: list[Path],
    output: Path,
    names: dict[str, str],
    expected_profiles: set[str],
) -> int:
    """Stream the shard parts into the three delivery files in profile order.

    Rejects a profile that is missing, unexpected, repeated, or not on the same
    position in all three files.
    """
    output.mkdir(parents=True, exist_ok=True)
    handles = []

    def stream(name: str, key):
        files = [(directory / name).open("r", encoding="utf-8", newline="") for directory in shard_dirs]
        handles.extend(files)
        return heapq.merge(*files, key=key)

    try:
        merged = zip_longest(
            stream("homepage.jsonl", _json_pid),
            stream("rule_content.txt", _content_pid),
            stream("grouping.jsonl", _json_pid),
        )
        count = 0
        previous = None
        with (output / names["homepage"]).open("w", encoding="utf-8", newline="") as homepage_file, (
            output / names["rule_content"]
        ).open("w", encoding="utf-8", newline="") as content_file, (
            output / names["grouping"]
        ).open("w", encoding="utf-8", newline="") as grouping_file:
            homepage = JsonArrayWriter(homepage_file)
            grouping = JsonArrayWriter(grouping_file)
            for homepage_line, content_line, grouping_line in merged:
                if homepage_line is None or content_line is None or grouping_line is None:
                    raise BatchError("OUTPUT_INVALID", "shard files disagree on profile count")
                pid = _json_pid(homepage_line)
                if _content_pid(content_line) != pid or _json_pid(grouping_line) != pid:
                    raise BatchError("OUTPUT_INVALID", f"profile {pid} is not aligned across files")
                if previous is not None and pid <= previous:
                    raise BatchError("OUTPUT_INVALID", f"profile {pid} appears more than once")
                if pid not in expected_profiles:
                    raise BatchError("OUTPUT_INVALID", f"unexpected profile {pid}")
                homepage.write_line(homepage_line.rstrip("\n"))
                content_file.write(content_line if content_line.endswith("\n") else content_line + "\n")
                grouping.write_line(grouping_line.rstrip("\n"))
                previous = pid
                count += 1
            homepage.close()
            grouping.close()
    finally:
        for handle in handles:
            handle.close()
    if count != len(expected_profiles):
        raise BatchError(
            "OUTPUT_INVALID", f"{len(expected_profiles) - count} expected profiles have no page"
        )
    return count


def aggregate_validation(shards: list[dict[str, Any]]) -> dict[str, Any]:
    """Profile-weighted rates and summed counts over all shards."""
    total = sum(shard["profiles"] for shard in shards)
    summary: dict[str, Any] = {"profiles": total}
    for name in HARD_OUTPUT_RATES:
        summary[name] = (
            sum(shard["validation"][name] * shard["profiles"] for shard in shards) / total
            if total
            else 0.0
        )
    for name in (*HARD_OUTPUT_COUNTS, "duplicate_across_rules_count"):
        summary[name] = sum(int(shard["validation"].get(name, 0)) for shard in shards)
    failures = [name for name in HARD_OUTPUT_RATES if summary[name] != 1.0] + [
        name for name in HARD_OUTPUT_COUNTS if summary[name] != 0
    ]
    summary["passed"] = bool(total) and not failures
    summary["failures"] = failures
    return summary


# --- runs ----------------------------------------------------------------------------


def _run(
    config: BatchConfig,
    batch_id: str,
    kind: str,
    body,
    *,
    allow_already_done: bool = True,
    run_id: str | None = None,
    expected_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Lock, track status, and turn any failure into a coded BatchError."""
    _check_batch_id(batch_id)
    run_id = run_id or uuid.uuid4().hex
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        raise BatchError("CONFIG_INVALID", "run_id must contain only letters, digits, dot, underscore or hyphen")
    base_work = config.root("work_root") / "batches" / batch_id / kind
    work = base_work / "runs" / run_id if kind in {"retrain", "inference"} else base_work
    status = BatchStatus(work / "status.json", batch_id, kind, run_id=run_id)
    lock_key = f"{kind}-{batch_id}-{run_id}" if kind in {"retrain", "inference"} else f"{kind}-{batch_id}"
    with BatchLock(config.root("work_root") / "locks", lock_key):
        previous = status.read()
        if previous and expected_details:
            recorded = previous.get("details", {})
            for key, value in expected_details.items():
                if key in recorded and recorded[key] != value:
                    raise BatchError(
                        "CONFIG_INVALID",
                        f"run_id {run_id} was already used with a different {key}",
                    )
        if allow_already_done and previous and previous["phase"] == status.phases[-1]:
            status.log("already_done", phase=previous["phase"])
            return {"status": "already_done", "run_id": run_id, **previous["details"]}
        status.begin()
        try:
            result = body(work, status)
            result.setdefault("run_id", run_id)
            if kind in {"retrain", "inference"}:
                atomic_write_text(base_work / "status.json", json.dumps(status.read(), indent=2, sort_keys=True))
                atomic_write_text(base_work / "latest_run.json", json.dumps({"run_id": run_id}, sort_keys=True))
            return result
        except BatchError as error:
            status.fail(error)
            if kind in {"retrain", "inference"}:
                atomic_write_text(base_work / "status.json", json.dumps(status.read(), indent=2, sort_keys=True))
                atomic_write_text(base_work / "latest_run.json", json.dumps({"run_id": run_id}, sort_keys=True))
            raise
        except Exception as error:
            code = PHASE_ERROR_CODES.get(status.phase or "", "INFERENCE_FAILED")
            if isinstance(error, MemoryError):
                code = "CAPACITY"
            wrapped = BatchError(code, f"{type(error).__name__}: {error}")
            status.fail(wrapped)
            if kind in {"retrain", "inference"}:
                atomic_write_text(base_work / "status.json", json.dumps(status.read(), indent=2, sort_keys=True))
                atomic_write_text(base_work / "latest_run.json", json.dumps({"run_id": run_id}, sort_keys=True))
            raise wrapped from error


def run_inference_batch(
    config: BatchConfig,
    batch_id: str,
    data_end_at: str | None = None,
    *,
    run_id: str | None = None,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    from recsys_vnpt.canonical import normalize_data_end_at

    cutoff = normalize_data_end_at(data_end_at) if data_end_at is not None else None

    def body(work: Path, status: BatchStatus) -> dict[str, Any]:
        timings: dict[str, float] = {}
        started = time.perf_counter()
        delivery = config.root("delivery_root")
        wait_for_ready(config.root("incoming_root") / batch_id, config.input_wait_seconds, config.poll_seconds)
        status.transition("VALIDATING_INPUT")
        manifest = _validate_input(config, batch_id, verify_checksums=False)
        try:
            current = read_current_delivery(delivery)
        except (OSError, ValueError, KeyError) as error:
            raise BatchError("PUBLISH_FAILED", f"current delivery is unreadable: {error}") from error
        if current and current.get("run_id") == status.run_id:
            if current.get("data_end_at") != (cutoff.isoformat() if cutoff is not None else None):
                raise BatchError("CONFIG_INVALID", "run_id was already published with a different data_end_at")
            status.transition("PUBLISHED", generation=current["generation"], recovered=True)
            return {"status": "already_published", "run_id": status.run_id,
                    "generation": current["generation"], "files": current.get("files", {})}
        current_batch_started_at = batch_started_at_from_metadata(current) if current else None
        if current and current.get("batch_id") != batch_id and current_batch_started_at and current_batch_started_at >= manifest.batch_started_at:
            raise BatchError(
                "INPUT_INTEGRITY",
                f"batch_started_at {manifest.batch_started_at.isoformat()} does not advance past the "
                f"published {current_batch_started_at.isoformat()} (batch {current['batch_id']})",
            )
        timings["validate_seconds"] = time.perf_counter() - started

        status.transition("LOADING_PREPARED_DATA", input_fingerprint=manifest.fingerprint)
        stage_started = time.perf_counter()
        prepared = load_prepared_dataset(config, manifest)
        data_root = prepared.artifact_root
        timings["prepared_data_validation_seconds"] = time.perf_counter() - stage_started
        status.log(
            "prepared_dataset_loaded",
            manifest=str(prepared.manifest_path),
            artifact_key=prepared.manifest["artifact_key"],
        )

        with _roots(config.root("incoming_root") / batch_id, data_root, data_end_at=cutoff):
            normalized_model_path = (
                str(Path(model_path).expanduser().resolve()) if model_path is not None else None
            )
            status.transition(
                "LOADING_MODEL",
                data_end_at=cutoff.isoformat() if cutoff is not None else None,
                requested_model_path=normalized_model_path,
                model_source="explicit" if model_path is not None else "promoted",
            )
            bundle, bundle_info = _promoted(config, model_path=model_path)
            if cutoff is not None:
                trained_value = bundle_info.get("model_data_end_at") or bundle_info.get("model_serving_cutoff")
                trained_through = pd.Timestamp(trained_value) if trained_value else None
                if trained_through is not None and trained_through.tzinfo is not None:
                    trained_through = trained_through.tz_localize(None)
                if trained_through is not None and trained_through > cutoff:
                    raise BatchError(
                        "MODEL_UNAVAILABLE",
                        f"selected model was trained through {trained_through}; it contains data after "
                        f"the requested inference cutoff {cutoff}",
                    )
            profile_ids = sample_profiles(_profile_ids(config.domain, cutoff, status.log), config.max_profiles)
            status.transition(
                "RUNNING_SHARDS",
                profiles=len(profile_ids),
                num_shards=config.num_shards,
                workers=config.workers,
                **bundle_info,
            )
            stage_started = time.perf_counter()
            identity = {
                "input_fingerprint": manifest.fingerprint,
                "model_sha256": bundle_info["model_sha256"],
                "page_fingerprint": config.page_fingerprint(),
                "code_version": code_version(),
                "data_end_at": cutoff.isoformat() if cutoff is not None else None,
            }
            shards = run_shards(
                config, profile_ids, work / "shards", identity, bundle, True,
                status.log, data_end_at=cutoff,
            )
            timings["shards_seconds"] = time.perf_counter() - stage_started

        status.transition("MERGING")
        stage_started = time.perf_counter()
        generation = f"{batch_id}_{status.run_id}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        stage = delivery / "generations" / f".{generation}.stage"
        try:
            names = production_output_names(config.domain, manifest.batch_started_at.strftime("%Y%m%d"))
            merge_shards(
                [work / "shards" / f"shard_{index:05d}" for index in range(config.num_shards)],
                stage,
                names,
                set(profile_ids),
            )
            timings["merge_seconds"] = time.perf_counter() - stage_started

            status.transition("VALIDATING_OUTPUT")
            validation = aggregate_validation(shards)
            if not validation["passed"]:
                raise BatchError("OUTPUT_INVALID", f"hard output checks failed: {validation['failures']}")
            atomic_write_text(stage / "validation_report.json", json.dumps(validation, indent=2, sort_keys=True))

            status.transition("PUBLISHING")
            snapshot = commit_delivery_generation(
                delivery,
                stage,
                generation,
                {
                    "batch_id": batch_id,
                    "domain": config.domain,
                    "input_fingerprint": manifest.fingerprint,
                    "batch_started_at": manifest.batch_started_at.isoformat(),
                    "serving_cutoff": shards[0]["serving_cutoff"] if shards else None,
                    "data_end_at": cutoff.isoformat() if cutoff is not None else None,
                    "run_id": status.run_id,
                    "profiles": len(profile_ids),
                    "page_fingerprint": config.page_fingerprint(),
                    "code_version": identity["code_version"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    **bundle_info,
                },
            )
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        timings["total_seconds"] = time.perf_counter() - started
        report = {
            "status": "published",
            "run_id": status.run_id,
            "data_end_at": cutoff.isoformat() if cutoff is not None else None,
            "serving_cutoff": shards[0]["serving_cutoff"] if shards else None,
            "generation": generation,
            "files": snapshot["files"],
            "profiles": len(profile_ids),
            "shards_reused": sum(bool(shard.get("reused")) for shard in shards),
            "validation": validation,
            "timings": timings,
            "sla_seconds": config.sla_seconds,
            "sla_met": timings["total_seconds"] <= config.sla_seconds,
            **bundle_info,
        }
        atomic_write_text(work / "report.json", json.dumps(report, indent=2, sort_keys=True))
        status.transition("PUBLISHED", generation=generation, sla_met=report["sla_met"])
        return report

    return _run(
        config, batch_id, "inference", body, run_id=run_id,
        expected_details={
            "data_end_at": cutoff.isoformat() if cutoff is not None else None,
            "requested_model_path": str(Path(model_path).expanduser().resolve()) if model_path is not None else None,
        },
    )


def _set_environment(values: dict[str, str]) -> None:
    os.environ.update(values)


def _evaluate(config: BatchConfig, bundle: Path | None = None) -> dict[str, Any]:
    """Test-window quality of one ordering, in a process of its own.

    By the time the gate runs, the retrain process carries the heap of fitting
    both scorers and of the evaluations before this one; stacking another
    evaluation on top exhausted a 17 GB host that runs any one of them alone
    with room to spare. A fresh process peaks at one evaluation.
    """
    roots = {
        name: os.environ[name]
        for name in ("RECSYS_VNPT_SOURCE_ROOT", "RECSYS_VNPT_DATA_ROOT", "RECSYS_VNPT_TEMPORAL_ROOT", "RECSYS_VNPT_DATA_END_AT")
        if name in os.environ
    }
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_set_environment,
        initargs=(roots,),
    ) as pool:
        return pool.submit(_evaluate_here, config, bundle).result()


def _evaluate_here(config: BatchConfig, bundle: Path | None = None) -> dict[str, Any]:
    """Scored exactly like evaluate-baseline, on the gate's profile sample:
    the configured ordering served from ``bundle``, or the hand-built
    ordering without one."""
    from recsys_vnpt.evaluation_runs import evaluate_popularity_baseline
    from recsys_vnpt.orderings import OrderingConfig

    rule_model = content_model = None
    fcn: dict[str, Any] = {}
    ordering = OrderingConfig()
    if bundle is not None:
        rule_model, content_model, _ = load_bundle(bundle)
        fcn = load_bundle_fcn(bundle, config.ordering.fcn_families)
        ordering = config.ordering
    result = evaluate_popularity_baseline(
        PROJECT_ROOT,
        config.domain,
        candidate_budget=config.candidate_budget,
        evaluation_split="test",
        maximum_evaluation_profiles=config.gate_evaluation_profiles,
        recency_candidates_per_rule=config.recency_candidates_per_rule,
        recency_reserved_per_rule=config.recency_reserved_per_rule,
        recency_max_age_days=config.recency_max_age_days,
        candidate_rules=config.candidate_rules,
        reserved_per_rule=config.reserved_per_rule,
        source_candidates_per_rule=config.source_candidates_per_rule,
        rule_ordering=ordering.rule_ordering,
        rule_scorer_fusion_weight=ordering.rule_scorer_fusion_weight,
        content_ordering=ordering.content_ordering,
        content_scorer_fusion_weight=ordering.content_scorer_fusion_weight,
        content_rank_scope=ordering.content_rank_scope,
        content_fcn_family=ordering.content_fcn_family,
        routed_fallback=ordering.routed_fallback,
        contents_per_rule_max=config.contents_per_rule_max,
        page_content_budget=config.page_content_budget,
        page_dedup=config.page_dedup,
        rule_scorer_model=rule_model,
        content_scorer_model=content_model,
        fcn_models=fcn,
        gbdt_backend=config.gbdt_backend,
        gbdt_device=config.gbdt_device,
    )
    return {
        "run_id": result.manifest.run_id,
        "profiles_evaluated": int(result.evaluation.get("profiles_evaluated", 0)),
        "novel_positive_pairs": int(result.evaluation.get("novel_positive_pairs", 0)),
        **{name: float(result.evaluation[name]) for name in GATE_METRICS},
    }


def retrain_gate(
    candidate: dict[str, float],
    promoted: dict[str, float] | None,
    max_regression: float,
    baseline: dict[str, float] | None = None,
) -> dict[str, Any]:
    """A new bundle may not fall more than ``max_regression`` below the
    promoted one on either primary metric. A test window without new
    positives proves nothing, so it fails.

    The comparison with the hand-built ordering is switched off (2026-11-10):
    combinations are now chosen by config, and some of them are meant to be
    served and measured even where they rank below the heuristic.
    """
    checks = [
        {
            "check": "test window has novel positives",
            "actual": candidate.get("novel_positive_pairs", 0),
            "threshold": 1,
            "passed": candidate.get("novel_positive_pairs", 0) >= 1,
        }
    ]
    for name in GATE_METRICS:
        # checks.append(
        #     {
        #         "check": f"{name} >= hand-built baseline",
        #         "actual": candidate[name],
        #         "threshold": baseline[name],
        #         "passed": candidate[name] >= baseline[name],
        #     }
        # )
        if promoted is not None:
            threshold = promoted[name] * (1.0 - max_regression)
            checks.append(
                {
                    "check": f"{name} >= promoted * (1 - {max_regression})",
                    "actual": candidate[name],
                    "threshold": threshold,
                    "passed": candidate[name] >= threshold,
                }
            )
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "baseline": baseline,
        "candidate": candidate,
        "promoted": promoted,
    }


def run_retrain_batch(
    config: BatchConfig,
    batch_id: str,
    data_end_at: str | None = None,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    from recsys_vnpt.canonical import normalize_data_end_at
    from recsys_vnpt.source_manifest import stable_hash

    from recsys_vnpt.content_scorer import train_content_scorer
    from recsys_vnpt.orderings import fit_fcn
    from recsys_vnpt.rule_scorer import train_rule_scorer
    from recsys_vnpt.serving import _interactions_loader

    data_end = normalize_data_end_at(data_end_at) if data_end_at is not None else None

    def body(work: Path, status: BatchStatus) -> dict[str, Any]:
        started = time.perf_counter()
        wait_for_ready(config.root("incoming_root") / batch_id, config.input_wait_seconds, config.poll_seconds)
        status.transition("VALIDATING_INPUT")
        manifest = _validate_input(config, batch_id, verify_checksums=False)
        status.transition("LOADING_PREPARED_DATA", input_fingerprint=manifest.fingerprint)
        prepared = load_prepared_dataset(config, manifest)
        data_root = prepared.artifact_root
        status.log(
            "prepared_dataset_loaded",
            manifest=str(prepared.manifest_path),
            artifact_key=prepared.manifest["artifact_key"],
        )

        temporal_root = None
        if data_end is not None:
            temporal_key = stable_hash({
                "input_fingerprint": manifest.fingerprint,
                "data_end_at": data_end.isoformat(),
                "domain": config.domain,
                "time_train": config.time_train,
                "time_valid": config.time_valid,
                "time_test": config.time_test,
            })
            temporal_root = config.root("work_root") / "temporal" / temporal_key

        with _roots(config.root("incoming_root") / batch_id, data_root, temporal_root, data_end):
            canonical = data_artifact_path("canonical", config.domain, project_root=PROJECT_ROOT)
            status.transition("TRAINING", data_end_at=data_end.isoformat() if data_end is not None else None)
            loader = _interactions_loader(canonical, data_end, status.log)
            if data_end is not None:
                from recsys_vnpt.pipeline_v2 import build_temporal_project

                status.log("manual_temporal_build", status="started", data_end_at=data_end.isoformat(),
                           temporal_cache=str(temporal_root))
                temporal_lock = stable_hash(str(temporal_root))[:16]
                with _waiting_lock(
                    config.root("work_root") / "locks",
                    f"temporal-{temporal_lock}",
                    config.input_wait_seconds,
                    config.poll_seconds,
                ):
                    build_temporal_project(
                        PROJECT_ROOT,
                        config.domain,
                        window_mode="relative",
                        time_train=config.time_train,
                        time_valid=config.time_valid,
                        time_test=config.time_test,
                        data_end_at=data_end,
                        temporal_root=temporal_root,
                        progress_callback=status.log,
                    )
                status.log("manual_temporal_build", status="completed", temporal_cache=str(temporal_root))
            contents = pd.read_parquet(canonical / "contents.parquet", columns=["content_id"])
            # The serving cutoff of this data, computed as the serving context
            # does, so the category levels match what inference will see.
            serving_cutoff = pd.to_datetime(
                loader(set(contents["content_id"].astype(str)), ["create_date"])["create_date"],
                errors="coerce",
            ).max()
            try:
                status.log("scorer_training", scorer="rule", status="started")
                rule_model = train_rule_scorer(
                    canonical, loader, backend=config.gbdt_backend, device=config.gbdt_device
                )
                status.log("scorer_training", scorer="rule", status="completed")
                status.log("scorer_training", scorer="content", status="started")
                content_model = train_content_scorer(
                    canonical, pd.Timestamp(serving_cutoff), loader,
                    backend=config.gbdt_backend, device=config.gbdt_device,
                )
                status.log("scorer_training", scorer="content", status="completed")
                fcn_dirs = {}
                for family in config.ordering.fcn_families:
                    status.log("fcn_training", family=family, epochs=config.fcn_epochs)
                    fcn_dirs[family] = work / "fcn" / family
                    fit_fcn(
                        canonical, config.domain, family, fcn_dirs[family],
                        epochs=config.fcn_epochs, schedule_epochs=config.fcn_schedule_epochs,
                    )
                    status.log("fcn_training", family=family, status="completed")
            except Exception as error:
                raise BatchError("RETRAIN_FAILED", f"{type(error).__name__}: {error}") from error
            bundle = save_bundle(
                config.root("work_root") / "models",
                rule_model,
                content_model,
                {
                    "batch_id": batch_id,
                    "input_fingerprint": manifest.fingerprint,
                    "batch_started_at": manifest.batch_started_at.isoformat(),
                    "data_end_at": data_end.isoformat() if data_end is not None else None,
                    "serving_cutoff": str(serving_cutoff),
                    "code_version": code_version(),
                    "ordering": asdict(config.ordering),
                },
                fcn_dirs=fcn_dirs,
            )
            status.transition("EVALUATING", bundle=str(bundle))
            # Off since 2026-11-10, see retrain_gate.
            # baseline = _evaluate(config)
            candidate = _evaluate(config, bundle)
            promoted = None
            current = promoted_bundle(config.root("model_registry"))
            if current is not None:
                try:
                    # Scored with the configured ordering; a bundle that lacks a
                    # model it needs cannot be compared and is skipped.
                    promoted = _evaluate(config, current)
                except (BundleError, OSError) as error:
                    status.log("promoted_bundle_unusable", error=str(error))

        status.transition("GATING")
        gate = retrain_gate(candidate, promoted, config.gate_max_regression)
        atomic_write_text(work / "gate.json", json.dumps(gate, indent=2, sort_keys=True))
        if not gate["passed"]:
            status.finish("REJECTED", gate_passed=False, bundle=str(bundle))
            raise BatchError(
                "QUALITY_GATE_FAILED",
                "; ".join(check["check"] for check in gate["checks"] if not check["passed"]),
            )
        promote(config.root("model_registry"), bundle, gate)
        report = {
            "status": "promoted",
            "bundle": str(bundle),
            "gate": gate,
            "seconds": time.perf_counter() - started,
        }
        atomic_write_text(work / "report.json", json.dumps(report, indent=2, sort_keys=True))
        status.transition("PROMOTED", bundle=str(bundle))
        return report

    return _run(
        config, batch_id, "retrain", body, run_id=run_id,
        expected_details={"data_end_at": data_end.isoformat() if data_end is not None else None},
    )


class _PeakMemory:
    """Samples the resident memory of this process and its workers."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        import psutil

        process = psutil.Process()
        while not self._stop.is_set():
            try:
                total = process.memory_info().rss + sum(
                    child.memory_info().rss for child in process.children(recursive=True)
                )
            except psutil.Error:
                total = 0
            self.peak = max(self.peak, total)
            self._stop.wait(self.interval)

    def __enter__(self) -> "_PeakMemory":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()


def qualify_hardware(config: BatchConfig, batch_id: str) -> dict[str, Any]:
    """Measure full inference cycles at each configured workload, without
    publishing. A workload larger than the input's profile count is reported as
    not measurable rather than extrapolated."""
    import psutil

    def body(work: Path, status: BatchStatus) -> dict[str, Any]:
        wait_for_ready(config.root("incoming_root") / batch_id, config.input_wait_seconds, config.poll_seconds)
        status.transition("VALIDATING_INPUT")
        manifest = _validate_input(config, batch_id, verify_checksums=False)
        status.transition("LOADING_PREPARED_DATA", input_fingerprint=manifest.fingerprint)
        prepared = load_prepared_dataset(config, manifest)
        data_root = prepared.artifact_root
        prepare_seconds = 0.0
        status.log(
            "prepared_dataset_loaded",
            manifest=str(prepared.manifest_path),
            artifact_key=prepared.manifest["artifact_key"],
        )
        results = []
        with _roots(config.root("incoming_root") / batch_id, data_root):
            status.transition("LOADING_MODEL")
            bundle, bundle_info = _promoted(config)
            # Hash order spreads any workload evenly over the population.
            ordered = sorted(
                _profile_ids(config.domain),
                key=lambda pid: hashlib.sha256(pid.encode("utf-8")).digest(),
            )
            status.transition("MEASURING", available_profiles=len(ordered))
            names = production_output_names(config.domain, manifest.batch_started_at.strftime("%Y%m%d"))
            identity = {"qualification": True, "page_fingerprint": config.page_fingerprint()}
            for workload in sorted(config.qualification_workloads):
                if workload > len(ordered):
                    results.append(
                        {"workload": workload, "status": "not_measurable", "available_profiles": len(ordered)}
                    )
                    continue
                subset = ordered[:workload]
                runs = []
                for repetition in range(config.qualification_repetitions):
                    scratch = work / "runs" / f"{workload}_{repetition}"
                    shutil.rmtree(scratch, ignore_errors=True)
                    with _PeakMemory() as memory:
                        run_started = time.perf_counter()
                        run_shards(config, subset, scratch / "shards", identity, bundle, False)
                        shards_seconds = time.perf_counter() - run_started
                        merge_shards(
                            [scratch / "shards" / f"shard_{index:05d}" for index in range(config.num_shards)],
                            scratch / "merged",
                            names,
                            set(subset),
                        )
                        seconds = time.perf_counter() - run_started
                    runs.append(
                        {
                            "seconds": seconds,
                            "shards_seconds": shards_seconds,
                            "peak_rss_bytes": memory.peak,
                            "output_sha256": {
                                name: sha256_file(scratch / "merged" / name) for name in names.values()
                            },
                        }
                    )
                    status.log("measured", workload=workload, repetition=repetition, seconds=seconds)
                    shutil.rmtree(scratch, ignore_errors=True)
                seconds = sorted(run["seconds"] for run in runs)
                results.append(
                    {
                        "workload": workload,
                        "status": "measured",
                        "seconds": seconds,
                        "median_seconds": seconds[len(seconds) // 2],
                        "max_seconds": seconds[-1],
                        "profiles_per_second": workload / seconds[len(seconds) // 2],
                        "peak_rss_bytes": max(run["peak_rss_bytes"] for run in runs),
                        "deterministic": len({json.dumps(run["output_sha256"], sort_keys=True) for run in runs}) == 1,
                    }
                )
        target = max(config.qualification_workloads)
        target_result = next(item for item in results if item["workload"] == target)
        total_memory = psutil.virtual_memory().total
        disk_free = shutil.disk_usage(config.root("work_root")).free
        if target_result["status"] != "measured":
            verdict = "not_measured"
        else:
            cycle = prepare_seconds + target_result["max_seconds"]
            verdict = (
                "pass"
                if cycle <= config.sla_seconds
                and target_result["deterministic"]
                and target_result["peak_rss_bytes"] <= 0.8 * total_memory
                and disk_free >= config.min_free_bytes
                else "fail"
            )
        report = {
            "verdict": verdict,
            "target_workload": target,
            "sla_seconds": config.sla_seconds,
            "prepare_seconds": prepare_seconds,
            "results": results,
            "disk_free_bytes_after": disk_free,
            "hardware": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "logical_cpus": os.cpu_count(),
                "memory_bytes": total_memory,
            },
            "num_shards": config.num_shards,
            "workers": config.workers,
            "code_version": code_version(),
            **bundle_info,
        }
        atomic_write_text(work / "qualification_report.json", json.dumps(report, indent=2, sort_keys=True))
        status.transition("QUALIFIED", verdict=verdict)
        return report

    return _run(config, batch_id, "qualification", body)
