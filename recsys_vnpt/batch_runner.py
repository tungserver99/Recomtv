"""Production batch coordinator: inference, retraining, hardware qualification.

Inference turns one committed input batch into one committed delivery
generation. Profiles are split into shards by a stable hash of ``profile_id``;
each shard is computed, validated and written with a success record last, so a
rerun after a failure recomputes only the shards that are missing or do not
verify. Shards are merged by streaming them in profile order, which makes the
three delivered files byte-identical whatever the shard or worker count.

Retraining fits both learned scorers on the batch, gates the new bundle against
the hand-built ordering and the currently promoted bundle on the temporal test
window, and moves the promotion pointer only when the gate passes. Inference
never fits anything: it serves the promoted bundle or fails.
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
from recsys_vnpt.runtime import atomic_write_text
from recsys_vnpt.scorer_bundle import (
    BundleError,
    load_bundle,
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
def _roots(source: Path, data: Path) -> Iterator[None]:
    """Point the raw readers at the batch and generated artifacts at ``data``."""
    names = ("RECSYS_VNPT_SOURCE_ROOT", "RECSYS_VNPT_DATA_ROOT")
    saved = {name: os.environ.get(name) for name in names}
    os.environ["RECSYS_VNPT_SOURCE_ROOT"] = str(source)
    os.environ["RECSYS_VNPT_DATA_ROOT"] = str(data)
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


def _validate_input(config: BatchConfig, batch_id: str) -> ReadyManifest:
    manifest = load_and_validate_ready(config.root("incoming_root") / batch_id, batch_id)
    work = config.root("work_root")
    work.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(work).free
    if free < config.min_free_bytes:
        raise BatchError("CAPACITY", f"{free} bytes free under {work}, need {config.min_free_bytes}")
    return manifest


def _prepare(config: BatchConfig, manifest: ReadyManifest, temporal: bool) -> Path:
    """Canonical (and for retraining, temporal) artifacts for this input,
    cached by input fingerprint under the work root."""
    from recsys_vnpt.pipeline_v2 import build_temporal_project, prepare_canonical_project

    data_root = config.root("work_root") / "artifacts" / manifest.fingerprint[:24]
    source = config.root("incoming_root") / manifest.batch_id
    with _waiting_lock(
        config.root("work_root") / "locks",
        f"prepare-{manifest.fingerprint[:24]}",
        config.input_wait_seconds,
        config.poll_seconds,
    ), _roots(source, data_root):
        try:
            prepare_canonical_project(
                PROJECT_ROOT,
                config.domain,
                include_behavior=config.include_behavior,
                user_reference_date=pd.Timestamp(manifest.data_cutoff.date()),
            )
            if temporal:
                build_temporal_project(PROJECT_ROOT, config.domain, window_mode="relative")
        except MemoryError as error:
            raise BatchError("CAPACITY", "out of memory while preparing data") from error
        except BatchError:
            raise
        except Exception as error:
            raise BatchError("SCHEMA_DRIFT", f"{type(error).__name__}: {error}") from error
    return data_root


def _profile_ids(domain: str) -> list[str]:
    mapping = pd.read_parquet(
        data_artifact_path("canonical", domain, "profile_mapping.parquet", project_root=PROJECT_ROOT),
        columns=["profile_id"],
    )
    return mapping["profile_id"].astype(str).drop_duplicates().tolist()


def _promoted(config: BatchConfig) -> tuple[Path | None, dict[str, Any]]:
    """The promoted bundle, verified loadable, or no bundle when the
    configuration serves only the hand-built orderings."""
    if not config.uses_learned_scorers:
        return None, {"bundle_id": None, "model_sha256": None}
    bundle = promoted_bundle(config.root("model_registry"))
    if bundle is None:
        raise BatchError(
            "MODEL_UNAVAILABLE",
            "no promoted scorer bundle; run run-retrain first, or configure "
            "rule_ordering=count and content_ordering=hand_sum",
        )
    try:
        _, _, manifest = load_bundle(bundle)
    except (BundleError, OSError) as error:
        raise BatchError("MODEL_UNAVAILABLE", str(error)) from error
    return bundle, {"bundle_id": manifest["bundle_id"], "model_sha256": manifest["model_sha256"]}


# --- shards ----------------------------------------------------------------------


def shard_of(profile_id: str, num_shards: int) -> int:
    digest = hashlib.sha256(str(profile_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def assign_shards(profile_ids: list[str], num_shards: int) -> list[list[str]]:
    shards: list[list[str]] = [[] for _ in range(num_shards)]
    for profile_id in sorted(set(map(str, profile_ids))):
        shards[shard_of(profile_id, num_shards)].append(profile_id)
    return shards


def _serving_context(config: BatchConfig, bundle: str | None):
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
    )
    rule_model = content_model = None
    if bundle is not None:
        rule_model, content_model, _ = load_bundle(bundle)
        serve_on(rule_model, config.gbdt_device)
        serve_on(content_model, config.gbdt_device)
    context = with_scorers(
        context,
        rule_model if config.rule_ordering == "learned_rrf" else None,
        content_model if config.content_ordering == "learned_rrf" else None,
        config.rule_scorer_fusion_weight,
        config.content_scorer_fusion_weight,
    )
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


def _init_worker(config_fields: dict[str, Any], bundle: str | None, roots: dict[str, str]) -> None:
    os.environ.update(roots)
    config = BatchConfig(**config_fields)
    _WORKER["config"] = config
    _WORKER["context"], _WORKER["member_ids"] = _serving_context(config, bundle)


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
) -> list[tuple[int, BaseException]]:
    """One pass over ``pending``; returns the shards that failed for reasons a
    retry can fix. A contract failure (BatchError) is never retried."""
    failures: list[tuple[int, BaseException]] = []
    if config.workers == 1:
        try:
            context, member_ids = _serving_context(config, bundle_path)
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
        for name in ("RECSYS_VNPT_SOURCE_ROOT", "RECSYS_VNPT_DATA_ROOT")
        if name in os.environ
    }
    # Spawn, not fork: forking while another thread (the qualification memory
    # sampler) holds a lock can deadlock the child. A worker that dies breaks
    # the whole pool, so every shard it leaves unfinished is retried together.
    with ProcessPoolExecutor(
        max_workers=min(config.workers, len(pending)),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(asdict(config), bundle_path, roots),
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
        failures = _compute_shards(config, pending, bundle_path, results, log)
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


def _run(config: BatchConfig, batch_id: str, kind: str, body) -> dict[str, Any]:
    """Lock, track status, and turn any failure into a coded BatchError."""
    _check_batch_id(batch_id)
    work = config.root("work_root") / "batches" / batch_id / kind
    status = BatchStatus(work / "status.json", batch_id, kind)
    with BatchLock(config.root("work_root") / "locks", f"{kind}-{batch_id}"):
        previous = status.read()
        if previous and previous["phase"] == status.phases[-1]:
            status.log("already_done", phase=previous["phase"])
            return {"status": "already_done", **previous["details"]}
        status.begin()
        try:
            return body(work, status)
        except BatchError as error:
            status.fail(error)
            raise
        except Exception as error:
            code = PHASE_ERROR_CODES.get(status.phase or "", "INFERENCE_FAILED")
            if isinstance(error, MemoryError):
                code = "CAPACITY"
            wrapped = BatchError(code, f"{type(error).__name__}: {error}")
            status.fail(wrapped)
            raise wrapped from error


def run_inference_batch(config: BatchConfig, batch_id: str) -> dict[str, Any]:
    def body(work: Path, status: BatchStatus) -> dict[str, Any]:
        timings: dict[str, float] = {}
        started = time.perf_counter()
        delivery = config.root("delivery_root")
        wait_for_ready(config.root("incoming_root") / batch_id, config.input_wait_seconds, config.poll_seconds)
        status.transition("VALIDATING_INPUT")
        manifest = _validate_input(config, batch_id)
        try:
            current = read_current_delivery(delivery)
        except (OSError, ValueError, KeyError) as error:
            raise BatchError("PUBLISH_FAILED", f"current delivery is unreadable: {error}") from error
        if current and current["batch_id"] == batch_id:
            # Committed by an earlier attempt that died before recording it.
            status.transition("PUBLISHED", generation=current["generation"])
            return {"status": "already_published", "generation": current["generation"]}
        if current and datetime.fromisoformat(current["data_cutoff"]) >= manifest.data_cutoff:
            raise BatchError(
                "INPUT_INTEGRITY",
                f"data_cutoff {manifest.data_cutoff.isoformat()} does not advance past the "
                f"published {current['data_cutoff']} (batch {current['batch_id']})",
            )
        timings["validate_seconds"] = time.perf_counter() - started

        status.transition("PREPARING_DATA", input_fingerprint=manifest.fingerprint)
        stage_started = time.perf_counter()
        data_root = _prepare(config, manifest, temporal=False)
        timings["prepare_seconds"] = time.perf_counter() - stage_started

        with _roots(config.root("incoming_root") / batch_id, data_root):
            status.transition("LOADING_MODEL")
            bundle, bundle_info = _promoted(config)
            profile_ids = _profile_ids(config.domain)
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
            }
            shards = run_shards(config, profile_ids, work / "shards", identity, bundle, True, status.log)
            timings["shards_seconds"] = time.perf_counter() - stage_started

        status.transition("MERGING")
        stage_started = time.perf_counter()
        generation = f"{batch_id}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        stage = delivery / "generations" / f".{generation}.stage"
        try:
            names = production_output_names(config.domain, manifest.data_cutoff.strftime("%Y%m%d"))
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
                    "data_cutoff": manifest.data_cutoff.isoformat(),
                    "serving_cutoff": shards[0]["serving_cutoff"] if shards else None,
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

    return _run(config, batch_id, "inference", body)


def _set_environment(values: dict[str, str]) -> None:
    os.environ.update(values)


def _evaluate(config: BatchConfig, rule_model=None, content_model=None) -> dict[str, Any]:
    """Test-window quality of one ordering, in a process of its own.

    By the time the gate runs, the retrain process carries the heap of fitting
    both scorers and of the evaluations before this one; stacking another
    evaluation on top exhausted a 17 GB host that runs any one of them alone
    with room to spare. A fresh process peaks at one evaluation.
    """
    roots = {
        name: os.environ[name]
        for name in ("RECSYS_VNPT_SOURCE_ROOT", "RECSYS_VNPT_DATA_ROOT")
        if name in os.environ
    }
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_set_environment,
        initargs=(roots,),
    ) as pool:
        return pool.submit(_evaluate_here, config, rule_model, content_model).result()


def _evaluate_here(config: BatchConfig, rule_model=None, content_model=None) -> dict[str, Any]:
    """Scored exactly like evaluate-baseline, on the gate's profile sample."""
    from recsys_vnpt.evaluation_runs import evaluate_popularity_baseline

    learned = rule_model is not None
    bundle = evaluate_popularity_baseline(
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
        rule_ordering="learned_rrf" if learned else "count",
        rule_scorer_fusion_weight=config.rule_scorer_fusion_weight,
        content_ordering="learned_rrf" if learned else "hand_sum",
        content_scorer_fusion_weight=config.content_scorer_fusion_weight,
        contents_per_rule_max=config.contents_per_rule_max,
        page_content_budget=config.page_content_budget,
        page_dedup=config.page_dedup,
        rule_scorer_model=rule_model,
        content_scorer_model=content_model,
    )
    return {
        "run_id": bundle.manifest.run_id,
        "profiles_evaluated": int(bundle.evaluation.get("profiles_evaluated", 0)),
        "novel_positive_pairs": int(bundle.evaluation.get("novel_positive_pairs", 0)),
        **{name: float(bundle.evaluation[name]) for name in GATE_METRICS},
    }


def retrain_gate(
    baseline: dict[str, float],
    candidate: dict[str, float],
    promoted: dict[str, float] | None,
    max_regression: float,
) -> dict[str, Any]:
    """A new bundle must beat the hand-built ordering on both primary metrics
    and may not fall more than ``max_regression`` below the promoted one. A
    test window without new positives proves nothing, so it fails."""
    checks = [
        {
            "check": "test window has novel positives",
            "actual": candidate.get("novel_positive_pairs", 0),
            "threshold": 1,
            "passed": candidate.get("novel_positive_pairs", 0) >= 1,
        }
    ]
    for name in GATE_METRICS:
        checks.append(
            {
                "check": f"{name} >= hand-built baseline",
                "actual": candidate[name],
                "threshold": baseline[name],
                "passed": candidate[name] >= baseline[name],
            }
        )
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


def run_retrain_batch(config: BatchConfig, batch_id: str) -> dict[str, Any]:
    from recsys_vnpt.content_scorer import train_content_scorer
    from recsys_vnpt.rule_scorer import train_rule_scorer
    from recsys_vnpt.serving import _interactions_loader

    def body(work: Path, status: BatchStatus) -> dict[str, Any]:
        started = time.perf_counter()
        wait_for_ready(config.root("incoming_root") / batch_id, config.input_wait_seconds, config.poll_seconds)
        status.transition("VALIDATING_INPUT")
        manifest = _validate_input(config, batch_id)
        status.transition("PREPARING_DATA", input_fingerprint=manifest.fingerprint)
        data_root = _prepare(config, manifest, temporal=True)

        with _roots(config.root("incoming_root") / batch_id, data_root):
            status.transition("TRAINING")
            canonical = data_artifact_path("canonical", config.domain, project_root=PROJECT_ROOT)
            loader = _interactions_loader(canonical)
            contents = pd.read_parquet(canonical / "contents.parquet", columns=["content_id"])
            # The serving cutoff of this data, computed as the serving context
            # does, so the category levels match what inference will see.
            cutoff = pd.to_datetime(
                loader(set(contents["content_id"].astype(str)), ["create_date"])["create_date"],
                errors="coerce",
            ).max()
            try:
                rule_model = train_rule_scorer(
                    canonical, loader, backend=config.gbdt_backend, device=config.gbdt_device
                )
                content_model = train_content_scorer(
                    canonical, pd.Timestamp(cutoff), loader,
                    backend=config.gbdt_backend, device=config.gbdt_device,
                )
            except Exception as error:
                raise BatchError("RETRAIN_FAILED", f"{type(error).__name__}: {error}") from error
            bundle = save_bundle(
                config.root("work_root") / "models",
                rule_model,
                content_model,
                {
                    "batch_id": batch_id,
                    "input_fingerprint": manifest.fingerprint,
                    "data_cutoff": manifest.data_cutoff.isoformat(),
                    "serving_cutoff": str(cutoff),
                    "code_version": code_version(),
                },
            )
            status.transition("EVALUATING", bundle=str(bundle))
            baseline = _evaluate(config)
            candidate = _evaluate(config, rule_model, content_model)
            promoted = None
            current = promoted_bundle(config.root("model_registry"))
            if current is not None:
                try:
                    previous_rule, previous_content, _ = load_bundle(current)
                    promoted = _evaluate(config, previous_rule, previous_content)
                except (BundleError, OSError) as error:
                    status.log("promoted_bundle_unusable", error=str(error))

        status.transition("GATING")
        gate = retrain_gate(baseline, candidate, promoted, config.gate_max_regression)
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

    return _run(config, batch_id, "retrain", body)


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
        manifest = _validate_input(config, batch_id)
        status.transition("PREPARING_DATA", input_fingerprint=manifest.fingerprint)
        prepare_started = time.perf_counter()
        data_root = _prepare(config, manifest, temporal=False)
        prepare_seconds = time.perf_counter() - prepare_started
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
            names = production_output_names(config.domain, manifest.data_cutoff.strftime("%Y%m%d"))
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
