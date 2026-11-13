from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from production.config import ProductionConfig
from production.cutoff import batch_id_for_slot, resolve_slot
from production.staging import StagingResult, build_input_snapshot
from production.state_store import StateStore
from recsys_vnpt.batch_contracts import BatchError, batch_started_at_from_metadata
from recsys_vnpt.batch_state import BatchLock
from recsys_vnpt.runtime import atomic_write_text


def _pipeline_state(config: ProductionConfig, batch_id: str, kind: str) -> dict[str, Any] | None:
    path = config.work_root / "batches" / batch_id / kind / "status.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"phase": "UNKNOWN", "details": {}, "history": []}


def _json_if_exists(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _report_path(config: ProductionConfig, batch_id: str) -> Path:
    return config.work_root / "batches" / batch_id / "production" / "report.json"


def _log_path(config: ProductionConfig, batch_id: str) -> Path:
    return config.work_root / "batches" / batch_id / "production" / "log.jsonl"


def _append_job_log(config: ProductionConfig, batch_id: str, event: str, **fields: Any) -> None:
    path = _log_path(config, batch_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"at": datetime.now(timezone.utc).isoformat(), "batch_id": batch_id, "kind": "production", "event": event, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")


def run_production(
    config: ProductionConfig,
    slot: datetime | str | None = None,
    *,
    store: StateStore | None = None,
    prepare_only: bool = False,
    data_root: str | Path | None = None,
    source_builder: Callable[[ProductionConfig, str, datetime], StagingResult] = build_input_snapshot,
    retrain: Callable[[Any, str], dict[str, Any]] | None = None,
    infer: Callable[[Any, str], dict[str, Any]] | None = None,
    promoted_resolver: Callable[[Any], tuple[Path | None, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Run one stable schedule slot using the existing batch training and serving code."""
    default_schedule = {
        "enabled": config.schedule_enabled,
        "run_times": list(config.run_times),
        "timezone": config.timezone,
    }
    state = store or StateStore(config.database_path)
    selected_data_root = data_root if data_root is not None else state.get_data_root(config.data_root)
    source_config = replace(config, data_root=config.resolve_data_root(selected_data_root))
    schedule = state.get_schedule(default_schedule)
    scheduled_at = resolve_slot(slot, schedule["timezone"], schedule["run_times"])
    batch_id = batch_id_for_slot(scheduled_at)
    previous = state.get_job(batch_id)
    if previous and previous["status"] == "SUCCESS":
        return {**previous["report"], "already_completed": True}
    batch_started_at = (
        datetime.fromisoformat(previous["batch_started_at"])
        if previous
        else datetime.now(timezone.utc)
    )

    report: dict[str, Any] = {
        "batch_id": batch_id,
        "scheduled_at": scheduled_at.isoformat(),
        "batch_started_at": batch_started_at.isoformat(),
        "actual_start": datetime.now(timezone.utc).isoformat(),
        "actual_end": None,
        "input_status": "PENDING",
        "input_fingerprint": None,
        "source_fingerprint": None,
        "data_root": str(source_config.data_root),
        "preparation_status": "PENDING",
        "retrain_status": "PENDING",
        "quality_gate_result": "PENDING",
        "gate_metrics": None,
        "model_used": None,
        "model_bundle_id": None,
        "inference_status": "PENDING",
        "publish_status": "PENDING",
        "output_generation": None,
        "output_files": [],
        "output_location": str(config.delivery_root),
        "steps": [],
    }
    if previous and previous["status"] == "PREPARED":
        for key in (
            "preparation_status",
            "preparation_result",
            "prepared_dataset",
        ):
            if key in previous["report"]:
                report[key] = previous["report"][key]
    started = time.perf_counter()
    claimed = False
    preserve_existing_report = False
    lock = None
    try:
        try:
            candidate_lock = BatchLock(config.work_root / "locks", "production-global")
            candidate_lock.__enter__()
            lock = candidate_lock
        except BatchError as error:
            state.record_skipped(batch_id, scheduled_at, batch_started_at, error.message)
            raise
        try:
            if not state.try_start_job(batch_id, scheduled_at, batch_started_at):
                current = state.current_job()
                if current and current["batch_id"] != batch_id:
                    state.record_skipped(batch_id, scheduled_at, batch_started_at, "another production job is active")
                    raise BatchError("BATCH_LOCKED", "another production job is active; this slot was skipped")
                previous = next((job for job in state.jobs(limit=500) if job["batch_id"] == batch_id), None)
                if previous and previous["status"] == "SUCCESS":
                    report.update(previous["report"])
                    report["already_completed"] = True
                    report["status"] = "SUCCESS"
                    preserve_existing_report = True
                    return report
                raise BatchError("BATCH_LOCKED", "production job could not be claimed")
            claimed = True
            batch_config = config.batch
            settings_defaults = {
                name: getattr(batch_config, name)
                for name in (
                    "time_train", "time_valid", "time_test", "max_profiles",
                    "gate_evaluation_profiles",
                )
            }
            data_settings = state.get_data_settings(settings_defaults)
            batch_config = replace(batch_config, **data_settings)
            report["data_settings"] = data_settings
            _append_job_log(config, batch_id, "data_settings", **data_settings)

            def update(phase: str) -> None:
                report["current_phase"] = phase
                state.update_job(batch_id, report, "RUNNING")
                _append_job_log(config, batch_id, "phase", phase=phase)

            update("CHECKING_PUBLISHED_SLOT")

            from recsys_vnpt.export import read_current_delivery

            current_delivery = read_current_delivery(config.delivery_root)
            if current_delivery and current_delivery.get("batch_id") == batch_id:
                report.update(
                    input_status="PUBLISHED_VERIFIED",
                    input_fingerprint=current_delivery.get("input_fingerprint"),
                    model_used="CURRENT_PUBLISHED_MODEL",
                    model_bundle_id=current_delivery.get("bundle_id"),
                    inference_status="SUCCESS",
                    publish_status="SUCCESS",
                    output_generation=current_delivery.get("generation"),
                    output_files={
                        name: str(Path(current_delivery["directory"]) / name)
                        for name in current_delivery.get("files", {})
                    },
                    output_file_paths=[
                        str(Path(current_delivery["directory"]) / name)
                        for name in current_delivery.get("files", {})
                    ],
                    output_location=current_delivery["directory"],
                    status="SUCCESS",
                    already_published=True,
                    steps=[
                        {"step": "PREPARING_DATA", "status": "REUSED_PUBLISHED_SLOT"},
                        {"step": "RETRAINING", "status": "SKIPPED"},
                        {"step": "INFERENCE", "status": "SKIPPED_ALREADY_PUBLISHED"},
                    ],
                )
                return report
            current_batch_started_at = batch_started_at_from_metadata(current_delivery) if current_delivery else None
            if current_batch_started_at and current_batch_started_at >= batch_started_at:
                raise BatchError(
                    "INPUT_INTEGRITY",
                    f"batch_started_at {batch_started_at.isoformat()} does not advance past published batch "
                    f"{current_delivery.get('batch_id')} at {current_batch_started_at.isoformat()}",
                )

            update("PREPARING_INPUT")
            _append_job_log(config, batch_id, "input_staging", status="started")
            if source_builder is build_input_snapshot:
                staged = source_builder(
                    source_config, batch_id, batch_started_at,
                    progress_callback=lambda event, **fields: _append_job_log(
                        config, batch_id, event, **fields
                    ),
                )
            else:
                staged = source_builder(config, batch_id, batch_started_at)
            report.update(
                input_status="VALIDATED",
                input_fingerprint=staged.fingerprint,
                source_fingerprint=staged.source_fingerprint,
                input_reused=staged.reused,
            )
            _append_job_log(
                config, batch_id, "input_staging", status="completed",
                reused=staged.reused, fingerprint=staged.fingerprint,
            )
            report["steps"].append({"step": "INPUT_STAGING", "status": "COMPLETED"})

            from recsys_vnpt import batch_runner

            if prepare_only:
                report["preparation_status"] = "PREPARING"
                update("PREPARING_DATA")
                prepared = batch_runner.run_prepare_data_batch(batch_config, batch_id)
                report["preparation_result"] = prepared
                report["preparation_status"] = "READY"
                report["prepared_dataset"] = prepared.get("prepared_dataset")
                report["status"] = "PREPARED"
                report["steps"].append({"step": "PREPARING_DATA", "status": "COMPLETED"})
                _append_job_log(
                    config,
                    batch_id,
                    "prepared_dataset",
                    status="READY",
                    manifest=report["prepared_dataset"],
                    reused=prepared.get("reused", False),
                )
                return report

            run_retrain = retrain or batch_runner.run_retrain_batch
            run_inference = infer or batch_runner.run_inference_batch
            resolve_promoted = promoted_resolver or batch_runner._promoted
            update("RETRAINING")
            try:
                retrain_result = run_retrain(batch_config, batch_id)
                report["retrain_result"] = retrain_result
                report["gate_metrics"] = retrain_result.get("gate")
                report["retrain_status"] = "PROMOTED"
                report["quality_gate_result"] = "PASS"
                report["model_used"] = "NEWLY_PROMOTED"
                report["model_bundle_id"] = Path(str(retrain_result.get("bundle", ""))).name or None
                report["steps"].append({"step": "RETRAINING", "status": "COMPLETED"})
                report["steps"].append({"step": "QUALITY_GATE", "status": "PASS"})
                _append_job_log(config, batch_id, "quality_gate", result="PASS", bundle=report["model_bundle_id"])
            except BatchError as error:
                if error.code != "QUALITY_GATE_FAILED":
                    raise
                report["retrain_status"] = "REJECTED_BY_QUALITY_GATE"
                report["quality_gate_result"] = "REJECTED"
                report["retrain_error"] = {"error_code": error.code, "message": error.message}
                report["steps"].append({"step": "RETRAINING", "status": "COMPLETED"})
                report["steps"].append({"step": "QUALITY_GATE", "status": "REJECTED"})
                _append_job_log(config, batch_id, "quality_gate", result="REJECTED", error=error.message)
                gate = _json_if_exists(config.work_root / "batches" / batch_id / "retrain" / "gate.json")
                report["gate_metrics"] = gate
                update("QUALITY_GATE_FALLBACK")
                _, promoted_info = resolve_promoted(batch_config)
                report["model_used"] = "PREVIOUS_PROMOTED"
                report["model_bundle_id"] = promoted_info.get("bundle_id")
                report["steps"].append({"step": "USING_PREVIOUS_MODEL", "status": "COMPLETED"})

            update("INFERENCING")
            inference_result = run_inference(batch_config, batch_id)
            report["inference_result"] = inference_result
            if inference_result.get("status") not in {"published", "already_published", "already_done"}:
                raise BatchError("INFERENCE_FAILED", f"unexpected inference result: {inference_result.get('status')!r}")
            report["inference_status"] = "SUCCESS"
            report["publish_status"] = "SUCCESS"
            report["output_generation"] = inference_result.get("generation")
            report["output_files"] = inference_result.get("files", [])
            if report["output_generation"]:
                generation_dir = config.delivery_root / "generations" / report["output_generation"]
                file_names = report["output_files"].keys() if isinstance(report["output_files"], dict) else report["output_files"]
                report["output_file_paths"] = [str(generation_dir / name) for name in file_names]
                report["output_location"] = str(generation_dir)
            if inference_result.get("bundle_id"):
                report["model_bundle_id"] = inference_result["bundle_id"]
            report["steps"].append({"step": "INFERENCE", "status": "COMPLETED"})
            report["steps"].append({"step": "OUTPUT_VALIDATION_AND_PUBLISHING", "status": "COMPLETED"})
            _append_job_log(
                config, batch_id, "published", generation=report["output_generation"],
                files=report["output_files"],
            )
            report["status"] = "SUCCESS"
            return report
        finally:
            pass
    except BatchError as error:
        _append_job_log(config, batch_id, "error", error_code=error.code, message=error.message)
        report["status"] = "SKIPPED_OVERLAP" if not claimed and error.code == "BATCH_LOCKED" else "FAILED"
        report["error_code"] = error.code
        report["error_message"] = error.message
        phase = report.get("current_phase", "COORDINATION")
        if phase == "PREPARING_INPUT":
            report["input_status"] = "FAILED"
        elif phase == "PREPARING_DATA":
            report["preparation_status"] = "FAILED"
        elif phase == "RETRAINING" and report["retrain_status"] == "PENDING":
            report["retrain_status"] = "FAILED"
        if phase == "INFERENCING" or phase == "QUALITY_GATE_FALLBACK":
            report["inference_status"] = "FAILED"
        elif phase == "RETRAINING":
            report["inference_status"] = "PENDING"
        report["steps"].append({"step": phase, "status": "SKIPPED" if report["status"] == "SKIPPED_OVERLAP" else "FAILED"})
        raise
    except Exception as error:
        _append_job_log(config, batch_id, "error", error_type=type(error).__name__, message=str(error))
        phase = report.get("current_phase", "PREPARING_INPUT")
        code = {"PREPARING_INPUT": "INPUT_INTEGRITY", "PREPARING_DATA": "SCHEMA_DRIFT", "RETRAINING": "RETRAIN_FAILED", "INFERENCING": "INFERENCE_FAILED"}.get(phase, "PUBLISH_FAILED")
        wrapped = BatchError(code, f"{type(error).__name__}: {error}")
        report["status"] = "FAILED"
        report["error_code"] = wrapped.code
        report["error_message"] = wrapped.message
        if phase == "PREPARING_INPUT":
            report["input_status"] = "FAILED"
        elif phase == "PREPARING_DATA":
            report["preparation_status"] = "FAILED"
        elif phase == "RETRAINING" and report["retrain_status"] == "PENDING":
            report["retrain_status"] = "FAILED"
        if phase == "INFERENCING" or phase == "QUALITY_GATE_FALLBACK":
            report["inference_status"] = "FAILED"
        elif phase == "RETRAINING":
            report["inference_status"] = "PENDING"
        report["steps"].append({"step": phase, "status": "FAILED"})
        raise wrapped from error
    finally:
        skip_report_write = preserve_existing_report or (not claimed and report.get("error_code") == "BATCH_LOCKED")
        if not skip_report_write:
            report["actual_end"] = datetime.now(timezone.utc).isoformat()
            report["duration_seconds"] = time.perf_counter() - started
            path = _report_path(config, batch_id)
            report["report_path"] = str(path)
            atomic_write_text(path, json.dumps(report, indent=2, sort_keys=True, default=str))
        if claimed:
            status = report.get("status", "FAILED")
            try:
                state.finish_job(
                    batch_id,
                    report,
                    status=status,
                    error_code=report.get("error_code"),
                    error_message=report.get("error_message"),
                )
            finally:
                if lock is not None:
                    lock.__exit__(None, None, None)
        elif lock is not None:
            lock.__exit__(None, None, None)


def run_batch_action(
    config: ProductionConfig,
    batch_id: str,
    kind: str,
    *,
    store: StateStore | None = None,
    scheduled_at: datetime | None = None,
    data_end_at: str | None = None,
    run_id: str | None = None,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run only one already-prepared batch operation through Batch CLI logic."""
    import uuid
    from dataclasses import replace

    if kind not in {"prepare", "retrain", "inference"}:
        raise ValueError("kind must be prepare, retrain or inference")
    if kind == "prepare":
        raise ValueError("Prepare Data uses run_production(..., prepare_only=True)")

    state = store or StateStore(config.database_path)
    run_id = run_id or uuid.uuid4().hex
    slot = scheduled_at or datetime.now(timezone.utc)
    selected_data_root = state.get_data_root(config.data_root)
    source_config = replace(config, data_root=config.resolve_data_root(selected_data_root))
    batch_config = source_config.batch
    data_settings = state.get_data_settings({
        name: getattr(batch_config, name) for name in (
            "time_train", "time_valid", "time_test", "max_profiles", "gate_evaluation_profiles"
        )
    })
    batch_config = replace(batch_config, **data_settings)

    from recsys_vnpt import batch_runner

    manifest = batch_runner._validate_input(batch_config, batch_id, verify_checksums=False)
    job_key = f"{batch_id}--{run_id}"
    report: dict[str, Any] = {
        "batch_id": batch_id,
        "run_id": run_id,
        "kind": kind,
        "mode": "Manual" if data_end_at else "Automatic",
        "data_end_at": data_end_at,
        "batch_started_at": manifest.batch_started_at.isoformat(),
        "actual_start": datetime.now(timezone.utc).isoformat(),
        "actual_end": None,
        "status": "PENDING",
        "data_settings": data_settings,
        "input_fingerprint": manifest.fingerprint,
        "output_files": {},
    }
    lock = BatchLock(config.work_root / "locks", "production-global")
    lock.__enter__()
    claimed = False
    try:
        if not state.try_start_job(job_key, slot, manifest.batch_started_at):
            raise BatchError("BATCH_LOCKED", "another production job is active")
        claimed = True
        report["status"] = "RUNNING"
        state.update_job(job_key, report, "RUNNING")
        if kind == "retrain":
            result = batch_runner.run_retrain_batch(
                batch_config, batch_id, data_end_at, run_id=run_id
            )
            report["retrain_result"] = result
            report["model_used"] = "NEWLY_PROMOTED"
            report["model_bundle_id"] = Path(str(result.get("bundle", ""))).name or None
            report["quality_gate_result"] = result.get("gate", {}).get("passed", "UNKNOWN")
        else:
            result = batch_runner.run_inference_batch(
                batch_config, batch_id, data_end_at, run_id=run_id, model_path=model_path
            )
            report["inference_result"] = result
            report["serving_cutoff"] = result.get("serving_cutoff")
            report["model_bundle_id"] = result.get("bundle_id")
            report["output_generation"] = result.get("generation")
            report["output_files"] = result.get("files", {})
            report["output_location"] = str(config.delivery_root / "generations" / str(result.get("generation", "")))
            report["model_used"] = result.get("bundle_id")
        report["status"] = "COMPLETED"
        return report
    except Exception as error:
        report["status"] = "FAILED"
        report["error_code"] = getattr(error, "code", type(error).__name__)
        report["error_message"] = getattr(error, "message", str(error))
        raise
    finally:
        report["actual_end"] = datetime.now(timezone.utc).isoformat()
        report["duration_seconds"] = (
            datetime.fromisoformat(report["actual_end"])
            - datetime.fromisoformat(report["actual_start"])
        ).total_seconds()
        report_path = config.work_root / "batches" / batch_id / kind / "runs" / run_id / "dashboard_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(report_path, json.dumps(report, indent=2, sort_keys=True, default=str))
        if claimed:
            state.finish_job(
                job_key, report,
                status="SUCCESS" if report["status"] == "COMPLETED" else "FAILED",
                error_code=report.get("error_code"),
                error_message=report.get("error_message"),
            )
        lock.__exit__(None, None, None)

