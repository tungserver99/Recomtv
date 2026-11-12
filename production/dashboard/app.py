from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Streamlit executes this file as a script and may put only the dashboard
# directory on sys.path. Add the repository root before importing project code.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from production.cutoff import batch_id_for_slot, resolve_slot, validate_run_times
from production.schedule import next_scheduled_slot


PHASES = {
    "WAITING_INPUT": "PENDING",
    "PREPARING_INPUT": "RUNNING",
    "VALIDATING_INPUT": "RUNNING",
    "PREPARING_DATA": "RUNNING",
    "LOADING_PREPARED_DATA": "RUNNING",
    "TRAINING": "RUNNING",
    "EVALUATING": "RUNNING",
    "GATING": "RUNNING",
    "LOADING_MODEL": "RUNNING",
    "RUNNING_SHARDS": "RUNNING",
    "MERGING": "RUNNING",
    "VALIDATING_OUTPUT": "RUNNING",
    "PUBLISHING": "RUNNING",
    "PROMOTED": "COMPLETED",
    "PUBLISHED": "COMPLETED",
    "REJECTED": "REJECTED",
    "FAILED": "FAILED",
}


def read_batch_status(path: str | Path) -> dict[str, Any] | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_logs(path: str | Path, limit: int | None = 100) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        return []
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    if limit is not None:
        lines = lines[-limit:] if limit else []
    result = []
    for line in lines:
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            result.append({"message": line})
    return result


def live_log_text(path: str | Path, limit: int | None = None) -> str:
    """Format structured log entries for the live overview."""
    return "\n".join(
        json.dumps(entry, ensure_ascii=False, sort_keys=True, default=str)
        for entry in read_logs(path, limit=limit)
    )


def _phase(status: dict[str, Any] | None, pending: str) -> tuple[str, str | None]:
    if not status:
        return pending, None
    phase = str(status.get("phase", "UNKNOWN"))
    return PHASES.get(phase, "PENDING"), phase


def build_pipeline_steps(
    report: dict[str, Any] | None,
    retrain_status: dict[str, Any] | None,
    inference_status: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    report = report or {}
    input_state, input_phase = _phase(
        {"phase": report.get("current_phase")} if report.get("current_phase") == "PREPARING_INPUT" else None,
        "PENDING",
    )
    if report.get("input_status") == "VALIDATED":
        input_state = "COMPLETED"
    elif report.get("input_status") == "FAILED":
        input_state = "FAILED"
    preparation = {
        "READY": "COMPLETED",
        "PREPARING": "RUNNING",
        "FAILED": "FAILED",
    }.get(report.get("preparation_status"), "PENDING")
    if report.get("current_phase") == "PREPARING_DATA":
        preparation = "RUNNING"
    if retrain_status and retrain_status.get("phase") == "PREPARING_DATA":
        preparation = "RUNNING"
    elif retrain_status and retrain_status.get("phase") == "LOADING_PREPARED_DATA":
        preparation = "COMPLETED"
    retrain_state, retrain_phase = _phase(retrain_status, "PENDING")
    if not retrain_status and report.get("current_phase") == "RETRAINING":
        retrain_state = "RUNNING"
    retrain_result = report.get("retrain_status")
    if retrain_result == "PROMOTED":
        retrain_state = "COMPLETED"
    elif retrain_result == "REJECTED_BY_QUALITY_GATE":
        retrain_state = "COMPLETED"
    elif retrain_result == "FAILED":
        retrain_state = "FAILED"

    gate = report.get("quality_gate_result", "PENDING")
    if gate == "PASS":
        gate = "COMPLETED"
    elif gate == "REJECTED":
        gate = "REJECTED"
    elif retrain_phase == "PROMOTED":
        gate = "COMPLETED"
    elif retrain_phase == "REJECTED":
        gate = "REJECTED"
    elif retrain_phase in {"EVALUATING", "GATING"}:
        gate = "RUNNING"
    elif report.get("current_phase") == "QUALITY_GATE_FALLBACK":
        gate = "REJECTED"

    model_state = "COMPLETED" if report.get("model_used") == "PREVIOUS_PROMOTED" else "PENDING"
    if report.get("model_used") and report.get("model_used") != "PREVIOUS_PROMOTED":
        model_state = "COMPLETED"

    inference, inference_phase = _phase(inference_status, "PENDING")
    if not inference_status and report.get("current_phase") == "INFERENCING":
        inference = "RUNNING"
    if report.get("inference_status") == "SUCCESS":
        inference = "COMPLETED"
    elif report.get("inference_status") == "FAILED":
        inference = "FAILED"
    publish = "PENDING"
    if inference_phase in {"VALIDATING_OUTPUT", "PUBLISHING"}:
        publish = "RUNNING"
    if inference_phase == "PUBLISHED" or report.get("publish_status") == "SUCCESS":
        publish = "COMPLETED"
    elif inference_phase == "FAILED" and report.get("inference_status") == "FAILED":
        publish = "FAILED"

    steps = [
        {"name": "Input validation & staging", "status": input_state, "phase": input_phase},
        {"name": "Preparing Data", "status": preparation, "phase": "PREPARING_DATA"},
        {"name": "Retraining", "status": retrain_state, "phase": retrain_phase},
        {"name": "Quality Gate", "status": gate, "phase": retrain_phase},
    ]
    if model_state != "PENDING":
        steps.append({"name": "Using Previous Model" if report.get("model_used") == "PREVIOUS_PROMOTED" else "Promoted Model", "status": model_state, "phase": None})
    steps.extend(
        [
            {"name": "Inference", "status": inference, "phase": inference_phase},
            {"name": "Output Validation & Publishing", "status": publish, "phase": inference_phase},
        ]
    )
    return steps


def demo_jobs() -> list[dict[str, Any]]:
    return [
        {
            "batch_id": "DEMO-20261113_0000-PASS",
            "scheduled_at": "2026-11-13T00:00:00+07:00",
            "batch_started_at": "2026-11-13T00:00:00+07:00",
            "status": "SUCCESS",
            "simulated": True,
            "demo_scenario": "PASS",
            "report": {
                "input_status": "VALIDATED",
                "retrain_status": "PROMOTED",
                "quality_gate_result": "PASS",
                "model_used": "NEWLY_PROMOTED",
                "inference_status": "SUCCESS",
                "publish_status": "SUCCESS",
                "output_generation": "DEMO-GEN-0000",
                "duration_seconds": 1820,
                "output_location": "Demo data only",
            },
        },
        {
            "batch_id": "DEMO-20261113_1200-FALLBACK",
            "scheduled_at": "2026-11-13T12:00:00+07:00",
            "batch_started_at": "2026-11-13T12:00:00+07:00",
            "status": "SUCCESS",
            "simulated": True,
            "demo_scenario": "FALLBACK",
            "report": {
                "input_status": "VALIDATED",
                "retrain_status": "REJECTED_BY_QUALITY_GATE",
                "quality_gate_result": "REJECTED",
                "model_used": "PREVIOUS_PROMOTED",
                "inference_status": "SUCCESS",
                "publish_status": "SUCCESS",
                "output_generation": "DEMO-GEN-1200",
                "duration_seconds": 1680,
                "output_location": "Demo data only",
            },
        },
    ]


def validate_schedule_form(timezone: str, run_times: list[str]) -> dict[str, Any]:
    try:
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise ValueError(f"unknown timezone {timezone!r}") from error
    return {"timezone": timezone, "run_times": list(validate_run_times(run_times))}


def _status_badge(status: str) -> str:
    icons = {"PENDING": "◯", "RUNNING": "◌", "COMPLETED": "✓", "FAILED": "✕", "REJECTED": "!"}
    return f"{icons.get(status, '•')} **{status}**"


def _render_pipeline(st, steps: list[dict[str, Any]]) -> None:
    for step in steps:
        left, right = st.columns([3, 1])
        detail = f" · `{step['phase']}`" if step.get("phase") else ""
        left.markdown(f"**{step['name']}**{detail}")
        right.markdown(_status_badge(step["status"]))


def _render_demo(st) -> None:
    st.warning("DEMO MODE · Nội dung bên dưới là dữ liệu mô phỏng, không đọc hoặc ghi trạng thái production.")
    jobs = demo_jobs()
    scenario = st.selectbox("Demo scenario", ["PASS", "FALLBACK"])
    job = next(item for item in jobs if item["demo_scenario"] == scenario)
    report = job["report"]
    st.subheader("Overview · Simulated")
    st.caption("MyTV Recommendation System")
    cols = st.columns(4)
    cols[0].metric("Schedule", "Enabled · 00:00 / 12:00")
    cols[1].metric("Next run", "12 Nov 2026 · 12:00 ICT")
    cols[2].metric("Batch started", "00:00 / 12:00 ICT")
    cols[3].metric("Current job", job["status"])
    st.info(f"Mô hình sử dụng: {'model đã promote trước đó' if scenario == 'FALLBACK' else 'model mới đã promote'}")
    _render_pipeline(st, build_pipeline_steps(report, {"phase": "REJECTED" if scenario == "FALLBACK" else "PROMOTED"}, {"phase": "PUBLISHED"}))
    st.subheader("Job History · Simulated")
    st.dataframe(
        [{"Batch ID": item["batch_id"], "Scheduled time": item["scheduled_at"], "Batch started": item["batch_started_at"], "Result": item["demo_scenario"], "Model used": item["report"]["model_used"], "Duration (s)": item["report"]["duration_seconds"]} for item in jobs],
        use_container_width=True,
        hide_index=True,
    )
    st.button("Run now", disabled=True)
    st.caption("Run now and configuration editing are disabled in demo mode.")


def _render_live_overview(st, config, store, schedule) -> None:
    now = datetime.now(ZoneInfo(schedule["timezone"]))
    next_slot = next_scheduled_slot(now, schedule["run_times"])
    current = store.current_job()
    jobs = store.jobs(limit=500)
    history = jobs[:1]
    job = current or (history[0] if history else None)
    report = job.get("report", {}) if job else {}
    ready_preparations = [
        item for item in jobs
        if item["status"] == "PREPARED" and item["report"].get("preparation_status") == "READY"
    ]
    prepared_job = ready_preparations[0] if ready_preparations else None
    if current and current.get("report", {}).get("preparation_status") == "PREPARING":
        preparation_state = "Preparing"
    elif job and job.get("report", {}).get("preparation_status") == "FAILED":
        preparation_state = "Failed"
    else:
        preparation_state = "Ready" if prepared_job else "Not prepared"
    prepared_for_next_slot = bool(
        prepared_job and prepared_job["batch_id"] == batch_id_for_slot(next_slot)
    )
    cards = st.columns(5)
    cards[0].metric("Schedule", "Enabled" if schedule["enabled"] else "Disabled")
    cards[1].metric("Next run", next_slot.strftime("%d %b %Y · %H:%M %Z"))
    cards[2].metric("Latest batch started", job.get("batch_started_at", "—") if job else "—")
    cards[3].metric("Current status", job["status"] if job else "IDLE")
    cards[4].metric("Prepared data", preparation_state)
    try:
        from recsys_vnpt.scorer_bundle import promoted_bundle

        bundle = promoted_bundle(config.model_registry)
    except Exception:
        bundle = None
    st.caption(f"Production model: {bundle.name if bundle else 'No promoted model found'}")
    if prepared_job:
        preparation_result = prepared_job["report"].get("preparation_result", {})
        duration = preparation_result.get("prepare_seconds")
        duration_text = f" · {duration:.1f}s" if isinstance(duration, (int, float)) else ""
        st.caption(
            f"Prepared batch {prepared_job['batch_id']} · input "
            f"{str(prepared_job['report'].get('input_fingerprint', ''))[:12]} · "
            f"{prepared_job['report'].get('prepared_dataset', 'manifest unavailable')}"
            f"{duration_text}"
        )
    if job:
        st.subheader("Current / Latest Pipeline")
        batch_id = job["batch_id"]
        batch_root = config.work_root / "batches" / batch_id
        retrain_status = read_batch_status(batch_root / "retrain" / "status.json")
        inference_status = read_batch_status(batch_root / "inference" / "status.json")
        _render_pipeline(st, build_pipeline_steps(report, retrain_status, inference_status))
        st.subheader("All job steps and logs")
        st.caption("All recorded entries are shown here; this panel updates every 5 seconds.")
        log_paths = {
            "Production orchestration": batch_root / "production" / "log.jsonl",
            "Data preparation": batch_root / "prepare" / "log.jsonl",
            "Retraining": batch_root / "retrain" / "log.jsonl",
            "Inference and publishing": batch_root / "inference" / "log.jsonl",
        }
        for label, log_path in log_paths.items():
            log_text = live_log_text(log_path)
            with st.expander(label, expanded=(label != "Inference and publishing")):
                if log_text:
                    st.code(log_text, language="json")
                else:
                    st.caption("No log entries yet.")
        if report.get("status") == "SUCCESS":
            st.success(f"Job completed · model {report.get('model_used', 'unknown')} · {report.get('duration_seconds', 0):.1f}s")
        elif report.get("error_message"):
            st.error(f"{report.get('error_code')}: {report['error_message']}")
    else:
        st.info("No production runs recorded yet.")
    prepare_disabled = bool(current or prepared_for_next_slot)
    prepare_column, run_column = st.columns(2)
    with prepare_column:
        if st.button("Prepare Data", disabled=prepare_disabled):
            try:
                request_id = store.enqueue_request(next_slot, kind="prepare")
            except ValueError as error:
                st.warning(str(error))
            else:
                st.success(
                    f"Prepare request queued ({request_id[:8]}) for next slot "
                    f"{batch_id_for_slot(next_slot)}."
                )
        if prepared_for_next_slot:
            st.caption("The next scheduled slot already has prepared data.")
        elif current:
            st.caption("Wait for the current job to finish before preparing another batch.")
    with run_column:
        run_clicked = st.button("Run now", type="primary", disabled=bool(current or not prepared_job))
        if run_clicked and prepared_job:
            slot = datetime.fromisoformat(prepared_job["scheduled_at"])
            batch_id = prepared_job["batch_id"]
            existing = next((item for item in jobs if item["batch_id"] == batch_id and item["status"] == "SUCCESS"), None)
            try:
                from recsys_vnpt.export import read_current_delivery

                delivery = read_current_delivery(config.delivery_root)
            except Exception as error:
                delivery = None
                st.error(f"Could not verify the current delivery: {error}")
            if delivery and delivery.get("batch_id") == batch_id:
                st.warning(f"Scheduled slot {batch_id} has already been published.")
            elif existing:
                st.warning(f"Scheduled slot {batch_id} has already completed.")
            else:
                try:
                    request_id = store.enqueue_request(slot, kind="production")
                except ValueError as error:
                    st.warning(str(error))
                else:
                    st.success(f"Run request queued ({request_id[:8]}) for prepared batch {batch_id}.")
        if not prepared_job:
            st.info("Prepare Data first. Retrain and inference only run against a prepared dataset.")


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="MyTV Recommendation System", page_icon="📺", layout="wide")
    st.title("MyTV Recommendation System")
    mode = st.sidebar.radio("Mode", ["Live", "Demo mode"])
    if mode == "Demo mode":
        _render_demo(st)
        return

    from production.config import load_production_config
    from production.state_store import StateStore

    config_path = Path(__file__).resolve().parents[1] / "config.json"
    try:
        config = load_production_config(config_path)
        store = StateStore(config.database_path)
    except Exception as error:
        st.error(f"Cannot load production configuration: {error}")
        return
    default_schedule = {
        "enabled": config.schedule_enabled,
        "run_times": list(config.run_times),
        "timezone": config.timezone,
    }
    schedule = store.get_schedule(default_schedule)
    page = st.sidebar.radio("Page", ["Overview", "Configuration", "Job History"])

    if page == "Overview":
        st.fragment(run_every="5s")(_render_live_overview)(st, config, store, schedule)

    elif page == "Configuration":
        st.subheader("Raw data folder")
        configured_data_root = store.get_data_root(config.data_root)
        with st.form("data_root_config"):
            data_root_value = st.text_input(
                "Data folder (contains duration, tags, content_tags, mytv_vmp_content)",
                value=configured_data_root,
            )
            data_root_submitted = st.form_submit_button("Save data folder")
        if data_root_submitted:
            try:
                selected_data_root = config.resolve_data_root(data_root_value)
                store.save_data_root(selected_data_root)
            except ValueError as error:
                st.error(str(error))
            else:
                st.success("Data folder saved. It will be used for the next Prepare Data or production job.")
                st.rerun()
        st.caption("The raw folder is copied into an immutable batch snapshot before preparation.")

        st.subheader("Schedule configuration")
        with st.form("schedule_config"):
            enabled = st.toggle("Schedule enabled", value=bool(schedule["enabled"]))
            timezone = st.text_input("Timezone", value=schedule["timezone"])
            times = list(schedule["run_times"])
            first = st.text_input("Run time 1 (HH:MM)", value=times[0] if times else "00:00")
            second = st.text_input("Run time 2 (HH:MM)", value=times[1] if len(times) > 1 else "12:00")
            submitted = st.form_submit_button("Save Configuration")
        new_times = [first.strip(), second.strip()]
        try:
            candidate = validate_schedule_form(timezone.strip(), new_times)
            local_now = datetime.now(ZoneInfo(candidate["timezone"]))
            for run_time in candidate["run_times"]:
                from datetime import timedelta
                hour, minute = map(int, run_time.split(":"))
                run_slot = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if run_slot <= local_now:
                    run_slot += timedelta(days=1)
                st.caption(f"Next batch slot: {run_slot.strftime('%Y-%m-%d %H:%M %Z')}")
        except ValueError as error:
            candidate = None
            st.error(str(error))
        if submitted and candidate:
            store.save_schedule({"enabled": enabled, **candidate})
            st.success("Configuration saved. The scheduler will apply it shortly.")
            st.rerun()

        st.divider()
        st.subheader("Data volume and temporal windows")
        batch_defaults = config.batch
        setting_names = (
            "time_train", "time_valid", "time_test", "max_profiles",
            "gate_evaluation_profiles",
        )
        data_settings = store.get_data_settings(
            {name: getattr(batch_defaults, name) for name in setting_names}
        )
        with st.form("data_settings"):
            train_hours = st.number_input(
                "Train window (hours; applies to each of 3 train windows)",
                min_value=1, max_value=8760, value=data_settings["time_train"], step=1,
            )
            valid_hours = st.number_input(
                "Validation window (hours)",
                min_value=1, max_value=8760, value=data_settings["time_valid"], step=1,
            )
            test_hours = st.number_input(
                "Test window (hours)",
                min_value=1, max_value=8760, value=data_settings["time_test"], step=1,
            )
            max_profiles = st.number_input(
                "Inference profiles (0 = all)",
                min_value=0, max_value=10_000_000,
                value=data_settings["max_profiles"], step=100,
            )
            gate_profiles = st.number_input(
                "Quality gate test profiles",
                min_value=1, max_value=10_000_000,
                value=data_settings["gate_evaluation_profiles"], step=100,
            )
            data_submitted = st.form_submit_button("Save data settings")
        if data_submitted:
            store.save_data_settings({
                "time_train": int(train_hours),
                "time_valid": int(valid_hours),
                "time_test": int(test_hours),
                "max_profiles": int(max_profiles),
                "gate_evaluation_profiles": int(gate_profiles),
            })
            st.success("Data settings saved. They will apply to the next batch.")
            st.rerun()
        st.caption(
            "The three train windows each use the Train duration. Quality gate profiles "
            "are sampled from Test; inference profile count does not limit retraining data."
        )

    else:
        st.subheader("Job History")
        jobs = store.jobs(limit=200)
        rows = []
        for item in jobs:
            report = item["report"]
            rows.append(
                {
                    "Batch ID": item["batch_id"],
                    "Scheduled time": item["scheduled_at"],
                    "Actual start": item["actual_start"],
                    "Batch started": item["batch_started_at"],
                    "Retrain result": report.get("retrain_status"),
                    "Preparation": report.get("preparation_status", "NOT PREPARED"),
                    "Quality gate": report.get("quality_gate_result"),
                    "Model used": report.get("model_used"),
                    "Inference": report.get("inference_status"),
                    "Duration (s)": report.get("duration_seconds"),
                    "Output location": report.get("output_location"),
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)
        if jobs:
            selected_id = st.selectbox("Inspect job", [item["batch_id"] for item in jobs])
            selected = next(item for item in jobs if item["batch_id"] == selected_id)
            st.json(selected["report"])
            batch_root = config.work_root / "batches" / selected_id
            st.subheader("Pipeline logs")
            for kind in ("prepare", "retrain", "inference"):
                status = read_batch_status(batch_root / kind / "status.json")
                with st.expander(f"{kind.title()} status · {status.get('phase') if status else 'not started'}"):
                    if status:
                        st.json(status)
                    st.dataframe(read_logs(batch_root / kind / "log.jsonl"), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

