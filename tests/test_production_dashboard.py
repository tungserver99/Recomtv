import sys
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from production.dashboard.app import build_pipeline_steps, demo_jobs, live_log_text, read_batch_status


def test_dashboard_entrypoint_imports_from_outside_project_root(tmp_path):
    root = Path(__file__).resolve().parents[1]
    entrypoint = root / "production" / "dashboard" / "app.py"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    probe = "import runpy, sys; runpy.run_path(sys.argv[1], run_name='dashboard_import_probe')"
    result = subprocess.run(
        [sys.executable, "-c", probe, str(entrypoint)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_dashboard_uses_real_batch_status_and_does_not_invent_percentages(tmp_path):
    status_path = tmp_path / "status.json"
    status_path.write_text(
        '{"batch_id":"20261113_1200","kind":"retrain","phase":"TRAINING","history":[{"phase":"TRAINING","at":"now"}],"details":{}}',
        encoding="utf-8",
    )
    assert read_batch_status(status_path)["phase"] == "TRAINING"
    steps = build_pipeline_steps(
        {},
        {"phase": "TRAINING"},
        None,
    )
    training = next(step for step in steps if step["name"] == "Retraining")
    assert training["status"] == "RUNNING"
    assert "percent" not in training


def test_dashboard_pipeline_shows_preparation_only_status():
    completed = build_pipeline_steps(
        {"preparation_status": "READY"}, None, None
    )
    assert next(step for step in completed if step["name"] == "Preparing Data")["status"] == "COMPLETED"

    running = build_pipeline_steps(
        {"preparation_status": "PREPARING"}, None, None
    )
    assert next(step for step in running if step["name"] == "Preparing Data")["status"] == "RUNNING"

    failed = build_pipeline_steps(
        {"preparation_status": "FAILED"}, None, None
    )
    assert next(step for step in failed if step["name"] == "Preparing Data")["status"] == "FAILED"


def test_dashboard_shows_input_staging_separately_from_preparation():
    steps = build_pipeline_steps(
        {"current_phase": "PREPARING_INPUT", "input_status": "PENDING"}, None, None
    )
    statuses = {step["name"]: step["status"] for step in steps}
    assert statuses["Input validation & staging"] == "RUNNING"
    assert statuses["Preparing Data"] == "PENDING"


def test_live_log_text_shows_only_the_latest_entries(tmp_path):
    log_path = tmp_path / "log.jsonl"
    log_path.write_text(
        "\n".join('{"message":"line-%s"}' % index for index in range(4)),
        encoding="utf-8",
    )

    result = live_log_text(log_path, limit=2)

    assert "line-0" not in result
    assert "line-2" in result
    assert "line-3" in result


def test_live_log_text_can_show_the_complete_log(tmp_path):
    log_path = tmp_path / "log.jsonl"
    log_path.write_text(
        "\n".join('{"message":"line-%s"}' % index for index in range(125)),
        encoding="utf-8",
    )

    result = live_log_text(log_path)

    assert "line-0" in result
    assert "line-124" in result


def test_gate_rejection_with_fallback_and_published_inference_is_successful():
    report = {
        "retrain_status": "REJECTED_BY_QUALITY_GATE",
        "quality_gate_result": "REJECTED",
        "model_used": "PREVIOUS_PROMOTED",
        "inference_status": "SUCCESS",
        "publish_status": "SUCCESS",
    }
    steps = build_pipeline_steps(report, {"phase": "REJECTED"}, {"phase": "PUBLISHED"})
    statuses = {step["name"]: step["status"] for step in steps}
    assert statuses["Quality Gate"] == "REJECTED"
    assert statuses["Using Previous Model"] == "COMPLETED"
    assert statuses["Inference"] == "COMPLETED"
    assert statuses["Output Validation & Publishing"] == "COMPLETED"


def test_demo_scenarios_are_static_and_explicitly_simulated():
    jobs = demo_jobs()
    assert {job["demo_scenario"] for job in jobs} == {"PASS", "FALLBACK"}
    assert all(job["simulated"] is True for job in jobs)
    assert all(job["batch_id"].startswith("DEMO-") for job in jobs)


def test_main_overview_renders_without_local_cutoff_import_shadowing(monkeypatch):
    fragment_settings = []

    def fragment(**kwargs):
        fragment_settings.append(kwargs)
        return lambda function: function

    class Column:
        def metric(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    class Sidebar:
        def radio(self, label, options):
            return "Live" if label == "Mode" else "Overview"

    fake_streamlit = SimpleNamespace(
        set_page_config=lambda **kwargs: None,
        fragment=fragment,
        title=lambda *args, **kwargs: None,
        sidebar=Sidebar(),
        columns=lambda count: [Column() for _ in range(count)],
        caption=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        button=lambda *args, **kwargs: False,
    )
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)

    class Store:
        def __init__(self, *_):
            pass

        def get_schedule(self, default):
            return default

        def current_job(self):
            return None

        def jobs(self, **kwargs):
            return []

    fake_config = SimpleNamespace(
        config_path=Path("production/config.json"),
        database_path=Path("state.db"),
        project_root=Path("."),
        schedule_enabled=True,
        run_times=("00:00", "12:00"),
        timezone="Asia/Ho_Chi_Minh",
        model_registry=Path("models"),
    )
    monkeypatch.setattr("production.config.load_production_config", lambda _: fake_config)
    monkeypatch.setattr("production.state_store.StateStore", Store)
    monkeypatch.setitem(sys.modules, "recsys_vnpt.scorer_bundle", SimpleNamespace(promoted_bundle=lambda _: None))

    from production.dashboard.app import main

    main()
    assert fragment_settings == [{"run_every": "5s"}]
