import json

from recsys_vnpt.batch_cli import main


def test_make_ready_cli_assigns_batch_start_without_cutoff_argument(tmp_path, capsys):
    batch = tmp_path / "slot-1"
    for relative in (
        "duration/part.csv",
        "mytv_vmp_content/content.json",
        "tags/TAG.json",
        "content_tags/CONTENT_TO_TAG.json",
        "month_mytv_info.parquet",
    ):
        path = batch / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)

    exit_code = main(["make-ready", "--batch-dir", str(batch)])

    result = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert result["batch_started_at"].endswith("+00:00")
    assert "data_cutoff" not in result


def test_make_ready_cli_rejects_manual_cutoff_argument(tmp_path):
    import pytest

    with pytest.raises(SystemExit):
        main(["make-ready", "--batch-dir", str(tmp_path), "--data-cutoff", "2025-03-01T00:00:00+00:00"])


def test_run_retrain_cli_forwards_manual_cutoff_and_retry_run_id(monkeypatch, capsys):
    from types import SimpleNamespace
    import recsys_vnpt.batch_cli as batch_cli
    import recsys_vnpt.batch_runner as runner

    monkeypatch.setattr(batch_cli, "load_config", lambda path: SimpleNamespace())
    calls = []
    monkeypatch.setattr(
        runner,
        "run_retrain_batch",
        lambda config, batch_id, data_end_at=None, *, run_id=None: calls.append(
            (batch_id, data_end_at, run_id)
        ) or {"status": "promoted", "run_id": run_id},
    )

    code = main([
        "run-retrain", "--config", "config.json", "--batch-id", "data",
        "--data-end-at", "2025-02-15T23:59:59+07:00", "--run-id", "retry-1",
    ])

    assert code == 0
    assert calls == [("data", "2025-02-15T23:59:59+07:00", "retry-1")]
    assert json.loads(capsys.readouterr().out)["run_id"] == "retry-1"


def test_run_inference_cli_forwards_optional_model_path(monkeypatch, capsys):
    from types import SimpleNamespace
    import recsys_vnpt.batch_cli as batch_cli
    import recsys_vnpt.batch_runner as runner

    monkeypatch.setattr(batch_cli, "load_config", lambda path: SimpleNamespace())
    calls = []
    monkeypatch.setattr(
        runner,
        "run_inference_batch",
        lambda config, batch_id, data_end_at=None, *, run_id=None, model_path=None: calls.append(
            (batch_id, run_id, model_path)
        ) or {"status": "published", "run_id": run_id},
    )

    code = main([
        "run-inference", "--config", "config.json", "--batch-id", "data",
        "--model-path", "/models/bundle-42", "--run-id", "inference-1",
    ])

    assert code == 0
    assert calls == [("data", "inference-1", "/models/bundle-42")]
    assert json.loads(capsys.readouterr().out)["run_id"] == "inference-1"


def test_run_retrain_cli_returns_config_error_for_timezone_naive_cutoff(monkeypatch, capsys):
    from types import SimpleNamespace
    import recsys_vnpt.batch_cli as batch_cli

    monkeypatch.setattr(batch_cli, "load_config", lambda path: SimpleNamespace())

    code = main([
        "run-retrain", "--config", "config.json", "--batch-id", "data",
        "--data-end-at", "2025-02-15T23:59:59",
    ])

    assert code == 14
    assert json.loads(capsys.readouterr().err)["error_code"] == "CONFIG_INVALID"


def test_failed_batch_cli_run_reports_run_id_for_retry(monkeypatch, capsys):
    from types import SimpleNamespace
    import recsys_vnpt.batch_cli as batch_cli
    import recsys_vnpt.batch_runner as runner
    from recsys_vnpt.batch_contracts import BatchError

    monkeypatch.setattr(batch_cli, "load_config", lambda path: SimpleNamespace())
    monkeypatch.setattr(
        runner,
        "run_inference_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(BatchError("MODEL_UNAVAILABLE", "no model")),
    )

    code = main([
        "run-inference", "--config", "config.json", "--batch-id", "data",
        "--run-id", "retry-44",
    ])

    error = json.loads(capsys.readouterr().err)
    assert code == 20
    assert error["run_id"] == "retry-44"
