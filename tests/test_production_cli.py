import json
from types import SimpleNamespace

from production.cli import main


def test_retrain_cli_dispatches_only_one_action_and_forwards_cutoff(monkeypatch, capsys):
    import production.cli as cli

    monkeypatch.setattr(cli, "load_production_config", lambda _: SimpleNamespace())
    calls = []
    monkeypatch.setattr(
        cli,
        "run_batch_action",
        lambda config, batch_id, kind, **kwargs: calls.append((batch_id, kind, kwargs))
        or {"status": "COMPLETED"},
    )

    result = main([
        "run-retrain", "--batch-id", "data",
        "--data-end-at", "2025-02-15T23:59:59+07:00",
    ])

    assert result == 0
    assert calls == [("data", "retrain", {
        "data_end_at": "2025-02-15T23:59:59+07:00", "run_id": None
    })]
    assert json.loads(capsys.readouterr().out)["status"] == "COMPLETED"


def test_inference_cli_forwards_optional_model_path(monkeypatch, capsys):
    import production.cli as cli

    monkeypatch.setattr(cli, "load_production_config", lambda _: SimpleNamespace())
    calls = []
    monkeypatch.setattr(
        cli,
        "run_batch_action",
        lambda config, batch_id, kind, **kwargs: calls.append((batch_id, kind, kwargs))
        or {"status": "COMPLETED"},
    )

    result = main([
        "run-inference", "--batch-id", "data", "--model-path", "/models/bundle-42",
    ])

    assert result == 0
    assert calls == [("data", "inference", {
        "data_end_at": None, "run_id": None, "model_path": "/models/bundle-42"
    })]
    assert json.loads(capsys.readouterr().out)["status"] == "COMPLETED"
