import json
import sys
from pathlib import Path

import pytest

from experiments import kaggle_runner as runner


def _valid_smoke_result() -> dict[str, object]:
    return {
        "gpu_verified": True,
        "rows_scored": 8,
        "prediction_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
    }


def _mock_successful_commands(working_root: Path, monkeypatch, *, exit_codes=None):
    codes = iter(exit_codes or (0, 0, 0))
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if "experiments.kaggle_gpu_smoke" in argv:
            output_dir = Path(argv[argv.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "gpu_smoke.json").write_text(
                json.dumps(_valid_smoke_result()), encoding="utf-8"
            )
        return {
            "exit_code": next(codes),
            "argv": argv,
            "duration_seconds": 0.0,
        }

    monkeypatch.setattr(runner, "run_command", run)
    monkeypatch.setattr(
        runner, "preflight", lambda *_args, **_kwargs: {"gpu_verified": True}
    )
    return calls


def make_private_input_with_canonical_manifest(tmp_path):
    source = tmp_path / "private-input"
    for name in (
        "behavior",
        "duration",
        "mytv_vmp_content",
        "tags",
        "content_tags",
        "canonical",
    ):
        (source / name).mkdir(parents=True)
    (source / "month_mytv_info.parquet").write_bytes(b"fixture")
    (source / "canonical" / "source_manifest.json").write_text(
        '{"fingerprint":"fixture"}', encoding="utf-8"
    )
    return source


def test_synthetic_suite_does_not_mount_private_data_and_passes(tmp_path, monkeypatch):
    working_root = tmp_path / "work"
    calls = _mock_successful_commands(working_root, monkeypatch)

    report = runner.run_suite(tmp_path, None, working_root, "synthetic")

    assert report["status"] == "passed"
    assert report["suite"] == "synthetic"
    assert len(calls) == 2
    assert not (working_root / "data").exists()
    assert report["artifacts"]["gpu_smoke_sha256"]
    assert (working_root / "kaggle_test_report.json").is_file()
    assert not list(working_root.glob(".*.stage"))


def test_runner_does_not_promote_failed_gpu_smoke(tmp_path, monkeypatch):
    _mock_successful_commands(tmp_path / "work", monkeypatch, exit_codes=(0, 1))
    calls = []
    monkeypatch.setattr(
        runner,
        "run_command",
        lambda argv, **_kwargs: calls.append(argv)
        or {"exit_code": 0 if len(calls) == 1 else 1, "argv": argv, "duration_seconds": 0.0},
    )

    report = runner.run_suite(tmp_path, None, tmp_path / "work", "synthetic")

    assert report["status"] == "failed"
    assert len(calls) == 2
    assert (tmp_path / "work" / "kaggle_test_report.json").exists()


def test_runner_fails_when_pytest_fails(tmp_path, monkeypatch):
    calls = _mock_successful_commands(tmp_path / "work", monkeypatch, exit_codes=(1, 0))

    report = runner.run_suite(tmp_path, None, tmp_path / "work", "synthetic")

    assert report["status"] == "failed"
    assert len(calls) == 1
    assert report["commands"][0]["exit_code"] == 1


def test_runner_rejects_missing_or_corrupt_smoke_result(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner, "preflight", lambda *_args, **_kwargs: {"gpu_verified": True}
    )
    monkeypatch.setattr(
        runner,
        "run_command",
        lambda argv, **_kwargs: {
            "exit_code": 0,
            "argv": argv,
            "duration_seconds": 0.0,
        },
    )

    report = runner.run_suite(tmp_path, None, tmp_path / "work", "synthetic")

    assert report["status"] == "failed"
    assert "gpu_smoke.json" in report["error"]


def test_runner_rejects_corrupt_smoke_json(tmp_path, monkeypatch):
    working_root = tmp_path / "work"
    calls = _mock_successful_commands(working_root, monkeypatch)

    def corrupt_second_command(argv, **kwargs):
        result = calls_original(argv, **kwargs)
        if "experiments.kaggle_gpu_smoke" in argv:
            output_dir = Path(argv[argv.index("--output-dir") + 1])
            (output_dir / "gpu_smoke.json").write_text("not-json", encoding="utf-8")
        return result

    calls_original = runner.run_command
    monkeypatch.setattr(runner, "run_command", corrupt_second_command)

    report = runner.run_suite(tmp_path, None, working_root, "synthetic")

    assert report["status"] == "failed"
    assert "smoke result" in report["error"].lower()


def test_real_suite_requires_private_input(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner, "preflight", lambda *_args, **_kwargs: {"gpu_verified": True}
    )

    report = runner.run_suite(tmp_path, None, tmp_path / "work", "real")

    assert report["status"] == "failed"
    assert "private input" in report["error"]


def test_real_suite_requires_dataset_version_before_infer(tmp_path, monkeypatch):
    source = make_private_input_with_canonical_manifest(tmp_path)
    monkeypatch.delenv("RECSYS_KAGGLE_DATASET_VERSION", raising=False)
    calls = _mock_successful_commands(tmp_path / "work", monkeypatch)

    report = runner.run_suite(tmp_path, source, tmp_path / "work", "real")

    assert report["status"] == "failed"
    assert "RECSYS_KAGGLE_DATASET_VERSION" in report["error"]
    assert calls == []


@pytest.mark.skipif(
    sys.platform == "win32", reason="mounts Kaggle inputs as symlinks, which Windows restricts"
)
def test_real_suite_records_manifest_fingerprint_and_hash(tmp_path, monkeypatch):
    source = make_private_input_with_canonical_manifest(tmp_path)
    monkeypatch.setenv("RECSYS_KAGGLE_DATASET_VERSION", "private-v7")
    calls = _mock_successful_commands(tmp_path / "work", monkeypatch)

    report = runner.run_suite(tmp_path, source, tmp_path / "work", "real")

    assert report["status"] == "passed"
    assert report["dataset_version"] == "private-v7"
    assert report["source_manifest"]["fingerprint"] == "fixture"
    assert len(report["source_manifest"]["sha256"]) == 64
    assert "RECSYS_VNPT_DATA_ROOT" in calls[0][1]["env"]
    assert any("all/infer.py" in call[0] for call in calls)


def test_kaggle_bootstrap_and_requirements_are_portable():
    root = Path(__file__).resolve().parents[1]
    base = {
        line.strip()
        for line in (root / "requirements.txt").read_text().splitlines()
        if line.strip()
    }
    kaggle = {
        line for line in base
        if not line.startswith("torch==") and not line.startswith("--extra-index-url")
    }
    assert "torch==2.7.0+cu128" in base
    assert kaggle <= base
    script = (root / "kaggle" / "bootstrap.sh").read_text()
    assert "python -m experiments.kaggle_runner" in script
    assert "requirements-kaggle.txt" in script  # generated in the work dir
    assert "grep -Ev" in script
    assert "--extra-index-url|torch==" in script
    assert '"$work_dir/repo/requirements.txt"' in script
    assert "--exclude='./data'" in script
    assert "source and working roots overlap" in script
    guide = (root / "docs" / "KAGGLE_TESTING.md").read_text(encoding="utf-8")
    for required in (
        "/kaggle/input",
        "/kaggle/working",
        "private",
        "Save & Run All",
        "kaggle_test_report.json",
    ):
        assert required in guide


def test_repository_provenance_fingerprints_code_bundle_without_git(tmp_path, monkeypatch):
    monkeypatch.delenv("RECSYS_KAGGLE_SOURCE_COMMIT", raising=False)
    (tmp_path / "experiments").mkdir()
    code_file = tmp_path / "experiments" / "runner.py"
    code_file.write_text("version = 1\n", encoding="utf-8")
    (tmp_path / "data").mkdir()
    private_file = tmp_path / "data" / "private.parquet"
    private_file.write_bytes(b"private")

    first = runner._repository_provenance(tmp_path)
    private_file.write_bytes(b"different private data")
    after_data_change = runner._repository_provenance(tmp_path)
    code_file.write_text("version = 2\n", encoding="utf-8")
    second = runner._repository_provenance(tmp_path)

    assert first["commit"] is None
    assert first["code_bundle_sha256"] == after_data_change["code_bundle_sha256"]
    assert first["code_bundle_sha256"] != second["code_bundle_sha256"]
    assert len(first["code_bundle_sha256"]) == 64
