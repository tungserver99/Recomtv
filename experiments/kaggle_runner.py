"""Run the clean-session Kaggle test suite and write fail-closed evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.kaggle_data import prepare_kaggle_data_root
from experiments.kaggle_preflight import preflight


def write_report_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_name(f".{path.name}.{uuid.uuid4().hex}.stage")
    stage.write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(stage, path)


def run_command(
    argv: list[str], *, cwd: Path, env: dict[str, str]
) -> dict[str, Any]:
    start = time.perf_counter()
    result = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    outcome = {
        "argv": argv,
        "exit_code": result.returncode,
        "duration_seconds": time.perf_counter() - start,
    }
    if result.returncode != 0:
        # A failed run must be diagnosable from the downloaded report alone.
        outcome["output_tail"] = (result.stdout + result.stderr)[-4000:]
    return outcome


def _repository_provenance(repo_root: Path) -> dict[str, Any]:
    commit = os.environ.get("RECSYS_KAGGLE_SOURCE_COMMIT")
    dirty_state: str | None = None
    try:
        if not commit:
            result = subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            if result.returncode == 0:
                commit = result.stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--short"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if status.returncode == 0:
            dirty_state = "clean" if not status.stdout.strip() else "dirty"
    except (OSError, subprocess.TimeoutExpired):
        pass
    digest = hashlib.sha256()
    ignored_roots = {".git", ".pytest_cache", "data"}
    repo_root = Path(repo_root)
    for directory, subdirectories, filenames in os.walk(
        repo_root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        relative_directory = directory_path.relative_to(repo_root)
        if relative_directory == Path("."):
            subdirectories[:] = [
                name for name in subdirectories if name not in ignored_roots
            ]
        subdirectories[:] = sorted(
            name
            for name in subdirectories
            if name != "__pycache__" and not (directory_path / name).is_symlink()
        )
        for name in sorted(filenames):
            path = directory_path / name
            if path.is_symlink() or not path.is_file():
                continue
            relative_bytes = path.relative_to(repo_root).as_posix().encode("utf-8")
            digest.update(len(relative_bytes).to_bytes(8, "big"))
            digest.update(relative_bytes)
            with path.open("rb") as source_file:
                for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                    digest.update(chunk)
    return {
        "commit": commit,
        "dirty_state": dirty_state or "unknown",
        "code_bundle_sha256": digest.hexdigest(),
    }


def _validate_smoke_result(smoke_path: Path) -> str:
    if not smoke_path.is_file():
        raise FileNotFoundError("GPU smoke command did not create gpu_smoke.json")
    try:
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid GPU smoke result: {error}") from error
    if smoke.get("gpu_verified") is not True:
        raise ValueError("GPU smoke result is missing gpu_verified=true")
    if smoke.get("rows_scored") != 8:
        raise ValueError("GPU smoke result must report exactly 8 scored rows")
    digest = smoke.get("prediction_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("GPU smoke result has an invalid prediction SHA-256")
    return hashlib.sha256(smoke_path.read_bytes()).hexdigest()


def run_suite(
    repo_root: Path,
    input_data_root: Path | None,
    working_root: Path,
    suite: str,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    working_root = Path(working_root)
    report: dict[str, Any] = {
        "schema_version": 1,
        "suite": suite,
        "status": "failed",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": _repository_provenance(repo_root),
        "commands": [],
        "artifacts": {},
    }
    child_env = os.environ.copy()
    report_path = working_root / "kaggle_test_report.json"

    try:
        if suite not in {"synthetic", "real"}:
            raise ValueError("suite must be synthetic or real")

        working_root.mkdir(parents=True, exist_ok=True)
        report["preflight"] = preflight(working_root, require_gpu=True)
        if report["preflight"].get("gpu_verified") is not True:
            raise RuntimeError("required Kaggle GPU preflight did not verify CUDA")

        if suite == "real":
            if input_data_root is None:
                raise ValueError("real suite requires private input data root")
            dataset_version = os.environ.get("RECSYS_KAGGLE_DATASET_VERSION")
            if not dataset_version:
                raise ValueError("RECSYS_KAGGLE_DATASET_VERSION is required for real suite")

            data_root = working_root / "data"
            report["source_paths"] = prepare_kaggle_data_root(input_data_root, data_root)
            source_manifest_path = data_root / "canonical" / "source_manifest.json"
            if not source_manifest_path.is_file():
                raise FileNotFoundError(
                    "real suite requires prepared canonical/source_manifest.json"
                )
            try:
                source_manifest = json.loads(
                    source_manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid canonical source manifest: {error}") from error
            fingerprint = source_manifest.get("fingerprint")
            if not isinstance(fingerprint, str) or not fingerprint:
                raise ValueError("canonical source manifest requires a fingerprint")
            manifest_sha = hashlib.sha256(source_manifest_path.read_bytes()).hexdigest()
            report["dataset_version"] = dataset_version
            report["source_manifest"] = {
                "fingerprint": fingerprint,
                "sha256": manifest_sha,
            }
            child_env["RECSYS_VNPT_DATA_ROOT"] = str(data_root.resolve())

        commands = [
            [sys.executable, "-m", "pytest", "-q"],
            [
                sys.executable,
                "-m",
                "experiments.kaggle_gpu_smoke",
                "--output-dir",
                str(working_root / "smoke"),
            ],
        ]
        if suite == "real":
            commands.append(
                [
                    sys.executable,
                    "all/infer.py",
                    "--root",
                    str(repo_root),
                    "--model-family",
                    "per_rule_popularity",
                    "--rule-ordering",
                    "count",
                    "--content-ordering",
                    "hand_sum",
                    "--num-profiles",
                    "1000",
                    "--candidate-budget",
                    "1000",
                    "--warm-repetitions",
                    "0",
                    "--output-dir",
                    str(working_root / "dry_run_output"),
                    "--dry-run",
                ]
            )

        for index, argv in enumerate(commands):
            command_result = run_command(argv, cwd=repo_root, env=child_env)
            report["commands"].append(command_result)
            if command_result["exit_code"] != 0:
                raise RuntimeError(
                    f"command failed with exit code {command_result['exit_code']}: {argv[1:3]}"
                )
            if index == 1:
                smoke_path = working_root / "smoke" / "gpu_smoke.json"
                report["artifacts"]["gpu_smoke_sha256"] = _validate_smoke_result(
                    smoke_path
                )
            if suite == "real" and index == 2:
                report["dry_run_output_dir"] = str(
                    (working_root / "dry_run_output").resolve()
                )

        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_report_atomic(report_path, report)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--input-data-root", type=Path)
    parser.add_argument("--working-root", required=True, type=Path)
    parser.add_argument("--suite", choices=("synthetic", "real"), required=True)
    args = parser.parse_args(argv)
    report = run_suite(
        args.repo_root,
        args.input_data_root,
        args.working_root,
        args.suite,
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
