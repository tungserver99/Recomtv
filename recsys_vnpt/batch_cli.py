"""Production entry points.

    python -m recsys_vnpt.batch_cli preflight        --config CFG --batch-id ID
    python -m recsys_vnpt.batch_cli prepare-data     --config CFG --batch-id ID
    python -m recsys_vnpt.batch_cli run-inference    --config CFG --batch-id ID
    python -m recsys_vnpt.batch_cli run-retrain      --config CFG --batch-id ID
    python -m recsys_vnpt.batch_cli qualify-hardware --config CFG --batch-id ID
    python -m recsys_vnpt.batch_cli make-ready       --batch-dir DIR

Success prints one JSON document on stdout and exits 0. A failure prints
``{"error_code": ..., "message": ...}`` on stderr and exits with the code listed
in ``batch_contracts.EXIT_CODES``; progress goes to stderr as JSON lines.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from recsys_vnpt.batch_contracts import BatchError, load_and_validate_ready, load_config, write_ready_manifest


def _preflight(config, batch_id: str) -> dict:
    """Checks with no side effects: the input verifies and a model is servable."""
    from recsys_vnpt.batch_runner import _check_batch_id, _promoted, _validate_input
    from recsys_vnpt.prepared_dataset import load_prepared_dataset

    _check_batch_id(batch_id)
    manifest = _validate_input(config, batch_id)
    prepared = load_prepared_dataset(config, manifest)
    _, bundle = _promoted(config)
    return {
        "batch_id": batch_id,
        "input_fingerprint": manifest.fingerprint,
        "batch_started_at": manifest.batch_started_at.isoformat(),
        "files": len(manifest.files),
        "prepared_dataset": str(prepared.manifest_path),
        **bundle,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "prepare-data", "run-inference", "run-retrain", "qualify-hardware"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--batch-id", required=True)
        if name in {"run-inference", "run-retrain"}:
            command.add_argument("--data-end-at", help="Optional ISO cutoff, e.g. 2025-02-15T23:59:59+07:00")
            command.add_argument("--run-id", help="Reuse an earlier run ID for an idempotent retry")
        if name == "run-inference":
            command.add_argument("--model-path", help="Use this verified model bundle directory instead of the promoted model")
    ready = commands.add_parser("make-ready", help="hash a finished input batch and write _READY.json")
    ready.add_argument("--batch-dir", required=True)
    args = parser.parse_args(argv)
    active_run_id: str | None = None

    try:
        if args.command == "make-ready":
            path = write_ready_manifest(args.batch_dir)
            manifest = load_and_validate_ready(Path(args.batch_dir), Path(args.batch_dir).name)
            result = {
                "ready": str(path),
                "batch_started_at": manifest.batch_started_at.isoformat(),
                "input_fingerprint": manifest.fingerprint,
                "files": len(manifest.files),
            }
        else:
            from recsys_vnpt import batch_runner

            config = load_config(args.config)
            if args.command in {"run-inference", "run-retrain"}:
                from recsys_vnpt.canonical import normalize_data_end_at

                if args.data_end_at is not None:
                    normalize_data_end_at(args.data_end_at)
                active_run_id = args.run_id or uuid.uuid4().hex
                if args.command == "run-inference":
                    result = batch_runner.run_inference_batch(
                        config,
                        args.batch_id,
                        args.data_end_at,
                        run_id=active_run_id,
                        model_path=args.model_path,
                    )
                else:
                    result = batch_runner.run_retrain_batch(
                        config, args.batch_id, args.data_end_at, run_id=active_run_id
                    )
            else:
                result = {
                    "preflight": _preflight,
                    "prepare-data": batch_runner.run_prepare_data_batch,
                    "qualify-hardware": batch_runner.qualify_hardware,
                }[args.command](config, args.batch_id)
    except BatchError as error:
        payload = {"error_code": error.code, "message": error.message}
        if active_run_id:
            payload["run_id"] = active_run_id
        print(json.dumps(payload), file=sys.stderr)
        return error.exit_code
    except ValueError as error:
        payload = {"error_code": "CONFIG_INVALID", "message": str(error)}
        if active_run_id:
            payload["run_id"] = active_run_id
        print(json.dumps(payload), file=sys.stderr)
        return 14
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if args.command == "qualify-hardware" and result.get("verdict") not in (None, "pass"):
        return 31
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
