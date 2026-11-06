"""Production entry points.

    python -m recsys_vnpt.batch_cli preflight        --config CFG --batch-id ID
    python -m recsys_vnpt.batch_cli run-inference    --config CFG --batch-id ID
    python -m recsys_vnpt.batch_cli run-retrain      --config CFG --batch-id ID
    python -m recsys_vnpt.batch_cli qualify-hardware --config CFG --batch-id ID
    python -m recsys_vnpt.batch_cli make-ready       --batch-dir DIR --data-cutoff ISO

Success prints one JSON document on stdout and exits 0. A failure prints
``{"error_code": ..., "message": ...}`` on stderr and exits with the code listed
in ``batch_contracts.EXIT_CODES``; progress goes to stderr as JSON lines.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from recsys_vnpt.batch_contracts import BatchError, load_and_validate_ready, load_config, write_ready_manifest


def _preflight(config, batch_id: str) -> dict:
    """Checks with no side effects: the input verifies and a model is servable."""
    from recsys_vnpt.batch_runner import _check_batch_id, _promoted, _validate_input

    _check_batch_id(batch_id)
    manifest = _validate_input(config, batch_id)
    _, bundle = _promoted(config)
    return {
        "batch_id": batch_id,
        "input_fingerprint": manifest.fingerprint,
        "data_cutoff": manifest.data_cutoff.isoformat(),
        "files": len(manifest.files),
        **bundle,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run-inference", "run-retrain", "qualify-hardware"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--batch-id", required=True)
    ready = commands.add_parser("make-ready", help="hash a finished input batch and write _READY.json")
    ready.add_argument("--batch-dir", required=True)
    ready.add_argument("--data-cutoff", required=True, help="ISO-8601 with offset, e.g. 2026-09-22T23:59:59+07:00")
    args = parser.parse_args(argv)

    try:
        if args.command == "make-ready":
            cutoff = datetime.fromisoformat(args.data_cutoff)
            if cutoff.tzinfo is None:
                raise BatchError("INPUT_INTEGRITY", "--data-cutoff needs a timezone offset")
            path = write_ready_manifest(args.batch_dir, cutoff)
            manifest = load_and_validate_ready(Path(args.batch_dir), Path(args.batch_dir).name)
            result = {"ready": str(path), "input_fingerprint": manifest.fingerprint, "files": len(manifest.files)}
        else:
            from recsys_vnpt import batch_runner

            config = load_config(args.config)
            result = {
                "preflight": _preflight,
                "run-inference": batch_runner.run_inference_batch,
                "run-retrain": batch_runner.run_retrain_batch,
                "qualify-hardware": batch_runner.qualify_hardware,
            }[args.command](config, args.batch_id)
    except BatchError as error:
        print(json.dumps({"error_code": error.code, "message": error.message}), file=sys.stderr)
        return error.exit_code
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if args.command == "qualify-hardware" and result.get("verdict") not in (None, "pass"):
        return 31
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
