from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from production.config import load_production_config
from production.orchestrator import run_batch_action, run_production
from recsys_vnpt.batch_contracts import BatchError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one independent RecSys-VNPT batch operation")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-data", help="stage input and prepare reusable artifacts only")
    prepare.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    prepare.add_argument("--slot", help="explicit scheduled slot with timezone offset")
    prepare.add_argument("--data-root", help="raw data folder; overrides dashboard/config setting")
    for name in ("run-retrain", "run-inference"):
        action = commands.add_parser(name, help=f"run only {name.removeprefix('run-')}")
        action.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
        action.add_argument("--batch-id", required=True, help="prepared input batch ID")
        action.add_argument("--data-end-at", help="optional ISO cutoff for a historical demo")
        action.add_argument("--run-id", help="reuse an earlier run ID to retry safely")
        if name == "run-inference":
            action.add_argument("--model-path", help="verified model bundle directory; defaults to the promoted model")
    args = parser.parse_args(argv)
    try:
        config = load_production_config(args.config)
        if args.command == "prepare-data":
            selected_data_root = config.resolve_data_root(args.data_root) if args.data_root else None
            result = run_production(config, args.slot, prepare_only=True, data_root=selected_data_root)
        else:
            action_args = {"data_end_at": args.data_end_at, "run_id": args.run_id}
            if args.command == "run-inference":
                action_args["model_path"] = args.model_path
            result = run_batch_action(
                config, args.batch_id, args.command.removeprefix("run-"), **action_args
            )
    except BatchError as error:
        print(json.dumps({"error_code": error.code, "message": error.message}), file=sys.stderr)
        return error.exit_code
    except (ValueError, OSError) as error:
        print(json.dumps({"error_code": "CONFIG_INVALID", "message": str(error)}), file=sys.stderr)
        return 14
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

