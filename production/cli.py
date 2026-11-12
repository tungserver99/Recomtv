from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from production.config import load_production_config
from production.orchestrator import run_production
from recsys_vnpt.batch_contracts import BatchError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one RecSys-VNPT production slot")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="prepare, retrain, gate, infer, and publish")
    run.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    run.add_argument("--slot", help="explicit scheduled slot with timezone offset")
    run.add_argument("--data-root", help="raw data folder for this run; overrides dashboard/config setting")
    run.add_argument(
        "--prepare-only",
        action="store_true",
        help="stage input and prepare the dataset without retraining or inference",
    )
    args = parser.parse_args(argv)
    try:
        config = load_production_config(args.config)
        selected_data_root = (
            config.resolve_data_root(args.data_root) if args.data_root else None
        )
        result = run_production(
            config,
            args.slot,
            prepare_only=args.prepare_only,
            data_root=selected_data_root,
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

