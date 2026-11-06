from __future__ import annotations

import argparse
from pathlib import Path

from recsys_vnpt.audit import audit_project


def write_audit_report(root: str | Path = ".", output: str | Path = "AUDIT_REPORT.md") -> Path:
    report = audit_project(root)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.to_markdown(), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a RecSys-VNPT artifact audit report.")
    parser.add_argument("--root", default=".", help="Project root to audit.")
    parser.add_argument("--output", default="AUDIT_REPORT.md", help="Markdown output path.")
    args = parser.parse_args()
    path = write_audit_report(args.root, args.output)
    print(f"Wrote audit report to {path}")


if __name__ == "__main__":
    main()
