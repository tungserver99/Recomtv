from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SkippedFile:
    path: Path
    reason: str


def load_json_records(paths: Iterable[str | Path]) -> tuple[pd.DataFrame, list[SkippedFile]]:
    records: list[dict] = []
    skipped: list[SkippedFile] = []

    for raw_path in paths:
        path = Path(raw_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # reported to caller, never hidden
            skipped.append(SkippedFile(path=path, reason=str(exc)))
            continue

        if isinstance(payload, list):
            records.extend(item for item in payload if isinstance(item, dict))
        elif isinstance(payload, dict):
            records.append(payload)
        else:
            skipped.append(SkippedFile(path=path, reason=f"unsupported JSON root: {type(payload).__name__}"))

    return pd.DataFrame(records), skipped
