"""Batch status document, exclusive lock, and structured log.

``status.json`` is replaced atomically on every transition, so a monitor never
reads a half-written state. The lock is a directory created exclusively; a lock
left behind by a killed run is never removed automatically, because only an
operator can tell whether its owner is really gone.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from recsys_vnpt.batch_contracts import BatchError
from recsys_vnpt.runtime import atomic_write_text


PHASES = {
    "inference": (
        "WAITING_INPUT",
        "VALIDATING_INPUT",
        "LOADING_PREPARED_DATA",
        "LOADING_MODEL",
        "RUNNING_SHARDS",
        "MERGING",
        "VALIDATING_OUTPUT",
        "PUBLISHING",
        "PUBLISHED",
    ),
    "retrain": (
        "WAITING_INPUT",
        "VALIDATING_INPUT",
        "LOADING_PREPARED_DATA",
        "TRAINING",
        "EVALUATING",
        "GATING",
        "PROMOTED",
    ),
    "qualification": (
        "WAITING_INPUT",
        "VALIDATING_INPUT",
        "LOADING_PREPARED_DATA",
        "LOADING_MODEL",
        "MEASURING",
        "QUALIFIED",
    ),
    "prepare": (
        "WAITING_INPUT",
        "VALIDATING_INPUT",
        "PREPARING_DATA",
        "PREPARED",
    ),
}
# Terminal outcomes that are not the last phase of the happy path.
TERMINAL = {"FAILED", "REJECTED"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BatchStatus:
    """Forward-only phase tracking for one run of one batch."""

    def __init__(self, path: Path, batch_id: str, kind: str):
        self.path = Path(path)
        self.batch_id = batch_id
        self.kind = kind
        self.phases = PHASES[kind]
        self.log_path = self.path.with_name("log.jsonl")

    def read(self) -> dict[str, Any] | None:
        if not self.path.is_file():
            return None
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, document: dict[str, Any]) -> None:
        atomic_write_text(self.path, json.dumps(document, indent=2, sort_keys=True, default=str))

    def begin(self) -> None:
        previous = self.read() or {}
        self._write(
            {
                "schema_version": 1,
                "batch_id": self.batch_id,
                "kind": self.kind,
                "attempt": int(previous.get("attempt", 0)) + 1,
                "phase": self.phases[0],
                "history": [{"phase": self.phases[0], "at": _now()}],
                "details": {},
            }
        )
        self.log("begin", attempt=int(previous.get("attempt", 0)) + 1)

    def transition(self, phase: str, **details: Any) -> None:
        document = self.read()
        if document is None or document.get("phase") in TERMINAL:
            raise RuntimeError("transition outside an active run")
        current = self.phases.index(document["phase"])
        if phase not in self.phases or self.phases.index(phase) <= current:
            raise RuntimeError(f"illegal transition {document['phase']} -> {phase}")
        document["phase"] = phase
        document["history"].append({"phase": phase, "at": _now()})
        document["details"].update(details)
        self._write(document)
        self.log("phase", phase=phase, **details)

    def finish(self, outcome: str, **details: Any) -> None:
        """Record REJECTED/FAILED without the forward-only check."""
        document = self.read() or {"history": [], "details": {}}
        document["phase"] = outcome
        document["history"].append({"phase": outcome, "at": _now()})
        document["details"].update(details)
        self._write(document)
        self.log("finish", outcome=outcome, **details)

    def fail(self, error: BatchError) -> None:
        stage = (self.read() or {}).get("phase")
        if stage in TERMINAL:
            self.log("error", error_code=error.code, error=error.message)
            return
        self.finish("FAILED", error_code=error.code, error_stage=stage, error=error.message)

    @property
    def phase(self) -> str | None:
        return (self.read() or {}).get("phase")

    def log(self, event: str, **fields: Any) -> None:
        line = json.dumps(
            {"at": _now(), "batch_id": self.batch_id, "kind": self.kind, "event": event, **fields},
            default=str,
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, file=sys.stderr, flush=True)


class BatchLock:
    def __init__(self, locks_root: Path, name: str):
        self.path = Path(locks_root) / name

    def __enter__(self) -> "BatchLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.mkdir()
        except FileExistsError as error:
            owner = self.path / "owner.json"
            detail = owner.read_text(encoding="utf-8") if owner.is_file() else "unknown owner"
            raise BatchError(
                "BATCH_LOCKED",
                f"{self.path} is held ({detail}); remove it only after confirming "
                "that process is gone",
            ) from error
        atomic_write_text(
            self.path / "owner.json",
            json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "since": _now()}),
        )
        return self

    def __exit__(self, *_: object) -> None:
        (self.path / "owner.json").unlink(missing_ok=True)
        self.path.rmdir()
