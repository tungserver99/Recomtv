from __future__ import annotations

import json
import os
import socket
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _encode(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _process_is_alive(pid: int, host: str) -> bool:
    if host.casefold() != socket.gethostname().casefold():
        return True  # Remote owners cannot be safely judged from this process.
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class StateStore:
    """SQLite-backed coordination state shared by Streamlit and scheduler."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'production',
                    data_end_at TEXT,
                    slot TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS requests_pending ON requests(status, created_at);
                CREATE TABLE IF NOT EXISTS jobs (
                    batch_id TEXT PRIMARY KEY,
                    scheduled_at TEXT NOT NULL,
                    batch_started_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    actual_start TEXT,
                    actual_end TEXT,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    report_json TEXT NOT NULL DEFAULT '{}',
                    error_code TEXT,
                    error_message TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS current_job (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    batch_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    owner_pid INTEGER,
                    owner_host TEXT
                );
                """
            )
            request_columns = {row["name"] for row in connection.execute("PRAGMA table_info(requests)")}
            if "kind" not in request_columns:
                connection.execute(
                    "ALTER TABLE requests ADD COLUMN kind TEXT NOT NULL DEFAULT 'production'"
                )
            if "data_end_at" not in request_columns:
                connection.execute("ALTER TABLE requests ADD COLUMN data_end_at TEXT")
            job_columns = {row["name"] for row in connection.execute("PRAGMA table_info(jobs)")}
            if "data_cutoff" in job_columns and "batch_started_at" not in job_columns:
                connection.executescript(
                    """
                    BEGIN IMMEDIATE;
                    ALTER TABLE jobs RENAME TO jobs_legacy;
                    CREATE TABLE jobs (
                        batch_id TEXT PRIMARY KEY,
                        scheduled_at TEXT NOT NULL,
                        batch_started_at TEXT NOT NULL,
                        status TEXT NOT NULL,
                        actual_start TEXT,
                        actual_end TEXT,
                        attempt INTEGER NOT NULL DEFAULT 0,
                        report_json TEXT NOT NULL DEFAULT '{}',
                        error_code TEXT,
                        error_message TEXT,
                        updated_at TEXT NOT NULL
                    );
                    INSERT INTO jobs(batch_id,scheduled_at,batch_started_at,status,actual_start,actual_end,
                        attempt,report_json,error_code,error_message,updated_at)
                    SELECT batch_id,scheduled_at,data_cutoff,status,actual_start,actual_end,
                        attempt,report_json,error_code,error_message,updated_at FROM jobs_legacy;
                    DROP TABLE jobs_legacy;
                    COMMIT;
                    """
                )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(current_job)")}
            if "owner_pid" not in columns:
                connection.execute("ALTER TABLE current_job ADD COLUMN owner_pid INTEGER")
            if "owner_host" not in columns:
                connection.execute("ALTER TABLE current_job ADD COLUMN owner_host TEXT")

    def get_schedule(self, default: dict[str, Any]) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT value_json FROM settings WHERE key='schedule'").fetchone()
            if row:
                return json.loads(row["value_json"])
            connection.execute(
                "INSERT OR IGNORE INTO settings(key, value_json, updated_at) VALUES('schedule', ?, ?)",
                (_encode(default), _now()),
            )
            row = connection.execute("SELECT value_json FROM settings WHERE key='schedule'").fetchone()
            return json.loads(row["value_json"])

    def save_schedule(self, schedule: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO settings(key, value_json, updated_at) VALUES('schedule', ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at",
                (_encode(schedule), _now()),
            )

    def get_data_root(self, default: str | Path) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value_json FROM settings WHERE key='data_root'"
            ).fetchone()
            if row:
                return str(json.loads(row["value_json"]))
            value = str(Path(default).expanduser().resolve())
            connection.execute(
                "INSERT OR IGNORE INTO settings(key, value_json, updated_at) VALUES('data_root', ?, ?)",
                (_encode(value), _now()),
            )
            return value

    def save_data_root(self, value: str | Path) -> str:
        raw = str(value).strip()
        if not raw:
            raise ValueError("data_root cannot be empty")
        candidate = Path(raw).expanduser().resolve()
        if not candidate.is_dir():
            raise ValueError(f"data_root must be an existing directory: {candidate}")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO settings(key, value_json, updated_at) VALUES('data_root', ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at",
                (_encode(str(candidate)), _now()),
            )
        return str(candidate)

    @staticmethod
    def _validate_data_settings(settings: dict[str, Any]) -> dict[str, int]:
        fields = (
            "time_train", "time_valid", "time_test", "max_profiles",
            "gate_evaluation_profiles",
        )
        values = {name: settings[name] for name in fields}
        for name, value in values.items():
            minimum = 0 if name == "max_profiles" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                qualifier = "non-negative" if name == "max_profiles" else "positive"
                raise ValueError(f"{name} must be a {qualifier} integer")
        return values

    def get_data_settings(self, default: dict[str, Any]) -> dict[str, int]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value_json FROM settings WHERE key='data_settings'"
            ).fetchone()
            if row:
                stored = json.loads(row["value_json"])
                return self._validate_data_settings({**default, **stored})
            values = self._validate_data_settings(default)
            connection.execute(
                "INSERT OR IGNORE INTO settings(key, value_json, updated_at) VALUES('data_settings', ?, ?)",
                (_encode(values), _now()),
            )
            return values

    def save_data_settings(self, settings: dict[str, Any]) -> None:
        values = self._validate_data_settings(settings)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO settings(key, value_json, updated_at) VALUES('data_settings', ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at",
                (_encode(values), _now()),
            )

    def enqueue_request(
        self,
        slot: datetime,
        kind: str = "production",
        *,
        batch_id: str | None = None,
        data_end_at: str | None = None,
    ) -> str:
        if slot.tzinfo is None:
            raise ValueError("request slot must be timezone-aware")
        if kind not in {"production", "prepare", "retrain", "inference"}:
            raise ValueError("request kind must be production, prepare, retrain or inference")
        batch_id = batch_id or slot.strftime("%Y%m%d_%H%M")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT request_id FROM requests WHERE batch_id=? AND kind=? AND status IN ('PENDING','CLAIMED') ORDER BY created_at LIMIT 1",
                (batch_id, kind),
            ).fetchone()
            if existing:
                connection.commit()
                return existing["request_id"]
            queued = connection.execute(
                "SELECT request_id,kind FROM requests WHERE status IN ('PENDING','CLAIMED') ORDER BY created_at LIMIT 1"
            ).fetchone()
            if queued:
                if queued["kind"] != kind:
                    connection.rollback()
                    raise ValueError("another dashboard request of a different kind is already queued")
                connection.commit()
                return queued["request_id"]
            request_id = uuid.uuid4().hex
            connection.execute(
                "INSERT INTO requests(request_id,batch_id,kind,data_end_at,slot,created_at,status) VALUES(?,?,?,?,?,?, 'PENDING')",
                (request_id, batch_id, kind, data_end_at, slot.isoformat(), _now()),
            )
            connection.commit()
            return request_id

    def claim_next_request(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM requests WHERE status='PENDING' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            connection.execute(
                "UPDATE requests SET status='CLAIMED' WHERE request_id=? AND status='PENDING'",
                (row["request_id"],),
            )
            connection.commit()
            return dict(row)

    def queued_requests(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM requests WHERE status IN ('PENDING','CLAIMED') ORDER BY created_at LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def finish_request(self, request_id: str, error: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE requests SET status=?, error=? WHERE request_id=?",
                ("FAILED" if error else "COMPLETED", error, request_id),
            )

    def try_start_job(self, batch_id: str, slot: datetime, batch_started_at: datetime) -> bool:
        if slot.tzinfo is None or batch_started_at.tzinfo is None:
            raise ValueError("slot and batch_started_at must be timezone-aware")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute("SELECT batch_id FROM current_job WHERE singleton=1").fetchone()
            if active:
                connection.commit()
                return False
            current = connection.execute("SELECT status FROM jobs WHERE batch_id=?", (batch_id,)).fetchone()
            if current and current["status"] == "SUCCESS":
                connection.commit()
                return False
            connection.execute(
                "INSERT INTO jobs(batch_id,scheduled_at,batch_started_at,status,actual_start,attempt,report_json,updated_at) "
                "VALUES(?,?,?,'RUNNING',?,1,'{}',?) ON CONFLICT(batch_id) DO UPDATE SET "
                "status='RUNNING', actual_start=excluded.actual_start, actual_end=NULL, "
                "attempt=jobs.attempt+1, error_code=NULL, error_message=NULL, updated_at=excluded.updated_at",
                (batch_id, slot.isoformat(), batch_started_at.isoformat(), _now(), _now()),
            )
            connection.execute(
                "INSERT INTO current_job(singleton,batch_id,started_at,owner_pid,owner_host) VALUES(1,?,?,?,?)",
                (batch_id, _now(), os.getpid(), socket.gethostname()),
            )
            connection.commit()
            return True

    def record_skipped(self, batch_id: str, slot: datetime, batch_started_at: datetime, reason: str) -> None:
        if slot.tzinfo is None or batch_started_at.tzinfo is None:
            raise ValueError("slot and batch_started_at must be timezone-aware")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO jobs(batch_id,scheduled_at,batch_started_at,status,error_message,updated_at) "
                "VALUES(?,?,?,'SKIPPED_OVERLAP',?,?) ON CONFLICT(batch_id) DO NOTHING",
                (batch_id, slot.isoformat(), batch_started_at.isoformat(), reason, _now()),
            )

    def update_job(self, batch_id: str, report: dict[str, Any], status: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET report_json=?, status=COALESCE(?,status), updated_at=? WHERE batch_id=?",
                (_encode(report), status, _now(), batch_id),
            )

    def finish_job(
        self,
        batch_id: str,
        report: dict[str, Any],
        status: str = "SUCCESS",
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE jobs SET status=?, actual_end=?, report_json=?, error_code=?, error_message=?, updated_at=? WHERE batch_id=?",
                (status, _now(), _encode(report), error_code, error_message, _now(), batch_id),
            )
            connection.execute("DELETE FROM current_job WHERE singleton=1 AND batch_id=?", (batch_id,))
            connection.commit()

    def get_job(self, batch_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE batch_id=?", (batch_id,)).fetchone()
            return self._job(row) if row else None

    def recover_interrupted(self, owner_check=None, scheduler_interrupted: bool = False) -> list[str]:
        if owner_check is None:
            owner_check = _process_is_alive
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute("SELECT batch_id,owner_pid,owner_host FROM current_job WHERE singleton=1").fetchall()
            ids = []
            for row in rows:
                if row["owner_pid"] is None or row["owner_host"] is None or owner_check(row["owner_pid"], row["owner_host"]):
                    continue
                batch_id = row["batch_id"]
                ids.append(batch_id)
                connection.execute(
                    "UPDATE jobs SET status='INTERRUPTED', actual_end=?, error_message='scheduler stopped while job was running', updated_at=? WHERE batch_id=?",
                    (_now(), _now(), batch_id),
                )
                connection.execute("DELETE FROM current_job WHERE singleton=1 AND batch_id=?", (batch_id,))
            if scheduler_interrupted:
                connection.execute(
                    "UPDATE requests SET status='INTERRUPTED', error='scheduler stopped after claiming request' WHERE status='CLAIMED'"
                )
            connection.commit()
            return ids

    def current_job(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT j.* FROM current_job c JOIN jobs j ON j.batch_id=c.batch_id WHERE c.singleton=1"
            ).fetchone()
            return self._job(row) if row else None

    def jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY scheduled_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [self._job(row) for row in rows]

    @staticmethod
    def _job(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["report"] = json.loads(value.pop("report_json"))
        if "batch_started_at" not in value and "data_cutoff" in value:
            value["batch_started_at"] = value["data_cutoff"]
        if "batch_started_at" not in value and value["report"].get("data_cutoff"):
            value["batch_started_at"] = value["report"]["data_cutoff"]
        if "batch_started_at" not in value["report"] and value["report"].get("data_cutoff"):
            value["report"]["batch_started_at"] = value["report"].pop("data_cutoff")
        return value

