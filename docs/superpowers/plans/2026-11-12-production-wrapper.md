# Production Wrapper, Scheduler, and Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an operational wrapper for scheduled/manual GxEp retraining and inference, with exact cutoff staging, SQLite state, a separate APScheduler process, and Streamlit dashboard.

**Architecture:** Keep `recsys_vnpt` algorithm code intact. Add focused modules under `production/` that reuse current batch config, manifest, status, locks, promotion, run functions, and publishing.

**Tech Stack:** Python stdlib (`sqlite3`, `zoneinfo`), pandas/pyarrow already used by the pipeline, APScheduler 3.x, Streamlit, pytest.

**Spec:** `docs/superpowers/specs/2026-11-12-production-wrapper-design.md`

## Global Constraints

- Keep production combination fixed at `GxEp`.
- Default timezone `Asia/Ho_Chi_Minh`, run times `00:00` and `12:00`, cutoff offset 4 hours.
- Filter event data by exact `create_date <= data_cutoff`; fail closed on date-only or unusable timestamps.
- Reuse `_READY.json`, `BatchStatus`, `BatchLock`, existing retrain/inference and publishing paths.
- Do not change algorithms, ML dependency versions, Docker/Nuitka, or run GPU work locally.

## Review Focus

- Changed source files for the same slot: ensure a committed snapshot is never silently replaced; test source conflict.
- Partial-day timestamps and timezone offsets: ensure cutoff is precise and ambiguous events fail closed; test both.
- Restart/overlap around scheduled slots: ensure no duplicate concurrent work and no catch-up; test atomic claims/lock.
- Quality-gate failure without a prior model: ensure no inference is reported successful; test explicit failure.
- Demo dashboard actions: ensure they cannot write production state; test read-only demo data behavior.

---

### Task 1: Slot configuration and cutoff-safe staging

**Files:** Create `production/config.py`, `production/cutoff.py`, `production/staging.py`, `production/__init__.py`, `production/config.json`; create `tests/test_production_schedule.py`, `tests/test_production_staging.py`.

**Interfaces:** `resolve_slot(slot, timezone, run_times, now) -> datetime`; `batch_id_for_slot(slot) -> str`; `data_cutoff_for_slot(slot, offset_hours) -> datetime`; `load_production_config(path) -> ProductionConfig`; `build_input_snapshot(config, batch_id, cutoff) -> StagingResult`.

- [ ] Add tests for the two defaults, slot validation/normalization, cutoff examples, exact timestamp inclusion/exclusion, day-only rejection, CSV and Parquet, manifest validation, retry reuse and source conflict.
- [ ] Run those tests and confirm missing module failures.
- [ ] Implement path resolution and runtime `BatchConfig` JSON that excludes wrapper-only fields and fixes `combination` to `GxEp`.
- [ ] Implement streaming duration filtering into temporary staging and source snapshot conflict detection, then reuse `write_ready_manifest` and `load_and_validate_ready`.
- [ ] Run `pytest -q tests/test_production_schedule.py tests/test_production_staging.py`.

### Task 2: SQLite coordination and single-host scheduler

**Files:** Create `production/state_store.py`, `production/schedule.py`, `production/scheduler.py`; create `tests/test_production_scheduler.py`.

**Interfaces:** `StateStore(path)` with schedule read/write, run request enqueue/claim, job start/update/finish, history/current queries; `build_scheduler(config, store, run_job, clock)`.

- [ ] Test schedule persistence, request idempotency, atomic job claims, duplicate-slot suppression, disabled schedule behavior, and slot selection without catch-up.
- [ ] Implement SQLite schema and transactional claims; use persistent `BatchLock` from the pipeline work root for production execution serialization.
- [ ] Implement APScheduler cron jobs with explicit slot arguments and short polling for config/manual request changes.
- [ ] Run `pytest -q tests/test_production_scheduler.py`.

### Task 3: Orchestration and CLI

**Files:** Create `production/orchestrator.py`, `production/cli.py`; create `tests/test_production_orchestrator.py`.

**Interfaces:** `run_production(config, slot, store=None, source_builder=..., retrain=..., infer=...) -> dict` and `main(argv=None) -> int`.

- [ ] Test quality-gate pass, gate rejection with usable prior model fallback, gate rejection without a usable model, unrelated training exceptions, inference failure, successful report, and retry of published slot.
- [ ] Implement locked orchestration, BatchStatus/report reading, only-special-case `QUALITY_GATE_FAILED`, prior model validation, idempotent published-slot handling, and atomic wrapper JSON reports.
- [ ] Implement `python -m production.cli run [--slot ISO]` with configured timezone/schedule validation and structured errors.
- [ ] Run `pytest -q tests/test_production_orchestrator.py`.

### Task 4: Streamlit dashboard, dependencies, and operator documentation

**Files:** Create `production/dashboard/__init__.py`, `production/dashboard/app.py`, `production/README.md`; modify `requirements.txt`; create dashboard/helper tests in `tests/test_production_dashboard.py`.

- [ ] Test status mapping from real `BatchStatus`, simulated scenarios, and that demo interactions do not mutate SQLite or production files.
- [ ] Implement Overview, Configuration, and Job History pages, Run now request enqueue, refresh, phase details/logs, and clearly labeled demo scenarios.
- [ ] Add exact compatible Streamlit/APScheduler pins without changing existing ML pins.
- [ ] Document server startup, test config, production config switch, data completeness and partial-day temporal-window limitations.
- [ ] Run all production unit tests and syntax checks; do not run training or inference.
