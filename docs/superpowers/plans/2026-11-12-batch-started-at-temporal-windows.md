# Batch Timestamp and Temporal Windows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `data_cutoff` with stable `batch_started_at` metadata and configurable hour-based temporal windows without changing the candidate training, test, quality gate, or promotion sequence.

**Architecture:** Keep timestamp lifecycle in batch contracts and the READY manifest; have production orchestration persist its timestamp before staging and reuse it on retries. Keep window arithmetic in `temporal.py`, source precision checks in preparation, and pass validated `BatchConfig` hours through `_prepare()` to `build_temporal_project()`.

**Tech Stack:** Python, pandas, SQLite, pytest, existing SHA-256 and batch-manifest code.

**Spec:** `docs/superpowers/specs/2026-11-12-batch-started-at-temporal-windows.md`

## Global Constraints

- Timestamp values must carry a timezone; a new batch timestamp is recorded once and reused on retry.
- `batch_started_at` must never be used as an interaction or serving cutoff.
- Temporal durations are positive integer hours; three train windows each use `time_train`.
- Preserve `window_mode="pinned"`, current window names, SHA-256 input verification, and old metadata reads.
- Do not add final retraining. Do not run training, inference, benchmarks, or download model weights locally.

## Review Focus

- Day-only `create_date` or `filename_date` fallback must not enter hourly evaluation: Task 3 tests rejection.
- Existing batch timestamp must survive a repeated make-ready or failed production retry: Tasks 1–2 test reuse.
- An old delivery with only `data_cutoff` must still block an older batch: Task 2 tests legacy ordering.
- Changing hours must invalidate temporal caches while pinned windows remain identical: Task 3 tests both.
- A window with no positives must fail before empty training/evaluation artifacts are accepted: Task 3 tests the error.

---

### Task 1: Batch timestamp contract and CLI

**Files:** `recsys_vnpt/batch_contracts.py`, `recsys_vnpt/batch_cli.py`, `recsys_vnpt/smoke_fixture.py`, `tests/test_batch_contracts.py`, `tests/test_batch_cli.py`

**Interfaces:** `ReadyManifest.batch_started_at: datetime`; `write_ready_manifest(batch_dir, batch_id=None, batch_started_at=None)`; READY v2 writes `batch_started_at`, legacy v1 `data_cutoff` loads as that timestamp.

- [ ] Add tests for timezone-aware timestamp round-trip, auto assignment, idempotent re-make-ready, legacy READY fallback, and rejection of timezone-naive timestamp.
- [ ] Run those focused tests and confirm the expected contract failures.
- [ ] Implement the READY v2 contract while preserving `created_at` as manifest completion time and all file/hash checks.
- [ ] Remove `make-ready --data-cutoff`; report `batch_started_at` from preflight.
- [ ] Run focused contract and CLI tests.

### Task 2: Publication, bundles, and production orchestration

**Files:** `recsys_vnpt/batch_runner.py`, `recsys_vnpt/export.py`, `production/config.py`, `production/cutoff.py`, `production/orchestrator.py`, `production/staging.py`, `production/state_store.py`, `production/dashboard/app.py`, `production/config*.json`, related `tests/test_batch_runner.py` and production tests.

**Interfaces:** New delivery/bundle/report records use `batch_started_at`; old metadata resolves through a compatibility helper. Production job state stores `batch_started_at` and migrates legacy `data_cutoff` values.

- [ ] Add CPU-only tests for batch timestamp metadata, legacy/current delivery ordering, old DB history migration, retry reuse, and source rows surviving production staging unchanged.
- [ ] Run only those focused tests; do not invoke model training or serving.
- [x] Replace new `data_cutoff` writes/reads with `batch_started_at`, preserve legacy reads, derive reference year and output filename date from the batch timestamp, and keep serving cutoff interaction-derived.
- [ ] Remove production schedule cutoff-offset/staging filter behavior; preserve slot scheduling and snapshot integrity.
- [ ] Run focused contract/production tests that do not execute training or inference.

### Task 3: Hour-based temporal windows and precision checks

**Files:** `recsys_vnpt/batch_contracts.py`, `recsys_vnpt/batch_runner.py`, `recsys_vnpt/prepare.py`, `recsys_vnpt/pipeline_v2.py`, `recsys_vnpt/temporal.py`, `production/config*.json`, `tests/test_temporal_sampling_v3.py`, related prepare/config tests.

**Interfaces:** `BatchConfig(time_train=24, time_valid=24, time_test=24)`; `derive_relative_temporal_windows(max_event_date, time_train=24, time_valid=24, time_test=24)`; `build_temporal_project(..., time_train=24, time_valid=24, time_test=24)`.

- [ ] Add tests for exact inclusive intervals across midnight/month boundary, unequal durations, validation of all three hours, timestamp precision rejection, positive-window checks, cache fingerprint changes, and pinned-mode stability.
- [ ] Run the focused tests and confirm the expected failures.
- [x] Implement exact rolling windows without `.normalize()`, validate source precision before canonical timestamp fallback, pass configured hours, persist the settings in windows/build metadata and cache fingerprint, and reject empty-positive windows.
- [ ] Run focused temporal/config/prepare tests; do not run model fitting.

### Task 4: Documentation and final verification

**Files:** `OPERATIONS.md`, `SPEC.md`, `TODO.md`, `production/README.md`, tests as needed.

- [ ] Update operator schema/examples, legacy compatibility notes, and batch reference-date/output naming.
- [ ] Run the focused safe test set and Python compile checks.
- [x] Review all remaining `data_cutoff` references and confirm they are either legacy readers/migrations or unrelated historical notes.
- [ ] Confirm the train/evaluate/gate/promote call flow has no final retraining and local checks did not execute training, inference, benchmark, or weight downloads.
