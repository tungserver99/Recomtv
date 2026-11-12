# Configurable Raw Data Root Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow an operator to select the raw MyTV data directory in the production CLI or dashboard.

**Architecture:** ProductionConfig exposes `data_root` while accepting legacy `source_root` config keys. The dashboard persists the selected path through StateStore; the CLI accepts a per-run `--data-root` override. `run_production` resolves the effective raw path before calling the existing immutable input snapshot builder.

**Tech Stack:** Python, argparse, dataclasses, SQLite, Streamlit.

**Spec:** `docs/superpowers/specs/2026-11-13-configurable-data-root.md`

## Global Constraints

- The selected path contains raw inputs and remains separate from incoming, work, delivery, and model registry paths.
- Preserve input snapshot staging, fingerprints, `_READY.json`, and prepared dataset identity.
- Keep low-level `recsys_vnpt.batch_cli` commands consuming committed batches.
- Preserve configs that still use `source_root`.

## Review Focus

- CLI explicit path overrides the persisted dashboard path; CLI without override uses persisted/default config path.
- Dashboard-saved path is used by both prepare-only and scheduled production requests.
- Missing, non-directory, and overlapping roots fail clearly.
- Legacy config keys continue to load.
- Data root must not leak into the prepared artifacts output root.

### Task 1: Config and StateStore path persistence

**Files:** `production/config.py`, `production/state_store.py`

- [x] Expose `ProductionConfig.data_root`, retain `source_root` as an internal compatibility alias, accept legacy config keys, and resolve/reject invalid or overlapping roots.
- [x] Add SQLite `get_data_root` / `save_data_root` methods with non-empty existing-directory validation.

### Task 2: Production CLI and orchestrator selection

**Files:** `production/cli.py`, `production/orchestrator.py`

- [x] Add CLI `--data-root` and `--prepare-only`; explicit CLI path overrides the persisted/default folder for one run.
- [x] Apply the effective raw path to the existing snapshot builder, preserving immutable staging and prepared dataset identity.

### Task 3: Dashboard setting and docs

**Files:** `production/dashboard/app.py`, `production/config.json`, `production/config.test.json`, `production/README.md`.

- [x] Add a validated “Raw data folder” form saved to SQLite; dashboard-queued jobs use that value through the orchestrator.
- [x] Rename bundled config examples to `data_root` while keeping legacy config support; document CLI and dashboard usage.
- [x] Run `compileall` and `git diff --check`.

Implementation note: No tests were added or run for this feature, following the session instruction to only add/run tests when requested. The requested lightweight compile and diff checks passed.
