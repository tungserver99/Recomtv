# Production Wrapper, Scheduler, and Dashboard Design

> Superseded on 2026-11-12 by `2026-11-12-batch-started-at-temporal-windows.md`:
> the later contract uses immutable `batch_started_at` metadata and does not
> filter duration rows by a batch cutoff.

## Intent

Provide VNPT operators with a local production wrapper around `recsys_vnpt`: schedule two daily GxEp retrain-and-inference runs, start a manual run, inspect live pipeline status and history, and retain inference service through a quality-gate rejection when a valid promoted model exists.

## Goals and constraints

- Preserve training, inference, evaluation, gate, model registry, output validation, and publishing behavior in the existing pipeline.
- Build cutoff-filtered event snapshots in `production/` and commit them with the existing `_READY.json` protocol.
- Keep Streamlit separate from the APScheduler process; SQLite carries schedules, requests, and job summaries.
- Use `BatchStatus`, `BatchLock`, existing promotion utilities, and atomic delivery publishing.
- Default to `Asia/Ho_Chi_Minh`, 00:00/12:00, four-hour cutoff offset, and fixed `GxEp`.
- No Docker, Nuitka, public API, distributed scheduler, or GPU run on local.

## Approaches

1. Extend `batch_runner` with production scheduling and UI behavior. This couples existing batch CLI behavior to operational concerns and is rejected.
2. Add a `production/` orchestration layer that stages data and calls existing batch functions. This keeps algorithms and legacy CLI intact and is selected.
3. Build a web service/queue around the batch code. This adds infrastructure beyond the single-host requirement and is rejected.

## Design

`production.config` resolves repo-relative paths once and writes a runtime-only batch JSON containing only `BatchConfig` keys, then calls `load_config`. `production.cutoff` resolves the requested scheduled slot in the configured zone; the stable batch ID and cutoff derive from that slot. `production.staging` copies required metadata and streams CSV/Parquet duration sources into a temporary batch, keeping rows whose exact `create_date` (used by the existing pipeline as event end) is at or before cutoff. Missing, invalid, or day-only event timestamps fail closed. It fingerprints source files, reuses a valid committed batch, and rejects changed source snapshots for a previously committed slot.

`production.orchestrator` uses a persistent global job lock and a SQLite claim, prepares/validates input, calls existing retraining and inference functions, handles only `QUALITY_GATE_FAILED` as a fallback case, verifies a promoted bundle, and writes a report. SQLite records schedule, requests, current job and history. APScheduler polls for schedule edits and queued requests, uses explicit scheduled slots, skips missed/overlapping slots, and never catches up downtime. Streamlit displays production status and BatchStatus phase/log files; demo scenarios remain in-memory/read-only and visibly marked simulated.

## Limits and assumptions

- The four-hour delay is an operational assumption, not upstream completeness proof.
- Only interaction rows with precise `create_date` are cutoff-certified; catalog, tags, and user metadata remain source snapshots.
- Relative temporal windows continue to use the latest event date in the data and may end on a partial day; this task does not change that algorithm.
- This is a single-host scheduler and assumes one scheduler process.
- Production UI is intended for localhost or a trusted internal network.
