# Batch timestamps and temporal windows

## Goal

Use one immutable, timezone-aware `batch_started_at` to identify batch order and reference dates, and configure relative train, validation, and test windows in hours from the latest precise interaction timestamp. Keep the current train/evaluate/quality-gate/promote flow; do not add final retraining.

## Decisions

- New `_READY.json`, delivery, bundle, and production job records write `batch_started_at`; `_READY.json` keeps `created_at` as manifest completion time.
- `make-ready` creates the timestamp automatically. Re-running it on a valid committed input reuses the existing timestamp. Retrain, inference, retry, and production orchestration read the committed timestamp; users never pass `--data-cutoff`.
- Legacy READY and delivery records with `data_cutoff` remain readable. When no new field exists, use the legacy value only as the historical batch ordering/reference timestamp. New records do not write `data_cutoff`.
- Batch ordering and output date/reference use `batch_started_at`. Serving feature cutoff remains computed from actual interactions by the existing serving context.
- Production staging copies event rows unchanged; the scheduled slot controls when a job starts, not the interaction rows available to training/inference.
- Relative windows end at the exact maximum interaction `create_date`. `time_train` applies independently to each of three train windows; validation and test use their own positive integer hour settings. Windows are adjacent, non-overlapping at nanosecond precision, and retain the current five names.
- Relative temporal evaluation rejects missing, filename-derived, or day-only duration timestamps. `window_mode="pinned"` remains unchanged.
- Existing trainers fit on train 1–3 plus validation for test evaluation; the test window is used for quality-gate evaluation only. A passing candidate bundle is promoted directly.

## Compatibility

Read older READY files, deliveries, bundles, and SQLite jobs that only contain `data_cutoff`. Preserve their SHA-256 and file verification behavior. Migrate production SQLite job history without discarding rows. Leave `serving_cutoff` as the interaction-derived value.

## Acceptance

- No CLI option requires a user-supplied cutoff; new metadata writes `batch_started_at` and preserves it across reruns.
- Older metadata remains readable and can be ordered against new batches.
- No event is excluded by the batch-start timestamp.
- Configurable hour windows match hand-calculated windows across date boundaries, validate positive integers, and are persisted/fingerprinted.
- Insufficient window events/positives and day-only event timestamps fail clearly.
- Existing training, test, gate, and direct promotion flow is preserved without a post-gate fit.
- Do not run training, inference, benchmarks, or model downloads during local verification.
