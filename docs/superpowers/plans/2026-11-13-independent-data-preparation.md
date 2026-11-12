# Independent Data Preparation Plan

## Goal

Make raw-data preparation an explicit, resumable `prepare-data` batch job. Retrain, inference, and hardware qualification must resolve and validate the exact prepared dataset for the requested input batch and current preparation configuration; they must never call canonical or temporal preparation as a fallback.

## Design calls

- Prepared artifacts are cached by full input fingerprint plus preparation-configuration hash. A per-batch manifest pins the exact cache entry; consumers never pick a latest version.
- The preparation configuration includes domain, behavior setting, temporal window sizes, reference date, and the preparation contract version. Operational settings such as worker count and inference sample size are excluded.
- Existing canonical and temporal cache mechanisms remain responsible for retry/resume. A changed input fingerprint gets a distinct cache directory. Since duration source files overlap in time and are not date-partitioned inputs, do not infer changed date partitions from filenames.
- Keep the scheduler single-job lock. Dashboard Prepare Data queues a prepare-only run for the next configured slot; the later production run reuses that slot's staged input and prepared manifest.
- Keep production model training, quality gate, promotion, and publishing semantics unchanged.

## Tasks

- [x] Add prepared-dataset manifest/config identity and exact validation helpers, with focused tests for config mismatch and corrupt/missing artifacts.
- [x] Add `prepare-data` CLI/job state, timings, failure reports, deterministic cache reuse, and retry behavior; test CLI dispatch and cache hits.
- [x] Replace implicit preparation in retrain, inference, and qualification with strict prepared-dataset loading; add an actionable `prepare-data` instruction.
- [x] Read playback date partitions selectively for each temporal label window while retaining full playback history for sequence features; interaction date partitions remain the source for temporal-history features.
- [x] Add a prepare-only scheduler request and dashboard status/action; prevent overlapping dashboard actions through the shared request queue and preparation lock.
- [x] Run focused project tests and lightweight compile/diff checks only; no training, inference, large benchmark, or weight download was run.

## Review focus

- A consumer cannot silently select another batch's prepared artifacts.
- A preparation config change invalidates the pinned prepared dataset.
- Retry after interruption uses deterministic paths and existing canonical/temporal caches.
- The dashboard's prepare-only action does not start training or inference.
- Date-partition optimization preserves labels and history at window boundaries.

## Verification record

- Focused unit tests: 59 passed across prepared dataset, production state/orchestration/scheduler/dashboard, and temporal sampling. Coverage includes changed raw inputs, cache reuse, retry after an interrupted prepare, and prepare-then-production history.
- One pre-existing Parquet-write test was excluded on the rerun because the `topmost` environment has neither `pyarrow` nor `fastparquet`; the first full focused run confirmed that exact environment limitation (57 passed, 1 failed at `DataFrame.to_parquet`).
- `python -m compileall -q recsys_vnpt production` completed without errors; `git diff --check` reported no whitespace errors.
