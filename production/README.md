# MyTV Recommendation System · Production Wrapper

This folder adds a single-host scheduler and operator dashboard around the existing `recsys_vnpt` batch pipeline. It keeps the current training, quality gate, model promotion, inference, validation, and three-file publishing implementation.

## Server setup

Install all project dependencies from the single root requirements file:

```bash
python -m pip install -r requirements.txt
```

Set `data_root` in `production/config.json` to the raw folder containing `duration/`, `mytv_vmp_content/`, `tags/`, `content_tags/`, and `month_mytv_info.parquet`. Relative paths resolve from the repository root. The dashboard's Configuration page can save a different raw data folder; that choice is stored in SQLite and used by scheduled jobs. Raw source data is read only. Staged inputs, work files, model registry, state, and delivery remain separate.

## Run each operation independently

`config.test.json` uses `max_profiles=1000`, `gate_evaluation_profiles=500`, one inference shard, and separate `data/production-test/` model/delivery/work/state roots. This keeps sample outputs, SQLite history, and the test model registry apart from production.

Prepare once when source data changes; retrain and inference consume the prepared batch independently. For the historical cutoff demo and complete Batch CLI sequence, see [manual-cutoff-demo.md](../docs/production/manual-cutoff-demo.md).

## Start services

Terminal 1: keep the scheduler process running. Closing the browser does not stop scheduled work.

```bash
python -m production.scheduler
```

Terminal 2: open the dashboard on the server or trusted internal network.

```bash
streamlit run production/dashboard/app.py --server.address 127.0.0.1
```

Stage raw data and prepare it:

```bash
python -m production.cli prepare-data --config production/config.test.json
```

Run only retraining or only inference against an already prepared batch:

```bash
python -m production.cli run-retrain --config production/config.test.json --batch-id <batch-id>
python -m production.cli run-inference --config production/config.test.json --batch-id <batch-id>
```

Pass `--data-root /path/to/raw-data` to `prepare-data` to override the dashboard/configured folder. Retrain and inference accept optional `--data-end-at` and `--run-id`; inference also accepts `--model-path /path/to/model-bundle` to use a verified bundle for that run. Without it, inference uses the promoted model as before. An explicit bundle does not change the promotion pointer, and the model cutoff is still checked against `--data-end-at`. A retry reuses its original run ID and cutoff. Dashboard requests are independent, and scheduler times are configured separately for retrain and inference. Prepare Data remains manual.

## Data and temporal limitations

- `batch_started_at` is the immutable UTC-aware time captured when the batch starts. It controls batch ordering, metadata reference year, and output filename date; it never filters training or inference interactions. `data_end_at` is optional and only selects historical interactions. `serving_cutoff` reports the actual latest interaction used by inference.
- Temporal retraining requires precise duration `create_date` values and derives adjacent, gap-free windows from the exact maximum event timestamp (or selected `data_end_at`). It rejects date-only values and does not fall back to `filename_date`. Set `time_train`, `time_valid`, and `time_test` (hours) under `batch` in `config.json`; defaults are 24 each. Three training windows each use `time_train`, followed by validation and test.
- When `include_behavior` is enabled, behavior CSVs are also cutoff-filtered using `create_date` or the existing `event_time` alias. Files without a precise timestamp fail the batch.
- Other catalog, tags, and user metadata are copied as the current source snapshot and are not proven point-in-time snapshots.
- Window durations are hours and must be positive integers. Each generated window must contain at least one positive sample.
- Demo mode uses static simulated pass and fallback runs. It does not open the production SQLite store or write delivery/model state.
- Run the dashboard only on localhost or a trusted internal network. It has no authentication layer.

## Runtime artifacts and recovery

SQLite state is stored at `data/production/state/production.db`; wrapper reports and the end-to-end orchestration log are stored under `data/production/work/batches/<batch_id>/production/`. Detailed `status.json` and `log.jsonl` files remain under each retrain/inference directory, including preparation progress by source chunk and temporal window.

Only one scheduler process should run for this repository's production directories. Jobs are serialized with a persistent filesystem `BatchLock`. After an abrupt process death, the state becomes `INTERRUPTED` and the lock is deliberately left for an operator to inspect; do not remove it until confirming the process is gone. Missed scheduled slots are not replayed automatically. A completed inference is idempotent when retried with the same run ID; a new run ID can publish the same prepared batch again.

The dashboard's Live mode reads actual SQLite, batch status/logs, model registry, and delivery-related reports. Demo mode is labeled as simulated and does not modify production state.
