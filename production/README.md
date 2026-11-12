# MyTV Recommendation System · Production Wrapper

This folder adds a single-host scheduler and operator dashboard around the existing `recsys_vnpt` batch pipeline. It keeps the current training, quality gate, model promotion, inference, validation, and three-file publishing implementation.

## Server setup

Install all project dependencies from the single root requirements file:

```bash
python -m pip install -r requirements.txt
```

Set `data_root` in `production/config.json` to the raw folder containing `duration/`, `mytv_vmp_content/`, `tags/`, `content_tags/`, and `month_mytv_info.parquet`. Relative paths resolve from the repository root. The dashboard's Configuration page can save a different raw data folder; that choice is stored in SQLite and used by scheduled jobs. Raw source data is read only. Staged inputs, work files, model registry, state, and delivery remain separate.

## Run in test mode first

`config.test.json` uses `max_profiles=1000`, `gate_evaluation_profiles=500`, one inference shard, and separate `data/production-test/` model/delivery/work/state roots. This keeps sample outputs, SQLite history, and the test model registry apart from production.

```bash
python -m production.cli run --config production/config.test.json
```

Once the GPU server run and its sample outputs are reviewed, use `production/config.json`; it defaults to `max_profiles=0` (all profiles), `gate_evaluation_profiles=5000`, 16 shards, and the production directories.

## Start services

Terminal 1: keep the scheduler process running. Closing the browser does not stop scheduled work.

```bash
python -m production.scheduler
```

Terminal 2: open the dashboard on the server or trusted internal network.

```bash
streamlit run production/dashboard/app.py --server.address 127.0.0.1
```

Optional manual production run:

```bash
python -m production.cli run
```

Prepare a dataset without retraining or inference:

```bash
python -m production.cli run --prepare-only --data-root /path/to/raw-data
```

`--data-root` is optional; when omitted, the CLI uses the dashboard's saved folder or the config's `data_root`. An explicit CLI value applies only to that run. The folder is staged into an immutable input batch before preparation.

The lower-level `python -m recsys_vnpt.batch_cli prepare-data --config ... --batch-id ...` command continues to read an already staged, committed batch from `incoming_root`.

Or run a specific configured slot:

```bash
python -m production.cli run --slot 2026-11-13T12:00:00+07:00
```

The dashboard Run now button only enqueues a SQLite request; the scheduler process executes it. The default schedule is daily at 00:00 and 12:00 in `Asia/Ho_Chi_Minh`. Schedule edits take effect through scheduler polling.

## Data and temporal limitations

- `batch_started_at` is the immutable UTC-aware time captured when the batch starts. It controls batch ordering, metadata reference year, and output filename date; it never filters training or inference interactions. Existing interaction-derived `serving_cutoff` remains the serving history cutoff.
- Temporal retraining requires precise duration `create_date` values and derives adjacent, gap-free windows from the exact maximum event timestamp. It rejects date-only values and does not fall back to `filename_date`. Set `time_train`, `time_valid`, and `time_test` (hours) under `batch` in `config.json`; defaults are 24 each. Three training windows each use `time_train`, followed by validation and test.
- When `include_behavior` is enabled, behavior CSVs are also cutoff-filtered using `create_date` or the existing `event_time` alias. Files without a precise timestamp fail the batch.
- Other catalog, tags, and user metadata are copied as the current source snapshot and are not proven point-in-time snapshots.
- Window durations are hours and must be positive integers. Each generated window must contain at least one positive sample.
- Demo mode uses static simulated pass and fallback runs. It does not open the production SQLite store or write delivery/model state.
- Run the dashboard only on localhost or a trusted internal network. It has no authentication layer.

## Runtime artifacts and recovery

SQLite state is stored at `data/production/state/production.db`; wrapper reports and the end-to-end orchestration log are stored under `data/production/work/batches/<batch_id>/production/`. Detailed `status.json` and `log.jsonl` files remain under each retrain/inference directory, including preparation progress by source chunk and temporal window.

Only one scheduler process should run for this repository's production directories. Jobs are serialized with a persistent filesystem `BatchLock`. After an abrupt process death, the state becomes `INTERRUPTED` and the lock is deliberately left for an operator to inspect; do not remove it until confirming the process is gone. Missed scheduled slots are not replayed automatically. A published slot is not rebuilt or published a second time.

The dashboard's Live mode reads actual SQLite, batch status/logs, model registry, and delivery-related reports. Demo mode is labeled as simulated and does not modify production state.
