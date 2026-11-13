# Manual cutoff demo (Batch CLI)

Run these from the repository root after the raw batch has been staged and the
runtime has the project dependencies installed. `data` is the prepared input
batch ID; replace it if the staged batch has another ID.

Prepare once:

```bash
python -m recsys_vnpt.batch_cli prepare-data --config configs/batch.gxgp.project-data.once.json --batch-id data
```

Demo February 15–20, 2025 (Vietnam time):

```bash
python -m recsys_vnpt.batch_cli run-retrain --config configs/batch.gxgp.project-data.once.json --batch-id data --data-end-at "2025-02-15T23:59:59+07:00"
python -m recsys_vnpt.batch_cli run-inference --config configs/batch.gxgp.project-data.once.json --batch-id data --data-end-at "2025-02-15T23:59:59+07:00"
python -m recsys_vnpt.batch_cli run-inference --config configs/batch.gxgp.project-data.once.json --batch-id data --data-end-at "2025-02-16T23:59:59+07:00"
python -m recsys_vnpt.batch_cli run-retrain --config configs/batch.gxgp.project-data.once.json --batch-id data --data-end-at "2025-02-17T23:59:59+07:00"
python -m recsys_vnpt.batch_cli run-inference --config configs/batch.gxgp.project-data.once.json --batch-id data --data-end-at "2025-02-17T23:59:59+07:00"
python -m recsys_vnpt.batch_cli run-inference --config configs/batch.gxgp.project-data.once.json --batch-id data --data-end-at "2025-02-18T23:59:59+07:00"
python -m recsys_vnpt.batch_cli run-inference --config configs/batch.gxgp.project-data.once.json --batch-id data --data-end-at "2025-02-19T23:59:59+07:00"
python -m recsys_vnpt.batch_cli run-inference --config configs/batch.gxgp.project-data.once.json --batch-id data --data-end-at "2025-02-20T23:59:59+07:00"
```

Without `--data-end-at`, Retrain and Inference use the latest available
interaction time. Each invocation creates and prints a new `run_id`; to retry
the same failed/interrupted invocation, pass its `--run-id` again. A retry must
use the same input and cutoff. Outputs are published as a new generation, and
the current generation changes only after output validation succeeds.
