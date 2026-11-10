# GxGp / GxEp staging runbook

These examples use the shared scorer composition and existing batch lifecycle. Paths in the JSON files are staging placeholders; change them to the server's staging directories before use. GxGp and GxEp have separate work, delivery, and model registry paths so their bundles and outputs do not overwrite one another. They contain no local Python environment paths.

## Before running

1. Install the project dependencies in the server environment, including a CUDA build of PyTorch for GxEp and XGBoost for both presets.
2. Prepare the normal temporal artifacts using the repository's current workflow. The first `run-retrain` can start with no promoted scorer bundle; later retrains compare against the currently promoted bundle.
3. Copy the corresponding example JSON to a server-owned staging config and check free disk space. GxEp's `fcn_refresh` trains `fcn_official` during `run-retrain`; it requires CUDA and is intended for the server.
4. Stage the batch inputs under `<incoming_root>/<batch-id>/`, then write `_READY.json` last. The ready manifest must include every input file and a timezone-aware cutoff.

## GxGp

```bash
python -m recsys_vnpt.batch_cli run-retrain --config configs/batch.gxgp.staging.example.json --batch-id 2026-11-09
python -m recsys_vnpt.batch_cli preflight --config configs/batch.gxgp.staging.example.json --batch-id 2026-11-09
python -m recsys_vnpt.batch_cli run-inference --config configs/batch.gxgp.staging.example.json --batch-id 2026-11-09
python -m recsys_vnpt.batch_cli qualify-hardware --config configs/batch.gxgp.staging.example.json --batch-id 2026-11-09
```

## GxEp

```bash
python -m recsys_vnpt.batch_cli run-retrain --config configs/batch.gxep.staging.example.json --batch-id 2026-11-09
python -m recsys_vnpt.batch_cli preflight --config configs/batch.gxep.staging.example.json --batch-id 2026-11-09
python -m recsys_vnpt.batch_cli run-inference --config configs/batch.gxep.staging.example.json --batch-id 2026-11-09
python -m recsys_vnpt.batch_cli qualify-hardware --config configs/batch.gxep.staging.example.json --batch-id 2026-11-09
```

For serving an already staged bundle without retraining, set `fcn_refresh` to `false` and provide the matching `fcn_checkpoint_path` and `fcn_contract_path` in the GxEp config. `preflight` checks the currently promoted bundle; run it after the first successful retrain, then before later inference batches. `run-retrain` evaluates and gates the candidate before promotion. Inspect `rule_ndcg@20` and `macro_content_ndcg@10` in the gate report; both must meet the hand-built baseline and remain within the configured regression limit of the previously promoted bundle. Keep the GxGp and GxEp model registries separate while comparing them.

## Focused evaluation commands

With temporal data already prepared, these commands exercise the same content composition in the evaluation CLI. GxEp requires a compatible FCN checkpoint and feature contract.

```bash
python -m recsys_vnpt.pipeline_cli evaluate-baseline --root /srv/recsys/staging/project --domain all --scoring-preset GxGp --rule-gbdt-backend xgboost --content-gbdt-backend xgboost
python -m recsys_vnpt.pipeline_cli evaluate-baseline --root /srv/recsys/staging/project --domain all --scoring-preset GxEp --rule-gbdt-backend xgboost --content-gbdt-backend xgboost --content-fcn-checkpoint /srv/recsys/staging/fcn/best_model_v2.pth --content-fcn-contract /srv/recsys/staging/fcn/feature_contract_v2.json --content-fcn-device cuda
```

These are server commands only. Local validation covered configuration, fusion arithmetic, mocked FCN boundaries, and artifact integrity; no training, GPU work, or large inference was run here. The default GxGp/GxEp fusion weights are configurable starting values and still need confirmation against the original benchmark artifacts before interpreting a comparison as a reproduced result.
