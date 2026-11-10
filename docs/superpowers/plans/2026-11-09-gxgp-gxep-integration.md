# GxGp and GxEp Integration Plan

> **For agentic workers:** Implement inline using this plan and the attached user brief. Keep all GPU training/inference and full benchmarks for the user's server.

**Goal:** Support configurable GxGp and GxEp through existing training, evaluation, live-test, batch retraining, and inference entrypoints while preserving legacy defaults.

**Architecture:** Extend the existing learned content scorer with an explicit rank scope and add a small composition layer for heuristic, GBDT, and FCN scores. Persist FCN checkpoint/contract artifacts alongside the existing immutable scorer bundle, and route one validated scoring configuration through offline evaluation and production. Keep old CLI/config behavior when the new fields are omitted.

**Tech Stack:** Python, pandas/NumPy, existing scikit-learn/XGBoost heads, existing PyTorch FCN checkpoint/FeatureContract, pytest.

**Spec:** User-provided `CODEX TASK — Integrate GxGp and GxEp into RecSys-VNPT` attachment.

## Global Constraints

- Do not rebuild the pipeline or remove legacy methods, commands, output schemas, gates, or rollback behavior.
- Do not train, run model inference, benchmark, download weights/data, or execute GPU tests on this workstation.
- Never put this workstation's Python environment path in runtime code, configuration, or server commands.
- Preserve the current legacy defaults: scikit-learn GBDT, existing hand-sum/per-rule RRF behavior, and current CLI arguments.
- Do not claim benchmark parity without the missing `.local/experiments/` reference and server validation.

## Review Focus

- Same content in multiple rules must not get an artificial rank advantage under profile-level fusion; test unique-content ranking and broadcast.
- Out-of-scope content must preserve the existing fallback and must not mix rank scales within the selected scope; test complete-profile fallback.
- Ties and shuffled rows must not change profile-level ranks; test reordered synthetic frames.
- A GxEp bundle missing or mismatching either FCN artifact must be rejected; test hash/contract validation without loading or predicting a real model.
- A bundle trained for one active combination must not be silently served under another; test combination validation and legacy schema loading.

---

### Task 1: Profile-level RRF and content composition

**Files:**
- Modify: `recsys_vnpt/content_scorer.py`
- Create: `recsys_vnpt/scoring_config.py`
- Test: `tests/test_content_scorer.py`, `tests/test_scoring_config.py`

**Interfaces:**
- `rank_within_rule(frame, column, constant=60.0, scope="per_rule") -> np.ndarray`; `profile` scope ranks unique `(profile_id, content_id)` candidates deterministically and maps the rank back to rows.
- `ScoringConfig` resolves legacy/GxGp/GxEp/custom components, scope, and weights without changing omitted-field defaults.

- [ ] Test old per-rule literal ranks, profile-level ranks, duplicate content across rules, deterministic ties under row shuffles, and validation of unsupported configurations.
- [ ] Run focused CPU-only tests and confirm the new cases fail before implementation.
- [ ] Implement profile ranking and configuration resolution; keep the current per-rule branch intact.
- [ ] Run focused CPU-only tests; expected: all assertions pass and no model fitting or inference occurs.

### Task 2: FCN artifact and bundle lifecycle

**Files:**
- Create: `recsys_vnpt/fcn_scorer.py`
- Modify: `recsys_vnpt/scorer_bundle.py`
- Test: `tests/test_scorer_bundle.py`, `tests/test_fcn_scorer.py`

**Interfaces:**
- `load_fcn_scorer(checkpoint, contract, device, model_family="fcn_official")` reuses `load_v2_checkpoint` and rejects family/contract/hash mismatches.
- Bundle save accepts optional FCN checkpoint/contract paths and scoring metadata; load verifies each artifact while preserving its existing three-value return and schema-1 bundles.

- [ ] Test artifact copy/hash checks, missing/corrupt artifact refusal, legacy bundles, and a fake inference boundary without invoking PyTorch forward.
- [ ] Run focused CPU-only tests and confirm missing behavior fails before implementation.
- [ ] Add validated FCN artifact copying/loading and explicit CUDA strictness while preserving CPU and auto defaults.
- [ ] Run focused CPU-only tests; expected: artifact lifecycle passes without loading weights or performing inference.

### Task 3: Evaluation, live-test, and production routing

**Files:**
- Modify: `recsys_vnpt/evaluation_runs.py`, `recsys_vnpt/serving.py`, `recsys_vnpt/batch_contracts.py`, `recsys_vnpt/batch_runner.py`, `recsys_vnpt/pipeline_cli.py`, `recsys_vnpt/domain_entrypoint.py`
- Test: `tests/test_cli_config_parity.py`, `tests/test_batch_contracts.py`, `tests/test_batch_runner.py`, `tests/test_serving_evaluation_v3.py`

**Interfaces:**
- Optional scoring fields select per-rule/profile scope, GBDT backends, FCN checkpoint reuse/refresh, and component weights; omitted fields resolve to legacy behavior.
- Evaluation and serving call the same content composition/scoring helper. FCN features use the existing as-of pair-feature builder and `FeatureContract` preparation in bounded batches.
- Retraining can reuse a configured FCN checkpoint or explicitly refresh it through the existing `train` pipeline; inference only loads the promoted bundle.

- [ ] Add config/parser compatibility tests first, including old config defaults, GxGp/GxEp preset resolution, strict CUDA configuration, FCN checkpoint-required errors, and command argument forwarding.
- [ ] Run focused tests and confirm new options fail before implementation.
- [ ] Wire evaluation, quality-gate scoring, retraining, live-test, production context, bundle identity, and shard fingerprint to the same active combination.
- [ ] Run focused CPU-only tests with mocks/synthetic rows; expected: GxGp never loads FCN, GxEp rejects absent artifacts, legacy defaults remain identical, and cache identity changes with the bundle/combination.

### Task 4: Staging configs and GPU-server runbook

**Files:**
- Create: `configs/batch.gxgp.staging.example.json`, `configs/batch.gxep.staging.example.json`
- Modify: `README.md`, `docs/KAGGLE_TESTING.md` or a new `docs/GXGP_GXEP_RUNBOOK.md`
- Test: `tests/test_batch_contracts.py`, `tests/test_cli_config_parity.py`

- [ ] Load both example configs through the production `load_config` parser and exercise the existing CLI help/parser with their option sets.
- [ ] Document preparation, retraining, persisted-bundle evaluation, inference, hardware qualification, expected metrics, isolation of staging roots, and the unverified benchmark assumptions.
- [ ] Run syntax checks and the selected lightweight tests only; report all GPU/server commands for the user to run.
