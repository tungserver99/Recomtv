# Configurable raw data root

## Intent

Let an operator choose the folder containing raw MyTV data instead of relying on the repository-relative `data/data` path. The same path must be usable for a one-off CLI run and editable from the production dashboard.

## Current behavior

`production/config.json` stores `source_root`, defaulting to `data/data`. The dashboard does not expose it. `production.cli` reads that value and stages immutable batch snapshots before running the existing pipeline. `recsys_vnpt.batch_cli` is a lower-level interface that consumes an already staged batch under `incoming_root/<batch-id>`; it does not read the raw source tree.

## Approaches

1. **Configure the source path at the production boundary (recommended).** Add a `--data-root` override and `--prepare-only` mode to `production.cli`; persist the dashboard's editable raw data root in the existing state database. Both paths feed the current input-snapshot builder, preserving fingerprinting, `_READY.json`, and prepared-dataset identity. Keep `batch_cli` scoped to committed snapshots.
2. **Make every low-level batch command read a raw directory directly.** This would duplicate or bypass snapshot staging and make input identity/retry semantics ambiguous.

Use approach 1. The dashboard value persists across scheduler runs. A CLI `--data-root` value overrides config for that single CLI run. Relative paths resolve from the production config's project root. The selected directory must contain the expected raw source layout, including `duration`, `tags`, `content_tags`, and `mytv_vmp_content`.

## Scope

- Rename the production config setting from `source_root` to `data_root` while accepting legacy `source_root` configs during transition.
- Add CLI `--data-root`; add `--prepare-only` so the CLI can create a prepared dataset without retraining or inference.
- Add a dashboard data-root setting saved in SQLite; use it for both Prepare Data and scheduled production staging.
- Validate the selected path exists and is separate from incoming, work, delivery, and model-registry roots.
- Do not change `recsys_vnpt.batch_cli` input semantics, model behavior, or prepared artifact storage.

## Acceptance

- CLI can run prepare-only from a supplied raw data root.
- Dashboard can save and use a raw data root for Prepare Data and normal production jobs.
- Existing production configs continue to load.
- No raw-source path is silently inferred when a value is absent.
