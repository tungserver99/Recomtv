"""Persisted learned scorers and the pointer to the promoted one.

Inference never fits a scorer: retraining writes an immutable bundle, a gate
decides whether it is promoted, and inference loads whatever the pointer names.
A bundle is unpickled only after its SHA-256 and library versions check out,
and only from the deployment's own model store.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from recsys_vnpt.content_scorer import ContentScorerModel
from recsys_vnpt.rule_scorer import RuleScorerModel
from recsys_vnpt.runtime import atomic_write_text
from recsys_vnpt.source_manifest import sha256_file


BUNDLE_SCHEMA_VERSION = 1
MODEL_FILE = "scorers.joblib"
MANIFEST_FILE = "bundle.json"
PROMOTED = "promoted_model.json"
PREVIOUS = "previous_model.json"


class BundleError(RuntimeError):
    """A bundle is missing, corrupt, or incompatible with this environment."""


def _xgboost_version() -> str | None:
    try:
        import xgboost
    except ImportError:
        return None
    return xgboost.__version__


def _library_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    if _xgboost_version() is not None:
        versions["xgboost"] = _xgboost_version()
    return versions


def save_bundle(
    models_root: str | Path,
    rule_model: RuleScorerModel,
    content_model: ContentScorerModel,
    metadata: dict[str, Any],
) -> Path:
    """Write a new immutable bundle under ``models_root`` and return its path."""
    root = Path(models_root)
    root.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc)
    stage = root / f".{uuid.uuid4().hex}.stage"
    stage.mkdir()
    try:
        joblib.dump({"rule": rule_model, "content": content_model}, stage / MODEL_FILE)
        digest = sha256_file(stage / MODEL_FILE)
        bundle_id = f"{created:%Y%m%dT%H%M%SZ}_{digest[:12]}"
        manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "bundle_id": bundle_id,
            "created_at": created.isoformat(),
            "model_file": MODEL_FILE,
            "model_sha256": digest,
            "libraries": _library_versions(),
            "rule_scorer": rule_model.metrics,
            "content_scorer": content_model.metrics,
            **metadata,
        }
        (stage / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        target = root / bundle_id
        os.replace(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return target


def read_bundle_manifest(bundle: str | Path) -> dict[str, Any]:
    path = Path(bundle) / MANIFEST_FILE
    if not path.is_file():
        raise BundleError(f"bundle manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise BundleError(f"unsupported bundle schema: {manifest.get('schema_version')}")
    return manifest


def load_bundle(
    bundle: str | Path,
) -> tuple[RuleScorerModel, ContentScorerModel, dict[str, Any]]:
    """Verify, then load. A pickle from another scikit-learn release is refused
    rather than trusted: tree internals are not stable across versions."""
    manifest = read_bundle_manifest(bundle)
    model_path = Path(bundle) / manifest["model_file"]
    if not model_path.is_file() or sha256_file(model_path) != manifest["model_sha256"]:
        raise BundleError(f"bundle payload is missing or corrupt: {model_path}")
    trained_with = manifest["libraries"].get("scikit-learn")
    if trained_with != sklearn.__version__:
        raise BundleError(
            f"bundle was trained with scikit-learn {trained_with}, this environment "
            f"has {sklearn.__version__}; retrain under the serving environment"
        )
    uses_xgboost = "xgboost" in (
        manifest.get("rule_scorer", {}).get("rule_scorer_backend"),
        manifest.get("content_scorer", {}).get("content_scorer_backend"),
    )
    if uses_xgboost and manifest["libraries"].get("xgboost") != _xgboost_version():
        raise BundleError(
            f"bundle was trained with xgboost {manifest['libraries'].get('xgboost')}, this environment "
            f"has {_xgboost_version()}; retrain under the serving environment"
        )
    payload = joblib.load(model_path)
    return payload["rule"], payload["content"], manifest


def promoted_bundle(registry: str | Path) -> Path | None:
    """The bundle the pointer names, or None when nothing was promoted yet."""
    pointer = Path(registry) / PROMOTED
    if not pointer.is_file():
        return None
    return Path(json.loads(pointer.read_text(encoding="utf-8"))["bundle_path"])


def promote(registry: str | Path, bundle: str | Path, gate: dict[str, Any]) -> None:
    """Point production at ``bundle``; the previous pointer is kept for rollback."""
    registry = Path(registry)
    manifest = read_bundle_manifest(bundle)
    current = registry / PROMOTED
    if current.is_file():
        atomic_write_text(registry / PREVIOUS, current.read_text(encoding="utf-8"))
    atomic_write_text(
        current,
        json.dumps(
            {
                "bundle_id": manifest["bundle_id"],
                "bundle_path": str(Path(bundle).resolve()),
                "model_sha256": manifest["model_sha256"],
                "promoted_at": datetime.now(timezone.utc).isoformat(),
                "gate": gate,
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
    )
