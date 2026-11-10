"""Persisted learned scorers and the pointer to the promoted one.

Inference never fits a scorer: retraining writes an immutable bundle, a gate
decides whether it is promoted, and inference loads whatever the pointer names.
A bundle is unpickled only after its SHA-256 and library versions check out,
and only from the deployment's own model store.
"""

from __future__ import annotations

import json
import hashlib
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
    rule_model: RuleScorerModel | None,
    content_model: ContentScorerModel | None,
    metadata: dict[str, Any],
    *,
    fcn_checkpoint_path: str | Path | None = None,
    fcn_contract_path: str | Path | None = None,
) -> Path:
    """Write a new immutable bundle under ``models_root`` and return its path."""
    root = Path(models_root)
    root.mkdir(parents=True, exist_ok=True)
    if (fcn_checkpoint_path is None) != (fcn_contract_path is None):
        raise BundleError("both FCN checkpoint and contract are required")
    if fcn_checkpoint_path is not None:
        for label, source in (
            ("checkpoint", Path(fcn_checkpoint_path)),
            ("contract", Path(fcn_contract_path)),
        ):
            if not source.is_file():
                raise BundleError(f"FCN {label} is missing: {source}")
    created = datetime.now(timezone.utc)
    stage = root / f".{uuid.uuid4().hex}.stage"
    stage.mkdir()
    try:
        joblib.dump({"rule": rule_model, "content": content_model}, stage / MODEL_FILE)
        digest = sha256_file(stage / MODEL_FILE)
        manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "created_at": created.isoformat(),
            "model_file": MODEL_FILE,
            "model_sha256": digest,
            "libraries": _library_versions(),
            "rule_scorer": rule_model.metrics if rule_model is not None else None,
            "content_scorer": content_model.metrics if content_model is not None else None,
            **metadata,
        }
        if fcn_checkpoint_path is not None and fcn_contract_path is not None:
            fcn_dir = stage / "fcn"
            fcn_dir.mkdir()
            checkpoint_name = "fcn/best_model_v2.pth"
            contract_name = "fcn/feature_contract_v2.json"
            shutil.copyfile(fcn_checkpoint_path, stage / checkpoint_name)
            shutil.copyfile(fcn_contract_path, stage / contract_name)
            manifest["fcn_artifacts"] = {
                "checkpoint_file": checkpoint_name,
                "checkpoint_sha256": sha256_file(stage / checkpoint_name),
                "contract_file": contract_name,
                "contract_sha256": sha256_file(stage / contract_name),
            }
            bundle_digest = hashlib.sha256(
                "\n".join(
                    (
                        digest,
                        manifest["fcn_artifacts"]["checkpoint_sha256"],
                        manifest["fcn_artifacts"]["contract_sha256"],
                    )
                ).encode("ascii")
            ).hexdigest()
            manifest["bundle_sha256"] = bundle_digest
            try:
                import torch
            except ImportError as error:
                raise BundleError("PyTorch is required for an FCN scorer bundle") from error
            manifest["libraries"]["torch"] = torch.__version__
        else:
            # Preserve the schema-1 legacy identity for scorer-only bundles.
            manifest["bundle_sha256"] = digest
        bundle_id = f"{created:%Y%m%dT%H%M%SZ}_{manifest['bundle_sha256'][:12]}"
        manifest["bundle_id"] = bundle_id
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
) -> tuple[RuleScorerModel | None, ContentScorerModel | None, dict[str, Any]]:
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
        (manifest.get("rule_scorer") or {}).get("rule_scorer_backend"),
        (manifest.get("content_scorer") or {}).get("content_scorer_backend"),
    )
    if uses_xgboost and manifest["libraries"].get("xgboost") != _xgboost_version():
        raise BundleError(
            f"bundle was trained with xgboost {manifest['libraries'].get('xgboost')}, this environment "
            f"has {_xgboost_version()}; retrain under the serving environment"
        )
    fcn_artifacts = manifest.get("fcn_artifacts")
    if fcn_artifacts is not None:
        try:
            import torch
        except ImportError as error:
            raise BundleError("this FCN bundle requires PyTorch") from error
        if manifest["libraries"].get("torch") != torch.__version__:
            raise BundleError(
                "bundle was trained with PyTorch "
                f"{manifest['libraries'].get('torch')}, this environment has "
                f"{torch.__version__}; install a compatible version"
            )
        for key, hash_key in (
            ("checkpoint_file", "checkpoint_sha256"),
            ("contract_file", "contract_sha256"),
        ):
            relative = Path(str(fcn_artifacts.get(key, "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise BundleError(f"FCN artifact path is unsafe: {relative}")
            artifact = Path(bundle) / relative
            if (
                not artifact.is_file()
                or sha256_file(artifact) != fcn_artifacts.get(hash_key)
            ):
                raise BundleError(f"FCN artifact is missing or corrupt: {artifact}")
        expected_bundle_digest = hashlib.sha256(
            "\n".join(
                (
                    manifest["model_sha256"],
                    fcn_artifacts["checkpoint_sha256"],
                    fcn_artifacts["contract_sha256"],
                )
            ).encode("ascii")
        ).hexdigest()
        if manifest.get("bundle_sha256") != expected_bundle_digest:
            raise BundleError("bundle identity does not match its FCN artifacts")
    elif manifest.get("bundle_sha256", manifest["model_sha256"]) != manifest["model_sha256"]:
        raise BundleError("scorer bundle identity does not match its model payload")
    payload = joblib.load(model_path)
    return payload.get("rule"), payload.get("content"), manifest


def bundle_fcn_artifact_paths(
    bundle: str | Path, manifest: dict[str, Any] | None = None
) -> tuple[Path, Path] | None:
    """Return verified FCN paths from a loaded bundle, or ``None`` for legacy."""
    active_manifest = manifest or read_bundle_manifest(bundle)
    artifacts = active_manifest.get("fcn_artifacts")
    if artifacts is None:
        return None
    return (
        Path(bundle) / artifacts["checkpoint_file"],
        Path(bundle) / artifacts["contract_file"],
    )


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
