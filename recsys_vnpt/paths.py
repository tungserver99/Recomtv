from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_root(project_root: str | Path | None = None) -> Path:
    root = Path(project_root) if project_root is not None else globals()["project_root"]()
    env_root = os.environ.get("RECSYS_VNPT_DATA_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    return root / "data" / "data"


def data_artifact_path(*parts: str, project_root: str | Path | None = None) -> Path:
    return data_root(project_root).joinpath(*parts)


def source_root(project_root: str | Path | None = None) -> Path:
    """Where raw inputs are read from; defaults to the data root.

    Production mounts each input batch read-only and writes generated artifacts
    elsewhere, so the two roots are configured independently.
    """
    env_root = os.environ.get("RECSYS_VNPT_SOURCE_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    return data_root(project_root)


def source_artifact_path(*parts: str, project_root: str | Path | None = None) -> Path:
    return source_root(project_root).joinpath(*parts)


def project_artifact_path(*parts: str, project_root: str | Path | None = None) -> Path:
    root = Path(project_root) if project_root is not None else globals()["project_root"]()
    return root.joinpath(*parts)


def domain_work_path(domain: str, *parts: str, project_root: str | Path | None = None) -> Path:
    root = Path(project_root) if project_root is not None else globals()["project_root"]()
    external = data_root(root).joinpath(domain, *parts)
    if external.exists():
        return external
    return root.joinpath(domain, *parts)


def model_artifact_path(
    domain: str,
    *parts: str,
    project_root: str | Path | None = None,
) -> Path:
    return data_artifact_path("model", domain, *parts, project_root=project_root)


def model_dir(domain: str, project_root: str | Path | None = None) -> Path:
    return model_artifact_path(domain, project_root=project_root)


def rule_info_path(project_root: str | Path | None = None) -> Path:
    root = Path(project_root) if project_root is not None else globals()["project_root"]()
    external = source_artifact_path("rule_info.parquet", project_root=root)
    if external.exists():
        return external
    return root / "rule_info.parquet"
