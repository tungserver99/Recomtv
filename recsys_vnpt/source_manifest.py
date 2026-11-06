from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SourceArtifact:
    path: str
    role: str
    bytes: int
    sha256: str
    schema_hash: str | None = None
    rows: int | None = None
    modified_ns: int = 0
    hash_mode: str = "content_sha256"


def sha256_file(path: str | Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def schema_hash(columns: Iterable[str]) -> str:
    return stable_hash(list(columns))


def build_source_artifact(
    path: str | Path,
    role: str,
    columns: Iterable[str] | None = None,
    rows: int | None = None,
    root: str | Path | None = None,
    full_hash: bool = True,
) -> SourceArtifact:
    source = Path(path)
    display = source
    if root is not None:
        try:
            display = source.resolve().relative_to(Path(root).resolve())
        except ValueError:
            pass
    stat = source.stat()
    if full_hash:
        digest = sha256_file(source)
        hash_mode = "content_sha256"
    else:
        digest = stable_hash(
            {
                "path": display.as_posix(),
                "bytes": stat.st_size,
                "modified_ns": stat.st_mtime_ns,
            }
        )
        hash_mode = "path_size_mtime_sha256"
    return SourceArtifact(
        path=display.as_posix(),
        role=role,
        bytes=stat.st_size,
        sha256=digest,
        schema_hash=schema_hash(columns) if columns is not None else None,
        rows=rows,
        modified_ns=stat.st_mtime_ns,
        hash_mode=hash_mode,
    )


def build_source_artifacts(
    paths: Iterable[str | Path],
    role: str,
    root: str | Path | None = None,
    full_hash: bool = False,
) -> list[SourceArtifact]:
    return [
        build_source_artifact(path, role, root=root, full_hash=full_hash)
        for path in sorted(map(Path, paths))
    ]


def manifest_fingerprint(artifacts: Iterable[SourceArtifact]) -> str:
    payload = [asdict(item) for item in sorted(artifacts, key=lambda item: item.path)]
    return stable_hash(payload)


def write_source_manifest(
    path: str | Path,
    artifacts: Iterable[SourceArtifact],
    skipped: Iterable[dict[str, str]] = (),
) -> Path:
    output = Path(path)
    items = sorted(artifacts, key=lambda item: item.path)
    payload = {
        "schema_version": 1,
        "fingerprint": manifest_fingerprint(items),
        "artifacts": [asdict(item) for item in items],
        "skipped": list(skipped),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def read_source_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
