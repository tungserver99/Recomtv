from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from recsys_vnpt.rule_assembly import RuleAssemblyConfig
from recsys_vnpt.runtime import atomic_write_text
from recsys_vnpt.source_manifest import sha256_file


def rank_result(data: dict[str, Any], n: int) -> dict[str, Any]:
    reordered_data: dict[str, Any] = {}
    for user_id, user_data in data.items():
        user_scores = []
        for content_id, content in user_data["suggested_content"].items():
            user_scores.append(
                {
                    "content_id": content_id,
                    "content_name": content.get("content_name", ""),
                    "tag_names": content.get("tag_names", ""),
                    "type_id": content.get("type_id", ""),
                    "score": content.get("score", 0.0),
                    "rule_id": content.get("rule_id"),
                }
            )
        sorted_scores = sorted(user_scores, key=lambda row: row["score"], reverse=True)[:n]
        reordered_data[user_id] = {
            "suggested_content": {
                row["content_id"]: {
                    key: value
                    for key, value in row.items()
                    if key != "content_id" and value is not None
                }
                for row in sorted_scores
            },
            "user": user_data["user"],
        }
    return reordered_data


def build_client_exports(result_with_rule: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    """Build rulename.json records and rule_content.txt text from scored results."""

    homepage_rule: list[dict[str, str]] = []
    rule_content_lines: list[str] = []

    for pid, info in result_with_rule.items():
        rule_to_content: OrderedDict[str, OrderedDict[str, str]] = OrderedDict()
        for content_id, content in info["suggested_content"].items():
            rule_id = content.get("rule_id")
            if str(rule_id) == "-100":
                continue
            rule_key = str(rule_id)
            rule_to_content.setdefault(rule_key, OrderedDict())[str(content_id)] = str(content.get("type_id", ""))

        homepage_rule.append({"pid": pid, "d": ",".join(rule_to_content)})

        line = f"{pid}|"
        chunks = []
        for rule_id, content_ids in rule_to_content.items():
            content_chunk = "".join(f"${type_id}#{content_id}" for content_id, type_id in content_ids.items())
            chunks.append(f"{rule_id},{content_chunk.lstrip('$')}")
        line += ";".join(chunks)
        rule_content_lines.append(line)

    return homepage_rule, "\n".join(rule_content_lines) + ("\n" if rule_content_lines else "")


def serialize_assembled_output(
    recommendations: dict[str, dict[str, Any]],
    config: RuleAssemblyConfig = RuleAssemblyConfig(),
) -> tuple[list[dict[str, str]], str]:
    homepage: list[dict[str, str]] = []
    lines: list[str] = []
    for profile_id in sorted(recommendations):
        payload = recommendations[profile_id]
        normal_rules = payload["normal_rules"]
        normal_ids = [str(row["rule_id"]) for row in normal_rules]
        special = [str(rule) for rule in payload.get("special_rules", config.special_rules)]
        homepage.append({"pid": str(profile_id), "d": ",".join(normal_ids + special)})
        chunks: list[str] = []
        for row in normal_rules:
            items = "$".join(
                f"{content['type_id']}#{content['content_id']}" for content in row["contents"]
            )
            chunks.append(f"{row['rule_id']},{items}")
        lines.append(f"{profile_id}|" + ";".join(chunks))
    return homepage, "\n".join(lines) + ("\n" if lines else "")


@dataclass(frozen=True)
class ClientOutputSnapshot:
    manifest: dict[str, Any]
    homepage: list[dict[str, str]]
    rule_content: str


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _generation_file_path(
    output: Path,
    manifest: dict[str, Any],
    filename: str,
) -> tuple[Path, str]:
    generation = str(manifest.get("generation", ""))
    entry = manifest.get("files", {}).get(filename)
    if not generation or not isinstance(entry, dict):
        raise ValueError("client output manifest is missing generation file metadata")
    expected_relative = Path("generations") / generation / filename
    relative = Path(str(entry.get("path", "")))
    if relative != expected_relative:
        raise ValueError(f"client output manifest has an invalid path for {filename}")
    digest = str(entry.get("sha256", ""))
    if len(digest) != 64:
        raise ValueError(f"client output manifest is missing a SHA-256 for {filename}")
    return output / relative, digest


def read_client_output_generation(
    output_dir: str | Path,
    manifest: dict[str, Any],
    homepage_name: str = "rulename.json",
    rule_content_name: str = "rule_content.txt",
) -> ClientOutputSnapshot:
    """Read and verify the immutable generation referenced by a manifest."""

    output = Path(output_dir)
    homepage_path, homepage_hash = _generation_file_path(output, manifest, homepage_name)
    content_path, content_hash = _generation_file_path(output, manifest, rule_content_name)
    homepage_text = homepage_path.read_text(encoding="utf-8")
    rule_content = content_path.read_text(encoding="utf-8")
    if _sha256(homepage_text) != homepage_hash or _sha256(rule_content) != content_hash:
        raise ValueError("client output generation hash mismatch")
    homepage = json.loads(homepage_text)
    if not isinstance(homepage, list):
        raise ValueError("client output homepage generation must be a JSON list")
    return ClientOutputSnapshot(
        manifest=manifest,
        homepage=homepage,
        rule_content=rule_content,
    )


def read_committed_client_output(
    output_dir: str | Path,
    manifest_name: str = "client_output_manifest.json",
) -> ClientOutputSnapshot:
    """Read the manifest-committed snapshot, retrying if it changes mid-read."""

    manifest_path = Path(output_dir) / manifest_name
    for _ in range(3):
        manifest_text = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(manifest_text)
        snapshot = read_client_output_generation(output_dir, manifest)
        if manifest_path.read_text(encoding="utf-8") == manifest_text:
            return snapshot
    raise RuntimeError("client output manifest changed during read")


def atomic_write_client_outputs(
    output_dir: str | Path,
    recommendations: dict[str, dict[str, Any]],
    homepage_name: str = "rulename.json",
    rule_content_name: str = "rule_content.txt",
    config: RuleAssemblyConfig = RuleAssemblyConfig(),
) -> tuple[Path, Path]:
    output = Path(output_dir)
    homepage, rule_content = serialize_assembled_output(recommendations, config)
    homepage_path = output / homepage_name
    content_path = output / rule_content_name
    homepage_text = json.dumps(homepage, indent=2, ensure_ascii=False)
    generation = uuid4().hex
    manifest_path = output / "client_output_manifest.json"
    generation_relative = Path("generations") / generation
    manifest_text = json.dumps(
        {
            "schema_version": 1,
            "generation": generation,
            "files": {
                homepage_name: {
                    "path": str(generation_relative / homepage_name),
                    "sha256": _sha256(homepage_text),
                },
                rule_content_name: {
                    "path": str(generation_relative / rule_content_name),
                    "sha256": _sha256(rule_content),
                },
            },
        },
        indent=2,
        sort_keys=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    generations = output / "generations"
    generations.mkdir(exist_ok=True)
    generation_stage = generations / f".{generation}.generation.stage"
    generation_stage.mkdir()
    (generation_stage / homepage_name).write_text(homepage_text, encoding="utf-8")
    (generation_stage / rule_content_name).write_text(rule_content, encoding="utf-8")
    os.replace(generation_stage, generations / generation)

    for path, text in ((homepage_path, homepage_text), (content_path, rule_content)):
        mirror_stage = path.with_name(f"{path.name}.{generation}.mirror.stage")
        mirror_stage.write_text(text, encoding="utf-8")
        os.replace(mirror_stage, path)

    manifest_stage = manifest_path.with_name(
        f"{manifest_path.name}.{generation}.stage"
    )
    manifest_stage.write_text(manifest_text, encoding="utf-8")
    os.replace(manifest_stage, manifest_path)
    return homepage_path, content_path


def production_output_names(
    domain: str,
    date: str,
    id_low: int = 0,
    id_high: int = 700000,
    part: int = 0,
) -> dict[str, str]:
    """Filenames matching the production MyTV `rs_output/` convention.

    `id_low`/`id_high`/`part` are the source batch identifiers; they default to
    the values in the `rs_output/` sample and are cosmetic, not derived.
    """

    return {
        "homepage": f"f_personalized_homepage_rule_ms_{domain}_{id_low}_{id_high}_DL_MyTV_{date}.json",
        "rule_content": f"f_personalized_rule_content_ms_{id_low}_{id_high}_DL_MyTV_{date}.csv",
        "grouping": f"f_profile_grouping_ms_DL_MyTV_{part}_{date}.json",
    }


def build_profile_grouping(
    homepage: list[dict[str, str]],
    gid: int = 3,
    member_ids: Mapping[str, str] | None = None,
    group_ids: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Profile-grouping records matching the `rs_output/` fixture shape.

    A member owns one or more profiles and only the default profile carries the
    member id as its own id, so `mid` is looked up per profile from
    `member_ids` (the canonical `profile_mapping` join) rather than derived from
    `pid`. A profile missing from the mapping falls back to `pid`: the two
    match for 99.98% of profiles in the delivered data, but that is a property
    of the sample, not the rule. `gid` is the A/B grouping and has no source in the
    delivered data, so it stays a caller-supplied constant unless `group_ids`
    carries VNPT's assignment.
    """

    members = member_ids or {}
    groups = group_ids or {}
    records = []
    for record in homepage:
        pid = str(record["pid"])
        member = members.get(pid)
        records.append(
            {
                "pid": pid,
                "mid": str(member) if member not in (None, "") else pid,
                "gid": int(groups.get(pid, gid)),
            }
        )
    return records


def _json_array_one_record_per_line(records: list[dict[str, Any]]) -> str:
    if not records:
        return "[]\n"
    inner = ",\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    return f"[\n{inner}\n]\n"


def write_production_named_outputs(
    output_dir: str | Path,
    homepage: list[dict[str, str]],
    rule_content: str,
    *,
    domain: str,
    date: str,
    id_low: int = 0,
    id_high: int = 700000,
    part: int = 0,
    gid: int = 3,
    member_ids: Mapping[str, str] | None = None,
    group_ids: Mapping[str, int] | None = None,
) -> dict[str, Path]:
    """Write the three `rs_output/`-named client files (homepage, rule-content, grouping).

    `homepage` and `rule_content` are the already-serialized client payloads
    (identical bytes to `rulename.json` / `rule_content.txt`). `member_ids`
    resolves each profile's `mid`; see `build_profile_grouping`.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    names = production_output_names(domain, date, id_low, id_high, part)
    homepage_path = output / names["homepage"]
    content_path = output / names["rule_content"]
    grouping_path = output / names["grouping"]
    def _atomic_write(path: Path, payload: str) -> None:
        # Stage-and-replace so a crash or concurrent reader never sees a
        # truncated production file; os.replace is atomic per file.
        staged = path.with_suffix(path.suffix + ".staging")
        staged.write_text(payload, encoding="utf-8")
        os.replace(staged, path)

    _atomic_write(homepage_path, _json_array_one_record_per_line(homepage))
    _atomic_write(content_path, rule_content)
    _atomic_write(
        grouping_path,
        _json_array_one_record_per_line(
            build_profile_grouping(homepage, gid, member_ids, group_ids)
        ),
    )
    return {
        "homepage": homepage_path,
        "rule_content": content_path,
        "grouping": grouping_path,
    }


class JsonArrayWriter:
    """Streams records in the one-record-per-line array layout of the
    `rs_output/` fixtures, byte-identical to `_json_array_one_record_per_line`,
    without holding every record in memory."""

    def __init__(self, handle):
        self.handle = handle
        self.count = 0

    def write_line(self, record_json: str) -> None:
        self.handle.write(("[\n" if self.count == 0 else ",\n") + record_json)
        self.count += 1

    def close(self) -> None:
        self.handle.write("\n]\n" if self.count else "[]\n")


DELIVERY_POINTER = "CURRENT.json"
DELIVERY_MANIFEST = "manifest.json"


def commit_delivery_generation(
    delivery_root: str | Path,
    stage: str | Path,
    generation: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish a fully written stage directory as an immutable generation.

    Every file in ``stage`` is hashed into ``manifest.json``; the directory is
    renamed into ``generations/`` and re-verified from disk; only then is
    ``CURRENT.json`` replaced. That replace is the single commit point, so a
    failure anywhere before it leaves the previous delivery current.
    """
    root = Path(delivery_root)
    stage = Path(stage)
    files = {
        path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(stage.iterdir())
        if path.is_file() and path.name != DELIVERY_MANIFEST
    }
    manifest = {"schema_version": 1, "generation": generation, **metadata, "files": files}
    (stage / DELIVERY_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    target = root / "generations" / generation
    os.replace(stage, target)
    snapshot = read_delivery_generation(root, generation)
    atomic_write_text(
        root / DELIVERY_POINTER,
        json.dumps(
            {
                "schema_version": 1,
                "generation": generation,
                "manifest_sha256": sha256_file(target / DELIVERY_MANIFEST),
            },
            indent=2,
            sort_keys=True,
        ),
    )
    return snapshot


def read_delivery_generation(delivery_root: str | Path, generation: str) -> dict[str, Any]:
    """Manifest of a committed generation, after re-hashing every file."""
    directory = Path(delivery_root) / "generations" / generation
    manifest = json.loads((directory / DELIVERY_MANIFEST).read_text(encoding="utf-8"))
    for name, entry in manifest["files"].items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"delivery generation {generation} is corrupt: {name}")
    return {**manifest, "directory": str(directory)}


def read_current_delivery(delivery_root: str | Path) -> dict[str, Any] | None:
    """The committed delivery, verified, or None before the first publish."""
    root = Path(delivery_root)
    pointer = root / DELIVERY_POINTER
    if not pointer.is_file():
        return None
    current = json.loads(pointer.read_text(encoding="utf-8"))
    manifest_path = root / "generations" / current["generation"] / DELIVERY_MANIFEST
    if sha256_file(manifest_path) != current["manifest_sha256"]:
        raise ValueError(f"{pointer} does not match its generation manifest")
    return read_delivery_generation(root, current["generation"])
