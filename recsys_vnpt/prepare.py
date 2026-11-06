"""Raw MyTV sources to canonical artifacts.

Everything upstream of temporal label building: streaming the duration logs,
resolving identities, auditing what the catalog can and cannot serve, and
writing the canonical parquet set.
"""


from collections import defaultdict
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import pandas as pd

from recsys_vnpt.canonical import (
    ContentSnapshot,
    cache_fingerprint,
    cache_is_current,
    canonicalize_behavior_events,
    canonicalize_content_tags,
    canonicalize_content_versions,
    canonicalize_contents,
    normalize_identifier,
    normalize_user_metadata,
    normalize_username,
    source_timestamp,
    write_partitioned_interactions,
)
from recsys_vnpt.data_sources import load_user_metadata, resolve_user_metadata_path
from recsys_vnpt.engagement import (
    DEFAULT_SESSION_GAP_MINUTES,
    DEFAULT_TARGET_VERSION,
    apply_atomic_duration,
    build_playback_events,
    fit_label_contract,
)
from recsys_vnpt.impressions import build_impressions
from recsys_vnpt.metrics_schema import MetricBundle
from recsys_vnpt.metrics_store import (
    MetricsStore,
    create_manifest,
    evaluate_gate,
    render_markdown,
)
from recsys_vnpt.paths import data_artifact_path, rule_info_path, source_artifact_path, source_root
from recsys_vnpt.rule_assembly import SPECIAL_RULES
from recsys_vnpt.source_manifest import (
    build_source_artifacts,
    manifest_fingerprint,
    schema_hash,
    write_source_manifest,
)

def _json_frame(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return pd.DataFrame(item for item in payload if isinstance(item, dict))
    if isinstance(payload, dict) and payload and all(isinstance(value, dict) for value in payload.values()):
        return pd.DataFrame(payload.values())
    if isinstance(payload, dict):
        return pd.DataFrame([payload])
    raise ValueError(f"Unsupported JSON root in {path}: {type(payload).__name__}")


def _content_sources(project_root: Path, domain: str) -> tuple[list[ContentSnapshot], pd.DataFrame, list[dict[str, str]]]:
    root = source_artifact_path("mytv_vmp_content", project_root=project_root)
    snapshots: list[ContentSnapshot] = []
    tombstones: list[pd.DataFrame] = []
    skipped: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.json")):
        if "overall_content" in path.parts:
            continue
        try:
            display_path = path.relative_to(project_root).as_posix()
        except ValueError:
            display_path = path.relative_to(source_root(project_root)).as_posix()
        name = path.name.lower()
        try:
            frame = _json_frame(path)
        except Exception as exc:
            skipped.append({"path": display_path, "reason": str(exc)})
            continue
        columns = {column.lower(): column for column in frame.columns}
        if "deleted" in name and "content_id" in columns:
            deleted = frame[[columns["content_id"]]].rename(columns={columns["content_id"]: "content_id"})
            deleted["_source_timestamp"] = source_timestamp(path)
            tombstones.append(deleted)
            continue
        matched_family = None
        for family, pattern in (
            ("movie", "content_movie_2"),
            ("clip", "content_clip_2"),
            ("music", "music_video_2"),
        ):
            if name.startswith(pattern):
                matched_family = family
                break
        if matched_family is None:
            skipped.append(
                {"path": display_path, "reason": "unsupported_content_family"}
            )
            continue
        if domain != "all" and matched_family != domain:
            skipped.append(
                {"path": display_path, "reason": "outside_requested_domain"}
            )
            continue
        frame["content_family"] = matched_family
        snapshots.append(ContentSnapshot(display_path, source_timestamp(path), frame))
    tombstone_frame = pd.concat(tombstones, ignore_index=True) if tombstones else pd.DataFrame(columns=["content_id"])
    return snapshots, tombstone_frame, skipped


def _latest_json(directory: Path, name: str) -> Path:
    paths = sorted(directory.rglob(name))
    if not paths:
        raise FileNotFoundError(f"No {name} under {directory}")
    return paths[-1]


def _parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema.names)
    except ImportError:
        return list(pd.read_parquet(path).columns)


def build_rule_membership_from_ids(
    content_tags: pd.DataFrame,
    rule_info: pd.DataFrame,
) -> pd.DataFrame:
    required = {"content_id", "tag_id"}
    if not required.issubset(content_tags) or not {"RULE_ID", "LIST_TAGS_ID"}.issubset(rule_info):
        raise ValueError("rule membership requires content/tag IDs and rule tag sets")
    inverted: dict[str, set[str]] = defaultdict(set)
    for row in content_tags[["content_id", "tag_id"]].drop_duplicates().itertuples(index=False):
        inverted[str(row.tag_id)].add(str(row.content_id))
    rows: list[dict[str, str]] = []
    for rule in rule_info[["RULE_ID", "LIST_TAGS_ID"]].drop_duplicates().itertuples(index=False):
        rule_id = str(rule.RULE_ID)
        required_tags = [tag.strip() for tag in str(rule.LIST_TAGS_ID).split(",") if tag.strip() and tag.strip() != "-1"]
        if not required_tags:
            continue
        pools = [inverted.get(tag, set()) for tag in required_tags]
        if not pools or any(not pool for pool in pools):
            continue
        content_ids = set.intersection(*pools)
        rows.extend({"content_id": content_id, "rule_id": rule_id} for content_id in content_ids)
    return pd.DataFrame(rows, columns=["content_id", "rule_id"]).drop_duplicates()


def _iter_duration_chunks(
    project_root: Path,
    max_rows: int | None = None,
):
    required = ["profile_id", "username", "content_id", "duration"]
    optional = [
        "member_id",
        "create_date",
        "filename_date",
        "uuid",
        "request_id",
        "recommendation",
        "refer_path",
        "date_time",
    ]
    remaining = max_rows
    duration_root = source_artifact_path("duration", project_root=project_root)
    found = False
    for path in sorted(duration_root.glob("*.csv")):
        found = True
        available = list(pd.read_csv(path, nrows=0).columns)
        missing = sorted(set(required) - set(available))
        if missing:
            raise ValueError(f"{path} is missing duration columns: {missing}")
        selected = required + [column for column in optional if column in available]
        for chunk in pd.read_csv(path, usecols=selected, chunksize=500_000, low_memory=False):
            if remaining is not None:
                chunk = chunk.head(remaining)
                remaining -= len(chunk)
            if not chunk.empty:
                yield _normalize_duration_chunk(chunk)
            if remaining is not None and remaining <= 0:
                break
        if remaining is not None and remaining <= 0:
            break
    if remaining is None or remaining > 0:
        for path in sorted(duration_root.glob("*.parquet")):
            found = True
            available = _parquet_columns(path)
            missing = sorted(set(required) - set(available))
            if missing:
                raise ValueError(f"{path} is missing duration columns: {missing}")
            selected = required + [column for column in optional if column in available]
            frame = pd.read_parquet(path, columns=selected)
            if remaining is not None:
                frame = frame.head(remaining)
                remaining -= len(frame)
            if not frame.empty:
                yield _normalize_duration_chunk(frame)
            if remaining is not None and remaining <= 0:
                break
    if not found:
        raise FileNotFoundError("No duration CSV or parquet sources")


def _normalize_duration_chunk(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "member_id" not in output:
        output["member_id"] = pd.NA
    if "create_date" not in output:
        if "filename_date" not in output:
            raise ValueError("Duration data require create_date or filename_date for temporal evaluation")
        output["create_date"] = pd.to_datetime(output["filename_date"].astype("string"), format="%Y%m%d", errors="coerce")
    output["create_date"] = pd.to_datetime(output["create_date"], errors="coerce")
    if "date_time" not in output:
        output["date_time"] = pd.NaT
    output["date_time"] = pd.to_datetime(output["date_time"], errors="coerce")
    output["duration"] = pd.to_numeric(output["duration"], errors="coerce").fillna(0.0)
    for column in ("profile_id", "content_id", "member_id"):
        output[column] = normalize_identifier(output[column])
    output["username"] = output["username"].astype("string").str.strip()
    for column in (
        "uuid",
        "request_id",
        "recommendation",
        "refer_path",
        "filename_date",
    ):
        if column not in output:
            output[column] = pd.Series(pd.NA, index=output.index, dtype="string")
        else:
            output[column] = output[column].astype("string").str.strip()
    return output


def _read_duration(project_root: Path, max_rows: int | None = None) -> pd.DataFrame:
    frames = list(_iter_duration_chunks(project_root, max_rows))
    if not frames:
        raise FileNotFoundError("No duration CSV or parquet sources")
    return pd.concat(frames, ignore_index=True)


@dataclass(frozen=True)
class StreamedDuration:
    interaction_rows: int
    interaction_columns: tuple[str, ...]
    interaction_parts: int
    raw_username_counts: pd.DataFrame
    username_counts: pd.DataFrame
    member_counts: pd.DataFrame


def _combine_pair_counts(
    frames: list[pd.DataFrame],
    account_col: str,
) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(
            columns=["profile_id", account_col, "mapping_interactions"]
        )
    return (
        pd.concat(frames, ignore_index=True)
        .groupby(["profile_id", account_col], as_index=False, dropna=False)[
            "mapping_interactions"
        ]
        .sum()
    )


def _stream_duration_dataset(
    project_root: Path,
    interactions_path: Path,
    date_partition_path: Path,
) -> StreamedDuration:
    staging_token = f"{os.getpid()}_{time.time_ns()}"
    temp_interactions = interactions_path.with_name(
        f".{interactions_path.name}.{staging_token}.tmp"
    )
    temp_dates = date_partition_path.with_name(
        f".{date_partition_path.name}.{staging_token}.tmp"
    )
    for path in (temp_interactions, temp_dates):
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        path.mkdir(parents=True, exist_ok=True)

    raw_counts: list[pd.DataFrame] = []
    username_counts: list[pd.DataFrame] = []
    member_counts: list[pd.DataFrame] = []
    rows = 0
    columns: tuple[str, ...] = ()
    dates: set[str] = set()
    try:
        for part_index, raw_chunk in enumerate(
            _iter_duration_chunks(project_root, max_rows=None)
        ):
            chunk = raw_chunk.copy()
            raw_pair = (
                chunk[["profile_id", "username"]]
                .dropna()
                .groupby(["profile_id", "username"], as_index=False)
                .size()
                .rename(columns={"size": "mapping_interactions"})
            )
            raw_counts.append(raw_pair)
            chunk["username"] = normalize_username(chunk["username"])
            username_counts.append(
                chunk[["profile_id", "username"]]
                .dropna()
                .groupby(["profile_id", "username"], as_index=False)
                .size()
                .rename(columns={"size": "mapping_interactions"})
            )
            valid_member = chunk[["profile_id", "member_id"]].dropna().copy()
            valid_member["member_id"] = (
                valid_member["member_id"].astype("string").str.strip()
            )
            valid_member = valid_member[
                valid_member["member_id"].ne("")
                & valid_member["member_id"].ne("0")
            ]
            if not valid_member.empty:
                member_counts.append(
                    valid_member.groupby(
                        ["profile_id", "member_id"], as_index=False
                    )
                    .size()
                    .rename(columns={"size": "mapping_interactions"})
                )

            chunk.to_parquet(
                temp_interactions / f"part_{part_index:05d}.parquet",
                index=False,
            )
            partition_dates = pd.to_datetime(
                chunk["create_date"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
            for date, date_rows in chunk.assign(
                _partition_date=partition_dates
            ).dropna(subset=["_partition_date"]).groupby(
                "_partition_date", sort=True
            ):
                date = str(date)
                dates.add(date)
                directory = temp_dates / f"event_date={date}"
                directory.mkdir(parents=True, exist_ok=True)
                date_rows.drop(columns="_partition_date").to_parquet(
                    directory / f"part_{part_index:05d}.parquet",
                    index=False,
                )
            rows += len(chunk)
            columns = tuple(map(str, chunk.columns))

        if rows == 0:
            raise FileNotFoundError("No duration rows were read")
        for target in (interactions_path, date_partition_path):
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
        temp_interactions.replace(interactions_path)
        temp_dates.replace(date_partition_path)
    except Exception:
        for path in (temp_interactions, temp_dates):
            if path.exists():
                shutil.rmtree(path)
        raise

    return StreamedDuration(
        interaction_rows=rows,
        interaction_columns=columns,
        interaction_parts=len(dates),
        raw_username_counts=_combine_pair_counts(raw_counts, "username"),
        username_counts=_combine_pair_counts(username_counts, "username"),
        member_counts=_combine_pair_counts(member_counts, "member_id"),
    )


def _mapping_from_pair_counts(
    pair_counts: pd.DataFrame,
    account_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    required = {"profile_id", account_col, "mapping_interactions"}
    if not required.issubset(pair_counts):
        raise ValueError(f"pair counts are missing {sorted(required - set(pair_counts))}")
    pairs = pair_counts.dropna(subset=["profile_id", account_col]).copy()
    pairs["profile_id"] = normalize_identifier(pairs["profile_id"])
    pairs[account_col] = normalize_identifier(pairs[account_col])
    pairs = pairs[
        pairs["profile_id"].ne("") & pairs[account_col].ne("")
    ]
    pairs = (
        pairs.groupby(["profile_id", account_col], as_index=False)[
            "mapping_interactions"
        ]
        .sum()
        .sort_values(
            ["profile_id", "mapping_interactions", account_col],
            ascending=[True, False, True],
            kind="stable",
        )
    )
    by_profile = pairs.groupby("profile_id")[account_col].nunique()
    by_account = pairs.groupby(account_col)["profile_id"].nunique()
    conflicts = pd.concat(
        [
            pairs[pairs["profile_id"].isin(by_profile[by_profile > 1].index)].assign(
                conflict="profile_to_accounts"
            ),
            pairs[pairs[account_col].isin(by_account[by_account > 1].index)].assign(
                conflict="account_to_profiles"
            ),
        ],
        ignore_index=True,
    ).drop_duplicates()
    mapping = pairs.drop_duplicates("profile_id", keep="first").reset_index(
        drop=True
    )
    metrics = {
        "unique_pairs": int(len(pairs)),
        "profiles": int(pairs["profile_id"].nunique()),
        "accounts": int(pairs[account_col].nunique()),
        "profiles_with_multiple_accounts": int((by_profile > 1).sum()),
        "accounts_with_multiple_profiles": int((by_account > 1).sum()),
    }
    return mapping, conflicts, metrics


def _write_behavior_parts(
    project_root: Path,
    output_dir: Path,
    max_rows: int | None,
) -> tuple[list[Path], dict[str, int]]:
    source_paths = sorted(source_artifact_path("behavior", project_root=project_root).glob("*.csv"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("part_*.parquet"):
        stale.unlink()
    remaining = max_rows
    output_paths: list[Path] = []
    source_rows = canonical_rows = quarantine_rows = 0
    part_index = 0
    desired = {"profile_id", "user_name", "username", "content_id", "action", "duration", "create_date"}
    for source in source_paths:
        available = list(pd.read_csv(source, nrows=0).columns)
        usecols = [column for column in available if column in desired]
        for chunk in pd.read_csv(source, usecols=usecols, chunksize=500_000, low_memory=False):
            if remaining is not None:
                chunk = chunk.head(remaining)
                remaining -= len(chunk)
            result = canonicalize_behavior_events(chunk)
            source_rows += len(chunk)
            canonical_rows += len(result.frame)
            quarantine_rows += len(result.quarantine)
            if not result.frame.empty:
                path = output_dir / f"part_{part_index:05d}.parquet"
                result.frame.to_parquet(path, index=False)
                output_paths.append(path)
                part_index += 1
            if remaining is not None and remaining <= 0:
                break
        if remaining is not None and remaining <= 0:
            break
    return output_paths, {
        "behavior_source_rows": source_rows,
        "behavior_canonical_rows": canonical_rows,
        "behavior_quarantine_rows": quarantine_rows,
        "behavior_parts": len(output_paths),
    }


def _canonical_output_paths(output: Path) -> dict[str, Path]:
    return {
        "users": output / "users.parquet",
        "interactions": output / "interactions.parquet",
        "profile_mapping": output / "profile_mapping.parquet",
        "contents": output / "contents.parquet",
        "content_versions": output / "content_versions.parquet",
        "content_tags": output / "content_tags.parquet",
        "membership": output / "content_rule_membership.parquet",
        "playback_events": output / "playback_events.parquet",
        "sessions": output / "sessions.parquet",
        "label_contract": output / "label_contract_v4.json",
        "label_metrics": output / "label_metrics.json",
    }


def _read_canonical_interactions(
    path: Path,
    content_ids: set[str] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read a canonical file/dataset with lazy catalog filtering for full runs."""
    if not path.exists():
        raise FileNotFoundError(f"Missing canonical interactions at {path}")
    if path.is_dir():
        import polars as pl

        scan = pl.scan_parquet(str(path / "*.parquet"))
        available = set(scan.collect_schema().names())
        selected = (
            [column for column in columns if column in available]
            if columns is not None
            else sorted(available)
        )
        if content_ids is not None:
            scan = scan.filter(
                pl.col("content_id").cast(pl.String).is_in(
                    sorted(map(str, content_ids))
                )
            )
        frame = scan.select(selected).collect(engine="streaming").to_pandas()
    else:
        frame = pd.read_parquet(path, columns=columns)
        if content_ids is not None:
            frame = frame[
                frame["content_id"].astype(str).isin(set(map(str, content_ids)))
            ]
    if "create_date" in frame:
        frame["create_date"] = pd.to_datetime(
            frame["create_date"], errors="coerce"
        )
    return frame.reset_index(drop=True)


def _interaction_catalog_audit(
    path: Path,
    eligible_content_ids: set[str],
    version_content_ids: set[str],
) -> dict[str, Any]:
    """Explain interaction scope filtering without treating exclusions as errors."""
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    class_names = ("eligible", "historical_version_only", "unknown")
    rows = {name: 0 for name in class_names}
    distinct = {name: set() for name in class_names}
    profiles = {name: set() for name in class_names}
    channel_hints = {name: 0 for name in class_names}
    missing_refer_path = {name: 0 for name in class_names}
    total_rows = 0
    missing_date_time_rows = 0
    duration_over_six_hours = 0
    duration_over_day = 0
    maximum_duration = 0.0
    channel_pattern = r"channel|kenh|truyen_hinh"
    for file in files:
        available = set(_parquet_columns(file))
        selected = [
            column
            for column in ("content_id", "profile_id", "refer_path", "date_time", "duration")
            if column in available
        ]
        frame = pd.read_parquet(file, columns=selected)
        total_rows += len(frame)
        content_ids = frame["content_id"].astype("string")
        eligible_mask = content_ids.astype(str).isin(eligible_content_ids)
        version_mask = content_ids.astype(str).isin(version_content_ids)
        masks = {
            "eligible": eligible_mask,
            "historical_version_only": version_mask & ~eligible_mask,
            "unknown": ~version_mask,
        }
        refer_path = frame.get(
            "refer_path", pd.Series(pd.NA, index=frame.index, dtype="string")
        ).astype("string")
        channel_mask = refer_path.fillna("").str.contains(
            channel_pattern, case=False, regex=True
        )
        missing_path_mask = refer_path.isna() | refer_path.str.strip().eq("")
        for name, mask in masks.items():
            subset = frame.loc[mask]
            rows[name] += int(mask.sum())
            distinct[name].update(subset["content_id"].dropna().astype(str).unique())
            if "profile_id" in subset:
                profiles[name].update(subset["profile_id"].dropna().astype(str).unique())
            channel_hints[name] += int((mask & channel_mask).sum())
            missing_refer_path[name] += int((mask & missing_path_mask).sum())
        eligible_frame = frame.loc[eligible_mask]
        if "date_time" in eligible_frame:
            missing_date_time_rows += int(
                pd.to_datetime(eligible_frame["date_time"], errors="coerce").isna().sum()
            )
        else:
            missing_date_time_rows += len(eligible_frame)
        durations = pd.to_numeric(
            eligible_frame.get("duration", pd.Series(dtype=float)), errors="coerce"
        ).fillna(0.0)
        duration_over_six_hours += int(durations.gt(6 * 60 * 60).sum())
        duration_over_day += int(durations.gt(24 * 60 * 60).sum())
        if len(durations):
            maximum_duration = max(maximum_duration, float(durations.max()))
    by_class = {
        name: {
            "rows": rows[name],
            "row_rate": rows[name] / max(total_rows, 1),
            "distinct_content_ids": len(distinct[name]),
            "profiles": len(profiles[name]),
            "channel_path_hint_rows": channel_hints[name],
            "channel_path_hint_rate": channel_hints[name] / max(rows[name], 1),
            "missing_refer_path_rows": missing_refer_path[name],
            "missing_refer_path_rate": missing_refer_path[name] / max(rows[name], 1),
        }
        for name in class_names
    }
    return {
        "source_rows": total_rows,
        "classification_policy": (
            "eligible=current active tagged positive-duration catalog; "
            "historical_version_only=known content version but not eligible; "
            "unknown=no supplied content version"
        ),
        "channel_hint_semantics": "refer_path regex only; not a definitive content family",
        "by_catalog_class": by_class,
        "eligible_missing_start_time_rows": missing_date_time_rows,
        "eligible_missing_start_time_rate": missing_date_time_rows
        / max(rows["eligible"], 1),
        "eligible_duration_over_6h_rows": duration_over_six_hours,
        "eligible_duration_over_24h_rows": duration_over_day,
        "eligible_maximum_duration_seconds": maximum_duration,
    }


def prepare_canonical_project(
    project_root: str | Path = ".",
    domain: str = "all",
    max_interaction_rows: int | None = None,
    max_behavior_rows: int | None = None,
    include_behavior: bool = True,
    force: bool = False,
    full_source_hash: bool = False,
    target_version: str = DEFAULT_TARGET_VERSION,
    session_gap_minutes: int = DEFAULT_SESSION_GAP_MINUTES,
    user_reference_date: str | pd.Timestamp = "2025-02-20",
) -> tuple[dict[str, Path], dict[str, Any]]:
    if max_interaction_rows is not None and max_interaction_rows < 1:
        raise ValueError("max_interaction_rows must be positive")
    if max_behavior_rows is not None and max_behavior_rows < 1:
        raise ValueError("max_behavior_rows must be positive")
    if target_version != DEFAULT_TARGET_VERSION:
        raise ValueError(f"unsupported target version: {target_version}")
    if session_gap_minutes < 1:
        raise ValueError("session_gap_minutes must be positive")
    root = Path(project_root).resolve()
    output = data_artifact_path("canonical", domain, project_root=root)
    output.mkdir(parents=True, exist_ok=True)
    paths = _canonical_output_paths(output)
    skipped: list[dict[str, str]] = []

    content_root = source_artifact_path("mytv_vmp_content", project_root=root)
    source_groups = {
        "user": [resolve_user_metadata_path(root)],
        "interaction": list(source_artifact_path("duration", project_root=root).glob("*.csv"))
        + list(source_artifact_path("duration", project_root=root).glob("*.parquet")),
        "content": [path for path in content_root.rglob("*.json") if "overall_content" not in path.parts],
        "tag": list(source_artifact_path("tags", project_root=root).rglob("TAG.json"))
        + list(source_artifact_path("content_tags", project_root=root).rglob("CONTENT_TO_TAG.json")),
        "rule": [rule_info_path(root)],
    }
    if include_behavior:
        source_groups["behavior"] = list(source_artifact_path("behavior", project_root=root).glob("*.csv"))
    artifacts = [
        artifact
        for role, source_paths in source_groups.items()
        for artifact in build_source_artifacts(source_paths, role, root=root, full_hash=full_source_hash)
    ]
    fingerprint = cache_fingerprint(
        manifest_fingerprint(artifacts),
        {
            "domain": domain,
            "max_interaction_rows": max_interaction_rows,
            "max_behavior_rows": max_behavior_rows,
            "include_behavior": include_behavior,
            "target_version": target_version,
            "session_gap_minutes": session_gap_minutes,
        },
        schema_version=10,
    )
    cache_path = output / "cache_metadata.json"
    behavior_cache_ready = (
        not include_behavior or any((output / "behavior").glob("part_*.parquet"))
    )
    if (
        not force
        and cache_is_current(cache_path, fingerprint)
        and all(path.exists() for path in paths.values())
        and behavior_cache_ready
    ):
        data_metrics = json.loads((output / "data_metrics.json").read_text(encoding="utf-8"))
        return paths, data_metrics

    users_result = normalize_user_metadata(
        load_user_metadata(project_root=root),
        reference_date=user_reference_date,
    )
    users_path = paths["users"]
    users_result.frame.to_parquet(users_path, index=False)
    users_result.quarantine.to_parquet(output / "users_quarantine.parquet", index=False)

    interactions_path = paths["interactions"]
    date_partition_path = output / "interactions_by_date"
    if max_interaction_rows is None:
        streamed = _stream_duration_dataset(
            root, interactions_path, date_partition_path
        )
        raw_username_pair_counts = streamed.raw_username_counts
        username_pair_counts = streamed.username_counts
        member_pair_counts = streamed.member_counts
        interaction_rows = streamed.interaction_rows
        interaction_columns = streamed.interaction_columns
        interaction_part_count = streamed.interaction_parts
    else:
        interactions = _read_duration(root, max_interaction_rows)
        interactions["profile_id"] = normalize_identifier(
            interactions["profile_id"]
        )
        interactions["content_id"] = normalize_identifier(
            interactions["content_id"]
        )
        raw_username_pair_counts = (
            interactions[["profile_id", "username"]]
            .dropna()
            .groupby(["profile_id", "username"], as_index=False)
            .size()
            .rename(columns={"size": "mapping_interactions"})
        )
        interactions["username"] = normalize_username(interactions["username"])
        username_pair_counts = (
            interactions[["profile_id", "username"]]
            .dropna()
            .groupby(["profile_id", "username"], as_index=False)
            .size()
            .rename(columns={"size": "mapping_interactions"})
        )
        member_pair_counts = (
            interactions[["profile_id", "member_id"]]
            .dropna()
            .groupby(["profile_id", "member_id"], as_index=False)
            .size()
            .rename(columns={"size": "mapping_interactions"})
        )
        for generated in (interactions_path, date_partition_path):
            if generated.exists():
                if generated.is_dir():
                    shutil.rmtree(generated)
                else:
                    generated.unlink()
        interactions.to_parquet(interactions_path, index=False)
        interaction_parts = write_partitioned_interactions(
            interactions, date_partition_path
        )
        interaction_rows = len(interactions)
        interaction_columns = tuple(map(str, interactions.columns))
        interaction_part_count = len(interaction_parts)

    raw_username_pairs = raw_username_pair_counts[
        ["profile_id", "username"]
    ].dropna().copy()
    raw_username_pairs["username"] = normalize_identifier(
        raw_username_pairs["username"]
    )
    raw_username_pairs = raw_username_pairs.drop_duplicates()
    raw_username_counts = raw_username_pairs.groupby("profile_id")["username"].nunique()
    casefold_pairs = raw_username_pairs.copy()
    casefold_pairs["username"] = normalize_username(casefold_pairs["username"])
    casefold_counts = casefold_pairs.drop_duplicates().groupby("profile_id")[
        "username"
    ].nunique()
    case_variant_profiles = set(
        raw_username_counts[raw_username_counts > 1].index
    ) - set(casefold_counts[casefold_counts > 1].index)
    raw_username_pairs[
        raw_username_pairs["profile_id"].isin(case_variant_profiles)
    ].to_parquet(output / "username_case_variants.parquet", index=False)
    profile_mapping, conflicts, mapping_metrics = _mapping_from_pair_counts(
        username_pair_counts, "username"
    )
    conflicts.to_parquet(output / "profile_mapping_conflicts.parquet", index=False)
    member_pair_counts["member_id"] = (
        member_pair_counts["member_id"].astype("string").str.strip()
    )
    member_pair_counts = member_pair_counts[
        member_pair_counts["member_id"].ne("")
        & member_pair_counts["member_id"].ne("0")
    ]
    if member_pair_counts.empty:
        member_conflicts = pd.DataFrame(
            columns=["profile_id", "member_id", "conflict"]
        )
        member_metrics = {
            "unique_pairs": 0,
            "profiles": 0,
            "accounts": 0,
            "profiles_with_multiple_accounts": 0,
            "accounts_with_multiple_profiles": 0,
        }
    else:
        member_mapping, member_conflicts, member_metrics = (
            _mapping_from_pair_counts(member_pair_counts, "member_id")
        )
        member_mapping = member_mapping.rename(
            columns={"mapping_interactions": "member_mapping_interactions"}
        )
        profile_mapping = profile_mapping.merge(
            member_mapping, on="profile_id", how="left"
        )
    member_conflicts.to_parquet(
        output / "member_mapping_conflicts.parquet", index=False
    )
    member_mapping_metrics = {
        "member_unique_pairs": member_metrics["unique_pairs"],
        "profiles_with_member_id": member_metrics["profiles"],
        "member_ids": member_metrics["accounts"],
        "profiles_with_multiple_member_ids": member_metrics[
            "profiles_with_multiple_accounts"
        ],
        "member_ids_with_multiple_profiles": member_metrics[
            "accounts_with_multiple_profiles"
        ],
    }
    profile_mapping.to_parquet(output / "profile_mapping.parquet", index=False)

    snapshots, tombstones, content_skipped = _content_sources(root, domain)
    skipped.extend(content_skipped)
    content_result = canonicalize_contents(snapshots, tombstones, require_recommendable=False)
    content_versions = canonicalize_content_versions(snapshots)
    content_versions.to_parquet(paths["content_versions"], index=False)
    if not content_result.quarantine.empty:
        skipped.extend(content_result.quarantine.to_dict("records"))

    tag_path = _latest_json(source_artifact_path("tags", project_root=root), "TAG.json")
    content_tag_path = _latest_json(source_artifact_path("content_tags", project_root=root), "CONTENT_TO_TAG.json")
    tags = _json_frame(tag_path)
    content_tags_raw = _json_frame(content_tag_path)
    content_tags = canonicalize_content_tags(content_tags_raw, tags)
    tag_names = content_tags.groupby("content_id")["tag_name"].agg(lambda values: ", ".join(sorted(set(map(str, values))))).rename("canonical_tag_names")
    contents = content_result.frame.copy()
    contents["content_id"] = contents["content_id"].astype("string")
    contents = contents.merge(tag_names, left_on="content_id", right_index=True, how="left")
    contents["tag_names"] = contents["canonical_tag_names"].fillna(contents.get("tag_names", ""))
    contents = contents.drop(columns=["canonical_tag_names"], errors="ignore")
    if "content_status" in contents:
        contents = contents[contents["content_status"].astype(str).eq("1")]
    contents = contents[contents["tag_names"].fillna("").str.contains(r"\w", regex=True)]
    if "content_duration" in contents:
        contents["content_duration"] = pd.to_numeric(contents["content_duration"], errors="coerce")
        contents = contents[contents["content_duration"].gt(0)]
    contents = contents.drop_duplicates("content_id", keep="last").reset_index(drop=True)
    contents_path = paths["contents"]
    content_tags_path = paths["content_tags"]
    content_tags.to_parquet(content_tags_path, index=False)

    rules = pd.read_parquet(rule_info_path(root))
    membership = build_rule_membership_from_ids(content_tags, rules)
    membership = membership[membership["content_id"].astype(str).isin(set(contents["content_id"].astype(str)))]
    membership_path = paths["membership"]
    membership.to_parquet(membership_path, index=False)

    interaction_catalog_audit = _interaction_catalog_audit(
        interactions_path,
        set(contents["content_id"].astype(str)),
        set(content_versions["content_id"].dropna().astype(str)),
    )
    playback_source = _read_canonical_interactions(
        interactions_path,
        content_ids=set(contents["content_id"].astype(str)),
    )
    playback_events, sessions, label_metrics = build_playback_events(
        playback_source,
        session_gap_minutes=session_gap_minutes,
    )
    # Fit the gain references on history ending 20 days before the data's last
    # event, the end of the training period under the 5x10-day window scheme, so
    # the references never see the validation or test labels on any data period.
    max_playback_event = pd.to_datetime(
        playback_events["event_end"], errors="coerce"
    ).max()
    label_fit_cutoff = (
        max_playback_event.normalize()
        - pd.Timedelta(days=20)
        + pd.offsets.Day(1)
        - pd.Timedelta(1, unit="ns")
        if pd.notna(max_playback_event)
        else pd.Timestamp("2025-01-31 23:59:59.999999999")
    )
    label_contract = fit_label_contract(
        contents,
        playback_events,
        domain=domain,
        fit_cutoff=label_fit_cutoff,
        session_gap_minutes=session_gap_minutes,
    )
    contents = apply_atomic_duration(contents, label_contract)
    contents.to_parquet(contents_path, index=False)
    playback_events.to_parquet(paths["playback_events"], index=False)
    sessions.to_parquet(paths["sessions"], index=False)
    label_contract.save(paths["label_contract"])
    label_metrics.update(
        {
            "label_contract_hash": label_contract.artifact_hash,
            "label_contract_fit_cutoff": label_contract.fit_cutoff,
            "atomic_duration_fallback_rows": int(
                contents["atomic_duration_fallback"].sum()
            ),
            "atomic_duration_fallback_rate": float(
                contents["atomic_duration_fallback"].mean()
            )
            if len(contents)
            else 0.0,
        }
    )
    paths["label_metrics"].write_text(
        json.dumps(label_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    counts = membership[~membership["rule_id"].astype(str).isin(SPECIAL_RULES)].groupby("rule_id")["content_id"].nunique()
    behavior_metrics: dict[str, int] = {}
    if include_behavior:
        _, behavior_metrics = _write_behavior_parts(root, output / "behavior", max_behavior_rows)
        # Shown-not-watched exposure, used only when impression negatives are requested.
        impression_sources = sorted(
            source_artifact_path("behavior", project_root=root).glob("*.csv")
        )
        if impression_sources:
            impressions, impression_metrics = build_impressions(
                impression_sources,
                set(contents["content_id"].astype(str)),
            )
            impressions.to_parquet(output / "impressions.parquet", index=False)
            (output / "impression_metrics.json").write_text(
                json.dumps(impression_metrics, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            behavior_metrics.update(
                {
                    key: value
                    for key, value in impression_metrics.items()
                    if isinstance(value, (int, float))
                }
            )
    if "_source_timestamp" in contents and "_source_timestamp" in tombstones:
        latest_delete = tombstones.groupby(tombstones["content_id"].astype("string"))["_source_timestamp"].max()
        deleted_at = contents["content_id"].map(latest_delete)
        tombstone_leakage = int((deleted_at.notna() & (contents["_source_timestamp"] <= deleted_at)).sum())
    else:
        tombstone_leakage = int(contents["content_id"].astype(str).isin(set(tombstones.get("content_id", pd.Series(dtype=str)).astype(str))).sum())
    role_schemas = {
        "user": schema_hash(users_result.frame.columns),
        "interaction": schema_hash(interaction_columns),
        "content": schema_hash(contents.columns),
        "tag": schema_hash(content_tags.columns),
        "rule": schema_hash(rules.columns),
        "behavior": schema_hash(
            ["profile_id", "username", "content_id", "action", "duration", "create_date"]
        ),
    }
    artifacts = [
        replace(artifact, schema_hash=role_schemas.get(artifact.role))
        for artifact in artifacts
    ]
    source_fingerprint = manifest_fingerprint(artifacts)
    data_metrics = {
        **content_result.metrics,
        **mapping_metrics,
        **member_mapping_metrics,
        "profiles_with_username_case_variants": len(case_variant_profiles),
        "interaction_rows": interaction_rows,
        "interaction_date_partitions": interaction_part_count,
        "content_version_rows": len(content_versions),
        "canonical_user_rows": len(users_result.frame),
        "canonical_content_rows": len(contents),
        "content_rule_rows": len(membership),
        "eligible_normal_rules": int((counts >= 10).sum()),
        "tombstone_leakage": tombstone_leakage,
        "missing_required_columns": int(sum(item.get("reason") == "missing_content_id" for item in skipped)),
        "silent_skipped_files": 0,
        "reported_skipped_files": len(skipped),
        "users_schema_hash": schema_hash(users_result.frame.columns),
        "interactions_schema_hash": schema_hash(interaction_columns),
        "contents_schema_hash": schema_hash(contents.columns),
        "playback_events_schema_hash": schema_hash(playback_events.columns),
        "label_contract_hash": label_contract.artifact_hash,
        "target_version": target_version,
        **label_metrics,
        "source_interaction_rows": interaction_rows,
        "playback_source_rows": len(playback_source),
        "playback_source_rate": len(playback_source) / max(interaction_rows, 1),
        "interaction_catalog_audit": interaction_catalog_audit,
        "source_fingerprint": source_fingerprint,
        "canonical_fingerprint": fingerprint,
        **behavior_metrics,
    }
    write_source_manifest(output / "source_manifest.json", artifacts, skipped)
    manifest = create_manifest(
        root,
        domain,
        "sprint-0",
        "canonical_data_contract",
        hashes={"source": source_fingerprint, "canonical": fingerprint},
        config={
            "max_interaction_rows": max_interaction_rows,
            "max_behavior_rows": max_behavior_rows,
            "include_behavior": include_behavior,
            "target_version": target_version,
            "session_gap_minutes": session_gap_minutes,
        },
    )
    bundle = MetricBundle(manifest=manifest, data=data_metrics)
    store = MetricsStore(root)
    directory = store.write_bundle(bundle)
    gate = evaluate_gate(bundle, "sprint-0")
    (directory / "gate_result.json").write_text(
        json.dumps(gate.to_dict(), indent=2), encoding="utf-8"
    )
    report = render_markdown(bundle, gate)
    (directory / "report.md").write_text(report, encoding="utf-8")
    data_metrics["metrics_run_id"] = manifest.run_id
    (output / "data_metrics.json").write_text(
        json.dumps(data_metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    cache_path.write_text(json.dumps({"fingerprint": fingerprint}, indent=2), encoding="utf-8")
    return paths, data_metrics
