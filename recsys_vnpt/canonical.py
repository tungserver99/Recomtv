from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from recsys_vnpt.source_manifest import stable_hash


GENERATED_PARTS = {"train_data", "infer_data", "merged_duration", "model", "result", "canonical", "metrics"}
DERIVED_CONTENT_PARTS = {"overall_content"}


def normalize_identifier(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip()
    return normalized.str.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)


def normalize_username(values: pd.Series) -> pd.Series:
    return normalize_identifier(values).str.casefold()


@dataclass(frozen=True)
class ContentSnapshot:
    source: str
    timestamp: pd.Timestamp
    frame: pd.DataFrame


@dataclass(frozen=True)
class CanonicalResult:
    frame: pd.DataFrame
    quarantine: pd.DataFrame
    metrics: dict[str, int | float | str]


def is_raw_source_path(path: str | Path) -> bool:
    parts = set(Path(path).parts)
    return not bool(parts & (GENERATED_PARTS | DERIVED_CONTENT_PARTS))


def source_timestamp(path: str | Path) -> pd.Timestamp:
    name = Path(path).name
    matches = re.findall(r"(20\d{6})(?:[_-]?(\d{6}))?", name)
    if not matches:
        return pd.Timestamp.min
    date, clock = matches[-1]
    return pd.to_datetime(date + (clock or "000000"), format="%Y%m%d%H%M%S")


def normalize_user_metadata(
    users: pd.DataFrame,
    reference_date: str | pd.Timestamp = "2025-02-20",
) -> CanonicalResult:
    required = {"username", "sex", "birthday", "package_code"}
    missing = sorted(required - set(users.columns))
    if missing:
        raise ValueError(f"user metadata is missing columns: {missing}")

    work = users.copy()
    issues: list[dict[str, object]] = []
    reference = pd.Timestamp(reference_date)

    raw_birthday = work["birthday"].astype("string")
    years = pd.to_numeric(raw_birthday.str.extract(r"^(\d{4})\d{4}$", expand=False), errors="coerce")
    valid_year = years.between(1900, reference.year)
    invalid_birthday = work["birthday"].notna() & ~valid_year
    for index in work.index[invalid_birthday]:
        issues.append({"row_index": index, "field": "birthday", "value": work.at[index, "birthday"], "reason": "invalid_yyyymmdd"})
    work["birth_year"] = years.where(valid_year)
    work["age"] = (reference.year - work["birth_year"]).where(valid_year)

    sex = pd.to_numeric(work["sex"], errors="coerce")
    invalid_sex = work["sex"].notna() & ~sex.isin([0, 1])
    for index in work.index[invalid_sex]:
        issues.append({"row_index": index, "field": "sex", "value": work.at[index, "sex"], "reason": "unsupported_sex"})
    work["sex"] = sex.where(sex.isin([0, 1])).astype("Int8")

    work["username"] = normalize_username(work["username"])
    empty_username = work["username"].isna() | work["username"].eq("")
    for index in work.index[empty_username]:
        issues.append({"row_index": index, "field": "username", "value": None, "reason": "missing_key"})
    work = work[~empty_username].drop_duplicates("username", keep="last").reset_index(drop=True)
    quarantine = pd.DataFrame(issues, columns=["row_index", "field", "value", "reason"])
    return CanonicalResult(
        frame=work,
        quarantine=quarantine,
        metrics={
            "source_rows": int(len(users)),
            "canonical_rows": int(len(work)),
            "quarantine_issues": int(len(quarantine)),
            "duplicate_usernames_removed": int(users["username"].duplicated().sum()),
        },
    )


def audit_profile_mapping(
    interactions: pd.DataFrame,
    profile_col: str = "profile_id",
    account_col: str = "username",
) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {profile_col, account_col}
    missing = sorted(required - set(interactions.columns))
    if missing:
        raise ValueError(f"profile mapping is missing columns: {missing}")
    pairs = interactions[[profile_col, account_col]].dropna().copy()
    pairs[profile_col] = normalize_identifier(pairs[profile_col])
    pairs[account_col] = normalize_identifier(pairs[account_col])
    pairs = pairs.drop_duplicates()
    by_profile = pairs.groupby(profile_col)[account_col].nunique()
    by_account = pairs.groupby(account_col)[profile_col].nunique()
    conflicts = pd.concat(
        [
            pairs[pairs[profile_col].isin(by_profile[by_profile > 1].index)].assign(conflict="profile_to_accounts"),
            pairs[pairs[account_col].isin(by_account[by_account > 1].index)].assign(conflict="account_to_profiles"),
        ],
        ignore_index=True,
    ).drop_duplicates()
    return conflicts, {
        "unique_pairs": int(len(pairs)),
        "profiles": int(pairs[profile_col].nunique()),
        "accounts": int(pairs[account_col].nunique()),
        "profiles_with_multiple_accounts": int((by_profile > 1).sum()),
        "accounts_with_multiple_profiles": int((by_account > 1).sum()),
    }


def canonical_profile_mapping(
    interactions: pd.DataFrame,
    profile_col: str = "profile_id",
    account_col: str = "username",
) -> pd.DataFrame:
    required = {profile_col, account_col}
    missing = sorted(required - set(interactions.columns))
    if missing:
        raise ValueError(f"profile mapping is missing columns: {missing}")
    work = interactions.dropna(subset=[profile_col, account_col]).copy()
    work[profile_col] = normalize_identifier(work[profile_col])
    work[account_col] = normalize_identifier(work[account_col])
    counts = (
        work.groupby([profile_col, account_col], dropna=False)
        .size()
        .rename("mapping_interactions")
        .reset_index()
    )
    counts = counts.sort_values(
        [profile_col, "mapping_interactions", account_col],
        ascending=[True, False, True],
        kind="stable",
    )
    return counts.drop_duplicates(profile_col, keep="first").reset_index(drop=True)


def canonicalize_contents(
    snapshots: Sequence[ContentSnapshot],
    tombstones: pd.DataFrame | None = None,
    require_recommendable: bool = True,
) -> CanonicalResult:
    rows: list[pd.DataFrame] = []
    issues: list[dict[str, object]] = []
    for snapshot in snapshots:
        if not is_raw_source_path(snapshot.source):
            issues.append({"source": snapshot.source, "reason": "derived_or_generated_source"})
            continue
        if "content_id" not in snapshot.frame.columns:
            issues.append({"source": snapshot.source, "reason": "missing_content_id"})
            continue
        frame = snapshot.frame.copy()
        frame["content_id"] = normalize_identifier(frame["content_id"])
        frame["_source"] = snapshot.source
        frame["_source_timestamp"] = snapshot.timestamp
        rows.append(frame)
    if not rows:
        return CanonicalResult(pd.DataFrame(), pd.DataFrame(issues), {"source_rows": 0, "canonical_rows": 0})

    combined = pd.concat(rows, ignore_index=True, sort=False)
    source_rows = len(combined)
    combined = combined[combined["content_id"].notna() & combined["content_id"].ne("")]
    combined = combined.sort_values(["_source_timestamp", "_source"], kind="stable")
    canonical = combined.drop_duplicates("content_id", keep="last")

    deleted_ids: set[str] = set()
    if tombstones is not None and not tombstones.empty:
        if "content_id" not in tombstones.columns:
            raise ValueError("tombstones are missing content_id")
        deletion = tombstones.copy()
        deletion["content_id"] = deletion["content_id"].astype("string")
        deleted_ids = set(deletion["content_id"].dropna().astype(str))
        if "_source_timestamp" in deletion:
            latest_deletion = deletion.groupby("content_id")["_source_timestamp"].max()
            canonical = canonical.merge(
                latest_deletion.rename("_deleted_timestamp"),
                left_on="content_id",
                right_index=True,
                how="left",
            )
            canonical = canonical[
                canonical["_deleted_timestamp"].isna()
                | (canonical["_source_timestamp"] > canonical["_deleted_timestamp"])
            ].drop(columns="_deleted_timestamp")
        else:
            canonical = canonical[~canonical["content_id"].astype(str).isin(deleted_ids)]

    if require_recommendable:
        if "content_status" in canonical:
            canonical = canonical[canonical["content_status"].astype(str).eq("1")]
        if "tag_names" in canonical:
            canonical = canonical[canonical["tag_names"].fillna("").astype(str).str.contains(r"\w", regex=True)]
        if "recommendable" in canonical:
            canonical = canonical[canonical["recommendable"].fillna(False).astype(bool)]

    canonical = canonical.reset_index(drop=True)
    return CanonicalResult(
        frame=canonical,
        quarantine=pd.DataFrame(issues),
        metrics={
            "source_rows": int(source_rows),
            "canonical_rows": int(len(canonical)),
            "duplicate_content_rows_removed": int(source_rows - combined["content_id"].nunique()),
            "tombstone_ids": int(len(deleted_ids)),
            "duplicate_canonical_ids": int(canonical["content_id"].duplicated().sum()),
            "skipped_sources": int(len(issues)),
        },
    )


def canonicalize_content_tags(
    content_tags: pd.DataFrame,
    tags: pd.DataFrame,
) -> pd.DataFrame:
    left = {column.lower(): column for column in content_tags.columns}
    right = {column.lower(): column for column in tags.columns}
    required_left = {"content_id", "tag_id"}
    required_right = {"tag_id", "tag_name"}
    if not required_left.issubset(left) or not required_right.issubset(right):
        raise ValueError("content-tag inputs require content_id, tag_id, and tag_name")
    mapping = content_tags.rename(columns={left["content_id"]: "content_id", left["tag_id"]: "tag_id"})
    definitions = tags.rename(columns={right["tag_id"]: "tag_id", right["tag_name"]: "tag_name"})
    if "status" in right:
        definitions = definitions[definitions[right["status"]].astype(str).eq("1")]
    mapping = mapping[["content_id", "tag_id"]].copy()
    definitions = definitions[["tag_id", "tag_name"]].copy()
    mapping["content_id"] = mapping["content_id"].astype("string")
    mapping["tag_id"] = mapping["tag_id"].astype("string")
    definitions["tag_id"] = definitions["tag_id"].astype("string")
    output = mapping.merge(
        definitions[["tag_id", "tag_name"]].drop_duplicates("tag_id", keep="last"),
        on="tag_id",
        how="inner",
    )
    return output.drop_duplicates(["content_id", "tag_id"]).reset_index(drop=True)


def canonicalize_behavior_events(behavior: pd.DataFrame) -> CanonicalResult:
    aliases = {"user_name": "username", "event_time": "create_date"}
    work = behavior.rename(
        columns={old: new for old, new in aliases.items() if old in behavior and new not in behavior}
    ).copy()
    if "profile_id" not in work or "create_date" not in work:
        raise ValueError("behavior data require profile_id and create_date")
    work["profile_id"] = normalize_identifier(work["profile_id"])
    if "username" in work:
        work["username"] = normalize_username(work["username"])
    work["create_date"] = pd.to_datetime(work["create_date"], errors="coerce")
    invalid = work["profile_id"].isna() | work["create_date"].isna()
    quarantine_columns = [
        column
        for column in ("profile_id", "username", "content_id", "action", "create_date")
        if column in work
    ]
    quarantine = work.loc[invalid, quarantine_columns].copy()
    if not quarantine.empty:
        quarantine["reason"] = "missing_profile_or_event_time"
    keep = [
        column
        for column in ("profile_id", "username", "content_id", "action", "duration", "create_date")
        if column in work
    ]
    canonical = work.loc[~invalid, keep].copy()
    if "content_id" in canonical:
        canonical["content_id"] = normalize_identifier(canonical["content_id"])
    if "duration" in canonical:
        canonical["duration"] = pd.to_numeric(canonical["duration"], errors="coerce").fillna(0.0)
    return CanonicalResult(
        frame=canonical.reset_index(drop=True),
        quarantine=quarantine.reset_index(drop=True),
        metrics={
            "behavior_source_rows": int(len(behavior)),
            "behavior_canonical_rows": int(len(canonical)),
            "behavior_quarantine_rows": int(len(quarantine)),
        },
    )


IMMUTABLE_CONTENT_FIELDS = {
    "content_id",
    "content_name",
    "content_family",
    "type_id",
    "content_duration",
    "content_publish_year",
    "publish_date",
    "content_publish_date",
    "VOD_CODE",
    "product_id",
}

# Attributes that describe what a content *is* rather than how it is being
# merchandised. They can be edited after a cutoff, so they are not immutable,
# but treating them as unknown deletes every content semantic the model has --
# in this dataset the earliest snapshot postdates every window, so the backfill
# path applies to 100% of rows and nulls all of these.
SLOW_MOVING_CONTENT_FIELDS = {
    "tag_names",
    "tag_ids",
    "content_category_name",
    "content_cate_id",
    "content_country",
    "content_director",
    "content_actor",
    "movie_cate_name",
    "vod_type",
    "content_single",
}

# Deliberately NOT slow-moving: total_episode grows while a series airs, and
# content_hot is a merchandising flag that turns over weekly. Both would carry
# post-cutoff information backwards, and neither carries measurable signal.
VOLATILE_CONTENT_FIELDS = {
    "total_episode",
    "content_hot",
    "content_status",
    "locked_level",
    "block_platform",
}


def canonicalize_content_versions(
    snapshots: Sequence[ContentSnapshot],
) -> pd.DataFrame:
    """Keep deduplicated source versions so temporal joins can select metadata as-of."""
    versions: list[pd.DataFrame] = []
    for snapshot in snapshots:
        if not is_raw_source_path(snapshot.source) or "content_id" not in snapshot.frame:
            continue
        frame = snapshot.frame.copy()
        frame["content_id"] = normalize_identifier(frame["content_id"])
        frame["_source"] = snapshot.source
        frame["_source_timestamp"] = pd.Timestamp(snapshot.timestamp)
        versions.append(frame)
    if not versions:
        return pd.DataFrame(columns=["content_id", "_source", "_source_timestamp"])
    output = pd.concat(versions, ignore_index=True, sort=False)
    output = output[
        output["content_id"].notna() & output["content_id"].astype(str).ne("")
    ]
    output = output.sort_values(
        ["content_id", "_source_timestamp", "_source"], kind="stable"
    ).drop_duplicates(["content_id", "_source_timestamp"], keep="last")
    output["_valid_from"] = output["_source_timestamp"]
    output["_valid_to"] = output.groupby("content_id")["_source_timestamp"].shift(-1)
    return output.reset_index(drop=True)


def select_content_metadata_as_of(
    versions: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    immutable_fields: set[str] | None = None,
    include_slow_moving: bool = True,
) -> pd.DataFrame:
    """Select latest metadata at cutoff and mark immutable-only future backfill.

    ``include_slow_moving`` keeps the descriptive attributes in
    ``SLOW_MOVING_CONTENT_FIELDS`` on backfilled rows. Without it the model
    receives no content semantics at all whenever the snapshots postdate the
    cutoffs, which is the case for the whole of this dataset. Volatile
    merchandising fields stay nulled either way.
    """
    required = {"content_id", "_source_timestamp"}
    if not required.issubset(versions):
        raise ValueError("content versions require content_id and _source_timestamp")
    cutoff_value = pd.Timestamp(cutoff)
    work = versions.copy()
    work["_source_timestamp"] = pd.to_datetime(
        work["_source_timestamp"], errors="coerce"
    )
    prior = (
        work[work["_source_timestamp"].le(cutoff_value)]
        .sort_values(["content_id", "_source_timestamp", "_source"], kind="stable")
        .drop_duplicates("content_id", keep="last")
        .copy()
    )
    prior["metadata_backfill"] = False
    represented = set(prior["content_id"].astype(str))

    remaining = work[~work["content_id"].astype(str).isin(represented)].copy()
    publish_column = next(
        (
            column
            for column in ("content_publish_date", "publish_date")
            if column in remaining
        ),
        None,
    )
    if publish_column is None or remaining.empty:
        return prior.reset_index(drop=True)
    remaining["_publish_date"] = pd.to_datetime(
        remaining[publish_column], errors="coerce"
    )
    backfill = (
        remaining[remaining["_publish_date"].le(cutoff_value)]
        .sort_values(["content_id", "_source_timestamp", "_source"], kind="stable")
        .drop_duplicates("content_id", keep="first")
        .copy()
    )
    allowed = set(immutable_fields or IMMUTABLE_CONTENT_FIELDS)
    if include_slow_moving:
        allowed |= SLOW_MOVING_CONTENT_FIELDS
    allowed -= VOLATILE_CONTENT_FIELDS
    provenance = {
        "_source",
        "_source_timestamp",
        "_valid_from",
        "_valid_to",
        "_publish_date",
    }
    for column in backfill.columns:
        if column not in allowed | provenance:
            backfill[column] = pd.NA
    backfill["metadata_backfill"] = True
    return pd.concat([prior, backfill], ignore_index=True, sort=False)


def write_partitioned_interactions(
    interactions: pd.DataFrame,
    output_dir: str | Path,
    time_col: str = "create_date",
) -> list[Path]:
    """Write deterministic event-date parquet partitions for bounded temporal reads."""
    if time_col not in interactions:
        raise ValueError(f"interactions are missing {time_col}")
    work = interactions.copy()
    work["_partition_date"] = pd.to_datetime(
        work[time_col], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    work = work[work["_partition_date"].notna()]
    root = Path(output_dir)
    paths: list[Path] = []
    for date, rows in work.groupby("_partition_date", sort=True):
        directory = root / f"event_date={date}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "part-00000.parquet"
        rows.drop(columns="_partition_date").to_parquet(path, index=False)
        paths.append(path)
    return paths


def read_partitioned_interactions(
    output_dir: str | Path,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Read only date partitions intersecting the requested closed interval."""
    start_date = pd.Timestamp(start).date() if start is not None else None
    end_date = pd.Timestamp(end).date() if end is not None else None
    paths: list[Path] = []
    for path in sorted(Path(output_dir).glob("event_date=*/part-*.parquet")):
        date = pd.Timestamp(path.parent.name.split("=", 1)[1]).date()
        if start_date is not None and date < start_date:
            continue
        if end_date is not None and date > end_date:
            continue
        paths.append(path)
    return (
        pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
        if paths
        else pd.DataFrame()
    )


def historical_behavior_features(
    behavior: pd.DataFrame,
    cutoff: str | pd.Timestamp,
) -> pd.DataFrame:
    canonical = canonicalize_behavior_events(behavior).frame
    canonical = canonical[canonical["create_date"] <= pd.Timestamp(cutoff)].copy()
    if canonical.empty:
        return pd.DataFrame(
            columns=[
                "profile_id",
                "historical_behavior_count",
                "historical_behavior_duration",
                "historical_behavior_recency_days",
            ]
        )
    if "duration" not in canonical:
        canonical["duration"] = 0.0
    grouped = canonical.groupby("profile_id", dropna=False).agg(
        historical_behavior_count=("create_date", "size"),
        historical_behavior_duration=("duration", "sum"),
        _last_behavior=("create_date", "max"),
    ).reset_index()
    grouped["historical_behavior_recency_days"] = (
        pd.Timestamp(cutoff) - grouped.pop("_last_behavior")
    ).dt.total_seconds().div(86400).clip(lower=0)
    return grouped


def historical_behavior_features_from_parts(
    paths: Sequence[str | Path],
    cutoff: str | pd.Timestamp,
    profile_ids: set[str] | None = None,
) -> pd.DataFrame:
    partials: list[pd.DataFrame] = []
    cutoff_time = pd.Timestamp(cutoff)
    for path in paths:
        frame = pd.read_parquet(
            path,
            columns=["profile_id", "duration", "create_date"],
        )
        frame["profile_id"] = frame["profile_id"].astype("string")
        frame["create_date"] = pd.to_datetime(frame["create_date"], errors="coerce")
        frame = frame[frame["create_date"].notna() & (frame["create_date"] <= cutoff_time)]
        if profile_ids is not None:
            frame = frame[frame["profile_id"].astype(str).isin(profile_ids)]
        if frame.empty:
            continue
        frame["duration"] = pd.to_numeric(frame["duration"], errors="coerce").fillna(0.0)
        partials.append(
            frame.groupby("profile_id", dropna=False).agg(
                historical_behavior_count=("create_date", "size"),
                historical_behavior_duration=("duration", "sum"),
                _last_behavior=("create_date", "max"),
            ).reset_index()
        )
    if not partials:
        return pd.DataFrame(
            columns=[
                "profile_id",
                "historical_behavior_count",
                "historical_behavior_duration",
                "historical_behavior_recency_days",
            ]
        )
    combined = pd.concat(partials, ignore_index=True)
    grouped = combined.groupby("profile_id", dropna=False).agg(
        historical_behavior_count=("historical_behavior_count", "sum"),
        historical_behavior_duration=("historical_behavior_duration", "sum"),
        _last_behavior=("_last_behavior", "max"),
    ).reset_index()
    grouped["historical_behavior_recency_days"] = (
        cutoff_time - grouped.pop("_last_behavior")
    ).dt.total_seconds().div(86400).clip(lower=0)
    return grouped


def cache_fingerprint(source_fingerprint: str, config: dict, schema_version: int = 1) -> str:
    return stable_hash({"source": source_fingerprint, "config": config, "schema_version": schema_version})


def cache_is_current(metadata_path: str | Path, expected_fingerprint: str) -> bool:
    path = Path(metadata_path)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("fingerprint") == expected_fingerprint


def write_canonical_result(
    result: CanonicalResult,
    output_path: str | Path,
    fingerprint: str,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.frame.to_parquet(output, index=False)
    result.quarantine.to_parquet(output.with_name(output.stem + "_quarantine.parquet"), index=False)
    metadata = {"fingerprint": fingerprint, "metrics": result.metrics}
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output
