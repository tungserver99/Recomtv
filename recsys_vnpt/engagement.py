from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


LABEL_CONTRACT_SCHEMA_VERSION = 4
DEFAULT_TARGET_VERSION = "engagement_v1"
DEFAULT_SESSION_GAP_MINUTES = 30
GAIN_COLUMNS = ("completion_gain", "watch_time_gain")
TARGET_COLUMNS = {
    "label",
    "completion_gain",
    "watch_time_gain",
    "target_semantics",
    "sample_weight",
    "sample_type",
    "episode_count",
    "total_watch_seconds",
    "max_episode_watch_seconds",
}


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _field_key(family: Any, type_id: Any) -> str:
    normalized_family = (
        "unknown" if pd.isna(family) or str(family).strip() == "" else str(family)
    )
    normalized_type = (
        "unknown" if pd.isna(type_id) or str(type_id).strip() == "" else str(type_id)
    )
    return f"{normalized_family.lower()}|{normalized_type}"


def _column_or_default(
    frame: pd.DataFrame,
    column: str,
    default: Any,
) -> pd.Series:
    if column in frame:
        return frame[column].fillna(default)
    return pd.Series(default, index=frame.index)


@dataclass(frozen=True)
class LabelContract:
    domain: str
    fit_cutoff: str
    session_gap_minutes: int = DEFAULT_SESSION_GAP_MINUTES
    target_version: str = DEFAULT_TARGET_VERSION
    duration_groups: dict[str, dict[str, float | int]] = field(default_factory=dict)
    global_duration_lower_seconds: float = 1.0
    global_duration_upper_seconds: float = 86400.0
    global_watch_cap_seconds: float = 3600.0
    schema_version: int = LABEL_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LABEL_CONTRACT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported label contract schema: {self.schema_version}"
            )
        if self.target_version != DEFAULT_TARGET_VERSION:
            raise ValueError(f"unsupported target version: {self.target_version}")
        if self.session_gap_minutes < 1:
            raise ValueError("session_gap_minutes must be positive")
        if self.global_duration_lower_seconds <= 0:
            raise ValueError("duration lower bound must be positive")
        if self.global_duration_upper_seconds < self.global_duration_lower_seconds:
            raise ValueError("duration bounds are invalid")
        if self.global_watch_cap_seconds <= 0:
            raise ValueError("watch cap must be positive")

    @property
    def artifact_hash(self) -> str:
        return _stable_hash(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifact_hash"] = self.artifact_hash
        payload["target_semantics"] = {
            "completion_gain": "max playback episode watch divided by atomic content duration, clipped to [0, 1]",
            "watch_time_gain": "log-normalized total playback watch capped by a training-only p95",
            "unobserved": "contrastive sample, not a verified negative",
        }
        return payload

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "LabelContract":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        expected_hash = payload.pop("artifact_hash", None)
        payload.pop("target_semantics", None)
        allowed = set(cls.__dataclass_fields__)
        contract = cls(**{key: value for key, value in payload.items() if key in allowed})
        if expected_hash is not None and expected_hash != contract.artifact_hash:
            raise ValueError("label contract artifact hash mismatch")
        return contract

    def group_values(self, family: Any, type_id: Any) -> dict[str, float | int]:
        return self.duration_groups.get(
            _field_key(family, type_id),
            {
                "duration_lower_seconds": self.global_duration_lower_seconds,
                "duration_upper_seconds": self.global_duration_upper_seconds,
                "watch_cap_seconds": self.global_watch_cap_seconds,
                "content_rows": 0,
                "playback_rows": 0,
            },
        )


def _normalized_uuid(values: pd.Series) -> pd.Series:
    output = values.astype("string").str.strip()
    return output.mask(output.isna() | output.isin(["", "0", "nan", "None"]))


def _row_hashes(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    selected = frame[list(columns)].copy()
    for column in selected:
        if pd.api.types.is_datetime64_any_dtype(selected[column]):
            selected[column] = selected[column].astype("int64")
        else:
            selected[column] = selected[column].astype("string").fillna("<NA>")
    values = pd.util.hash_pandas_object(selected, index=False, categorize=True)
    return values.map(lambda value: f"{int(value):016x}").astype("string")


def _union_interval_seconds(starts: np.ndarray, ends: np.ndarray) -> float:
    valid = (starts != np.iinfo(np.int64).min) & (ends != np.iinfo(np.int64).min)
    pairs = sorted(
        (int(start), int(end))
        for start, end in zip(starts[valid], ends[valid])
        if end > start
    )
    if not pairs:
        return 0.0
    total = 0
    active_start, active_end = pairs[0]
    for start, end in pairs[1:]:
        if start <= active_end:
            active_end = max(active_end, end)
        else:
            total += active_end - active_start
            active_start, active_end = start, end
    total += active_end - active_start
    return float(total / 1_000_000_000)


def build_playback_events(
    interactions: pd.DataFrame,
    session_gap_minutes: int = DEFAULT_SESSION_GAP_MINUTES,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Deduplicate raw duration rows and build deterministic playback episodes."""

    required = {"profile_id", "content_id", "create_date", "duration"}
    missing = sorted(required - set(interactions))
    if missing:
        raise ValueError(f"playback events are missing columns: {missing}")
    if session_gap_minutes < 1:
        raise ValueError("session_gap_minutes must be positive")

    work = interactions.copy()
    raw_rows = len(work)
    work["profile_id"] = work["profile_id"].astype("string").str.strip()
    work["content_id"] = work["content_id"].astype("string").str.strip()
    # Interval arithmetic below reads timestamps as int64 nanoseconds; parquet
    # sources arrive in microseconds, so fix the unit before anything else.
    work["event_end"] = pd.to_datetime(work["create_date"], errors="coerce").dt.as_unit("ns")
    work["duration"] = pd.to_numeric(work["duration"], errors="coerce")
    work["uuid"] = _normalized_uuid(
        work.get("uuid", pd.Series(pd.NA, index=work.index, dtype="string"))
    )
    invalid = (
        work["profile_id"].isna()
        | work["profile_id"].eq("")
        | work["content_id"].isna()
        | work["content_id"].eq("")
        | work["event_end"].isna()
        | work["duration"].isna()
        | work["duration"].le(0)
    )
    invalid_rows = int(invalid.sum())
    work = work.loc[~invalid].copy()

    parsed_start = pd.to_datetime(
        work.get("date_time", pd.Series(pd.NaT, index=work.index)),
        errors="coerce",
    ).dt.as_unit("ns")
    derived_start = work["event_end"] - pd.to_timedelta(work["duration"], unit="s")
    valid_start = parsed_start.notna() & parsed_start.le(work["event_end"])
    work["event_start"] = parsed_start.where(valid_start, derived_start)
    work["session_base"] = work["uuid"].fillna("__NO_UUID__")

    exact_columns = [
        "profile_id",
        "content_id",
        "event_start",
        "event_end",
        "duration",
        "session_base",
    ]
    work["source_event_hash"] = _row_hashes(work, exact_columns)
    exact_duplicate_rows = int(work.duplicated(exact_columns, keep="first").sum())
    work = work.drop_duplicates(exact_columns, keep="first")

    end_key = ["profile_id", "content_id", "session_base", "event_end"]
    work = work.sort_values(
        end_key + ["duration", "source_event_hash"],
        ascending=[True, True, True, True, False, True],
        kind="stable",
    )
    same_end_duplicate_rows = int(work.duplicated(end_key, keep="first").sum())
    work = work.drop_duplicates(end_key, keep="first")

    work = work.sort_values(
        ["profile_id", "session_base", "event_end", "content_id"],
        kind="stable",
    ).reset_index(drop=True)
    grouped_session = work.groupby(
        ["profile_id", "session_base"], sort=False, dropna=False
    )
    gaps = grouped_session["event_end"].diff().dt.total_seconds().div(60)
    new_session = gaps.isna() | gaps.gt(session_gap_minutes)
    work["session_sequence"] = (
        new_session.groupby([work["profile_id"], work["session_base"]])
        .cumsum()
        .astype("int64")
    )
    session_starts = work.groupby(
        ["profile_id", "session_base", "session_sequence"], sort=False
    )["event_end"].transform("min")
    session_tokens = (
        work["profile_id"].astype(str)
        + "|"
        + work["session_base"].astype(str)
        + "|"
        + work["session_sequence"].astype(str)
        + "|"
        + session_starts.astype("int64").astype(str)
    )
    work["session_id"] = session_tokens.map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    )

    work = work.sort_values(
        ["profile_id", "session_id", "event_end", "content_id"], kind="stable"
    ).reset_index(drop=True)
    previous_content = work.groupby("session_id", sort=False)["content_id"].shift()
    episode_boundary = previous_content.isna() | previous_content.ne(work["content_id"])
    work["episode_sequence"] = episode_boundary.groupby(work["session_id"]).cumsum()

    optional = [
        column
        for column in (
            "uuid",
            "username",
            "member_id",
            "request_id",
            "recommendation",
            "refer_path",
        )
        if column in work
    ]
    rows: list[dict[str, Any]] = []
    group_columns = ["session_id", "episode_sequence"]
    for (_, episode_sequence), group in work.groupby(group_columns, sort=False):
        starts = group["event_start"].astype("int64").to_numpy()
        ends = group["event_end"].astype("int64").to_numpy()
        watch_seconds = _union_interval_seconds(starts, ends)
        if watch_seconds <= 0:
            continue
        first = group.iloc[0]
        row: dict[str, Any] = {
            "profile_id": str(first["profile_id"]),
            "content_id": str(first["content_id"]),
            "session_id": str(first["session_id"]),
            "episode_index": int(episode_sequence),
            "event_start": group["event_start"].min(),
            "event_end": group["event_end"].max(),
            "watch_seconds": watch_seconds,
            "source_event_count": int(len(group)),
        }
        for column in optional:
            values = group[column].dropna()
            row[column] = values.iloc[0] if len(values) else pd.NA
        rows.append(row)

    event_columns = [
        "profile_id",
        "content_id",
        "session_id",
        "episode_index",
        "event_start",
        "event_end",
        "watch_seconds",
        "source_event_count",
        *optional,
    ]
    events = pd.DataFrame(rows, columns=event_columns)
    if not events.empty:
        events = events.sort_values(
            ["profile_id", "event_end", "session_id", "episode_index"],
            kind="stable",
        ).reset_index(drop=True)
        events["event_hash"] = _row_hashes(
            events,
            [
                "profile_id",
                "content_id",
                "session_id",
                "episode_index",
                "event_start",
                "event_end",
                "watch_seconds",
            ],
        )
        session_rows: list[dict[str, Any]] = []
        for session_id, group in events.groupby("session_id", sort=False):
            sequence = group.sort_values(
                ["event_end", "episode_index"], kind="stable"
            )["content_id"].astype(str).tolist()
            session_rows.append(
                {
                    "session_id": str(session_id),
                    "profile_id": str(group["profile_id"].iloc[0]),
                    "session_start": group["event_start"].min(),
                    "session_end": group["event_end"].max(),
                    "playback_count": int(len(group)),
                    "distinct_contents": int(group["content_id"].nunique()),
                    "content_sequence": json.dumps(sequence, separators=(",", ":")),
                }
            )
        sessions = pd.DataFrame(session_rows)
    else:
        events["event_hash"] = pd.Series(dtype="string")
        sessions = pd.DataFrame(
            columns=[
                "session_id",
                "profile_id",
                "session_start",
                "session_end",
                "playback_count",
                "distinct_contents",
                "content_sequence",
            ]
        )

    uuid_coverage = (
        float(work["uuid"].notna().mean()) if len(work) else 0.0
    )
    session_depth = sessions["distinct_contents"] if len(sessions) else pd.Series(dtype=float)
    metrics = {
        "target_version": DEFAULT_TARGET_VERSION,
        "session_gap_minutes": int(session_gap_minutes),
        "raw_interaction_rows": int(raw_rows),
        "invalid_playback_rows": invalid_rows,
        "exact_duplicate_rows_removed": exact_duplicate_rows,
        "same_end_duplicate_rows_removed": same_end_duplicate_rows,
        "playback_events": int(len(events)),
        "playback_sessions": int(len(sessions)),
        "uuid_coverage_rate": uuid_coverage,
        "multi_content_session_rate": float(session_depth.ge(2).mean())
        if len(session_depth)
        else 0.0,
        "duplicate_event_hashes": int(events["event_hash"].duplicated().sum())
        if len(events)
        else 0,
        "duplicate_leakage": 0,
    }
    return events, sessions, metrics


def _raw_atomic_duration(items: pd.DataFrame) -> pd.Series:
    duration = pd.to_numeric(items.get("content_duration"), errors="coerce")
    episodes = pd.to_numeric(
        items.get("total_episode", pd.Series(1.0, index=items.index)),
        errors="coerce",
    )
    divisor = episodes.where(episodes.gt(1), 1.0)
    return duration.div(divisor)


def fit_label_contract(
    items: pd.DataFrame,
    playback_events: pd.DataFrame,
    domain: str,
    fit_cutoff: str | pd.Timestamp,
    session_gap_minutes: int = DEFAULT_SESSION_GAP_MINUTES,
) -> LabelContract:
    required_items = {"content_id", "content_duration"}
    if not required_items.issubset(items):
        raise ValueError(
            f"label contract items are missing {sorted(required_items - set(items))}"
        )
    metadata = items.copy()
    metadata["content_family"] = _column_or_default(
        metadata, "content_family", domain
    )
    metadata["type_id"] = _column_or_default(metadata, "type_id", "unknown")
    metadata["raw_atomic_duration"] = _raw_atomic_duration(metadata)
    valid_duration = metadata["raw_atomic_duration"].replace([np.inf, -np.inf], np.nan).dropna()
    valid_duration = valid_duration[valid_duration.gt(0)]
    global_lower = float(valid_duration.quantile(0.01)) if len(valid_duration) else 1.0
    global_upper = float(valid_duration.quantile(0.99)) if len(valid_duration) else 86400.0
    global_lower = max(global_lower, 1.0)
    global_upper = max(global_upper, global_lower)

    event_metadata = playback_events.copy()
    if not event_metadata.empty:
        event_metadata["event_end"] = pd.to_datetime(
            event_metadata["event_end"], errors="coerce"
        )
        event_metadata = event_metadata[
            event_metadata["event_end"].le(pd.Timestamp(fit_cutoff))
        ].merge(
            metadata[["content_id", "content_family", "type_id"]],
            on="content_id",
            how="left",
        )
    valid_watch = pd.to_numeric(
        event_metadata.get("watch_seconds", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    valid_watch = valid_watch[valid_watch.gt(0)]
    global_watch_cap = float(valid_watch.quantile(0.95)) if len(valid_watch) else 3600.0
    global_watch_cap = max(global_watch_cap, 1.0)

    duration_groups: dict[str, dict[str, float | int]] = {}
    for (family, type_id), group in metadata.groupby(
        ["content_family", "type_id"], dropna=False, sort=True
    ):
        values = pd.to_numeric(group["raw_atomic_duration"], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        values = values[values.gt(0)]
        lower = float(values.quantile(0.01)) if len(values) >= 5 else global_lower
        upper = float(values.quantile(0.99)) if len(values) >= 5 else global_upper
        lower = max(lower, 1.0)
        upper = max(upper, lower)
        key = _field_key(family, type_id)
        event_group = event_metadata[
            event_metadata.get("content_family", pd.Series(dtype=str)).astype(str).eq(str(family))
            & event_metadata.get("type_id", pd.Series(dtype=str)).astype(str).eq(str(type_id))
        ] if not event_metadata.empty else pd.DataFrame()
        watches = pd.to_numeric(
            event_group.get("watch_seconds", pd.Series(dtype=float)), errors="coerce"
        ).dropna()
        watches = watches[watches.gt(0)]
        watch_cap = float(watches.quantile(0.95)) if len(watches) >= 5 else global_watch_cap
        duration_groups[key] = {
            "duration_lower_seconds": lower,
            "duration_upper_seconds": upper,
            "watch_cap_seconds": max(watch_cap, 1.0),
            "content_rows": int(len(group)),
            "playback_rows": int(len(event_group)),
        }

    return LabelContract(
        domain=domain,
        fit_cutoff=str(pd.Timestamp(fit_cutoff)),
        session_gap_minutes=session_gap_minutes,
        duration_groups=duration_groups,
        global_duration_lower_seconds=global_lower,
        global_duration_upper_seconds=global_upper,
        global_watch_cap_seconds=global_watch_cap,
    )


def apply_atomic_duration(
    items: pd.DataFrame,
    contract: LabelContract,
) -> pd.DataFrame:
    output = items.copy()
    output["content_family"] = _column_or_default(
        output, "content_family", contract.domain
    )
    output["type_id"] = _column_or_default(output, "type_id", "unknown")
    output["raw_atomic_duration_seconds"] = _raw_atomic_duration(output)
    atomic: list[float] = []
    fallback: list[bool] = []
    watch_caps: list[float] = []
    for row in output.itertuples(index=False):
        group = contract.group_values(
            getattr(row, "content_family", contract.domain),
            getattr(row, "type_id", "unknown"),
        )
        lower = float(group["duration_lower_seconds"])
        upper = float(group["duration_upper_seconds"])
        raw = float(getattr(row, "raw_atomic_duration_seconds"))
        invalid = not np.isfinite(raw) or raw <= 0
        value = lower if invalid else float(np.clip(raw, lower, upper))
        atomic.append(value)
        fallback.append(bool(invalid or value != raw))
        watch_caps.append(float(group["watch_cap_seconds"]))
    output["atomic_duration_seconds"] = atomic
    output["atomic_duration_fallback"] = fallback
    output["watch_cap_seconds"] = watch_caps
    return output


def label_engagement_window(
    playback_events: pd.DataFrame,
    items: pd.DataFrame,
    contract: LabelContract,
    label_start: str | pd.Timestamp,
    label_end: str | pd.Timestamp,
) -> pd.DataFrame:
    required = {"profile_id", "content_id", "event_end", "watch_seconds"}
    missing = sorted(required - set(playback_events))
    if missing:
        raise ValueError(f"engagement labeling is missing columns: {missing}")
    work = playback_events.copy()
    work["event_end"] = pd.to_datetime(work["event_end"], errors="coerce")
    work = work[
        work["event_end"].between(pd.Timestamp(label_start), pd.Timestamp(label_end))
    ]
    if work.empty:
        return pd.DataFrame(
            columns=[
                "profile_id",
                "content_id",
                "label",
                "completion_gain",
                "watch_time_gain",
                "sample_type",
                "sample_weight",
                "target_semantics",
            ]
        )
    metadata = apply_atomic_duration(items, contract)
    metadata_columns = [
        "content_id",
        "content_family",
        "type_id",
        "atomic_duration_seconds",
        "atomic_duration_fallback",
        "watch_cap_seconds",
    ]
    work = work.merge(
        metadata[metadata_columns].drop_duplicates("content_id", keep="last"),
        on="content_id",
        how="inner",
    )
    grouped = work.groupby(["profile_id", "content_id"], dropna=False).agg(
        episode_count=("watch_seconds", "size"),
        total_watch_seconds=("watch_seconds", "sum"),
        max_episode_watch_seconds=("watch_seconds", "max"),
        content_family=("content_family", "first"),
        type_id=("type_id", "first"),
        atomic_duration_seconds=("atomic_duration_seconds", "first"),
        atomic_duration_fallback=("atomic_duration_fallback", "first"),
        watch_cap_seconds=("watch_cap_seconds", "first"),
    ).reset_index()
    grouped["completion_gain"] = (
        grouped["max_episode_watch_seconds"]
        .div(grouped["atomic_duration_seconds"])
        .clip(lower=0.0, upper=1.0)
    )
    capped_watch = np.minimum(
        grouped["total_watch_seconds"].to_numpy(dtype=float),
        grouped["watch_cap_seconds"].to_numpy(dtype=float),
    )
    grouped["watch_time_gain"] = np.log1p(capped_watch) / np.log1p(
        grouped["watch_cap_seconds"].to_numpy(dtype=float)
    )
    grouped["watch_time_gain"] = grouped["watch_time_gain"].clip(0.0, 1.0)
    grouped["label"] = 1
    grouped["sample_type"] = "observed_engagement"
    grouped["sample_weight"] = 1.0
    grouped["target_semantics"] = contract.target_version
    return grouped


def historical_sequence_features(
    playback_events: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    maximum_sequence_length: int = 50,
) -> pd.DataFrame:
    if maximum_sequence_length < 1:
        raise ValueError("maximum_sequence_length must be positive")
    required = {"profile_id", "content_id", "session_id", "event_end"}
    if not required.issubset(playback_events):
        raise ValueError(
            f"sequence features are missing {sorted(required - set(playback_events))}"
        )
    work = playback_events.copy()
    work["event_end"] = pd.to_datetime(work["event_end"], errors="coerce")
    work = work[work["event_end"].le(pd.Timestamp(cutoff))]
    if work.empty:
        return pd.DataFrame(
            columns=[
                "profile_id",
                "historical_session_count",
                "historical_playback_count",
                "historical_sequence_depth",
                "historical_content_sequence",
            ]
        )
    rows: list[dict[str, Any]] = []
    for profile_id, group in work.groupby("profile_id", sort=True):
        ordered = group.sort_values(["event_end", "content_id"], kind="stable")
        sequence = ordered["content_id"].astype(str).tolist()[-maximum_sequence_length:]
        rows.append(
            {
                "profile_id": str(profile_id),
                "historical_session_count": int(ordered["session_id"].nunique()),
                "historical_playback_count": int(len(ordered)),
                "historical_sequence_depth": int(len(sequence)),
                "historical_content_sequence": ",".join(sequence),
            }
        )
    return pd.DataFrame(rows)
