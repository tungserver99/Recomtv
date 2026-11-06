from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from recsys_vnpt.pipeline_v2 import _evaluation_seen_entities
from recsys_vnpt.retrieval import pair_keys_hash, retrieval_truth_attribution
from recsys_vnpt.rule_assembly import SPECIAL_RULES
from recsys_vnpt.source_manifest import sha256_file


def _available_columns(path: Path, requested: tuple[str, ...]) -> list[str]:
    available = set(pq.ParquetFile(path).schema_arrow.names)
    return [column for column in requested if column in available]


def _truth_attributes(
    temporal: Path,
    split: str,
    truth: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    seen_profiles, seen_contents, seen_metrics = _evaluation_seen_entities(
        temporal, split
    )
    split_path = temporal / f"{split}.parquet"
    columns = _available_columns(
        split_path,
        (
            "profile_id",
            "content_id",
            "sample_type",
            "historical_interaction_count",
            "content_family",
            "type_id",
        ),
    )
    observed = pd.read_parquet(split_path, columns=columns)
    if "sample_type" in observed:
        observed = observed[
            observed["sample_type"].astype(str).eq("observed_engagement")
        ]
    observed["profile_id"] = observed["profile_id"].astype(str)
    observed["content_id"] = observed["content_id"].astype(str)
    observed = observed.drop(columns=["sample_type"], errors="ignore").drop_duplicates(
        ["profile_id", "content_id"], keep="last"
    )

    attributes = truth[["profile_id", "content_id"]].copy()
    attributes["profile_id"] = attributes["profile_id"].astype(str)
    attributes["content_id"] = attributes["content_id"].astype(str)
    attributes = attributes.merge(
        observed, on=["profile_id", "content_id"], how="left", validate="one_to_one"
    )
    attributes["observed_history_profile_oov"] = ~attributes["profile_id"].isin(
        seen_profiles
    )
    attributes["observed_history_content_oov"] = ~attributes["content_id"].isin(
        seen_contents
    )
    history = pd.to_numeric(
        attributes.get(
            "historical_interaction_count",
            pd.Series(0, index=attributes.index),
        ),
        errors="coerce",
    ).fillna(0)
    attributes["activity_bucket"] = pd.cut(
        history,
        bins=[-1, 0, 5, 20, float("inf")],
        labels=["cold", "low", "medium", "high"],
    ).astype("string")
    attributes = attributes.drop(
        columns=["historical_interaction_count"], errors="ignore"
    )
    return attributes, seen_metrics


def _observed_history_pairs(temporal: Path, split: str) -> pd.DataFrame:
    window_names = ["train_1", "train_2", "train_3"]
    if split == "test":
        window_names.append("validation")
    frames: list[pd.DataFrame] = []
    for window_name in window_names:
        path = temporal / f"{window_name}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(
            path, columns=["profile_id", "content_id", "sample_type"]
        )
        frame = frame[frame["sample_type"].astype(str).eq("observed_engagement")]
        frames.append(frame[["profile_id", "content_id"]])
    if not frames:
        return pd.DataFrame(columns=["profile_id", "content_id"])
    observed = pd.concat(frames, ignore_index=True)
    observed[["profile_id", "content_id"]] = observed[
        ["profile_id", "content_id"]
    ].astype(str)
    return observed.drop_duplicates(["profile_id", "content_id"])


def _add_content_diagnostic_attributes(
    attributes: pd.DataFrame,
    observed_history: pd.DataFrame,
    contents: pd.DataFrame,
    cutoff: str | pd.Timestamp,
) -> pd.DataFrame:
    output = attributes.copy()
    popularity = (
        observed_history.groupby("content_id", observed=True)["profile_id"]
        .nunique()
        .rename("historical_profile_count")
    )
    output = output.merge(
        popularity, left_on="content_id", right_index=True, how="left"
    )
    output["historical_profile_count"] = pd.to_numeric(
        output["historical_profile_count"], errors="coerce"
    ).fillna(0)
    output["historical_popularity_bucket"] = pd.cut(
        output["historical_profile_count"],
        bins=[-1, 0, 20, 100, float("inf")],
        labels=["0", "1-20", "21-100", "101+"],
    ).astype("string")

    publish_column = next(
        (
            column
            for column in ("publish_date", "_publish_date", "start_time")
            if column in contents
        ),
        None,
    )
    if publish_column is None:
        output["latest_publish_age_bucket"] = "missing"
        return output
    catalog = contents[["content_id", publish_column]].copy()
    catalog["content_id"] = catalog["content_id"].astype(str)
    catalog = catalog.drop_duplicates("content_id", keep="last")
    catalog["_publish_time"] = pd.to_datetime(
        catalog[publish_column], errors="coerce"
    )
    output = output.merge(
        catalog[["content_id", "_publish_time"]],
        on="content_id",
        how="left",
        validate="many_to_one",
    )
    age_days = (
        pd.Timestamp(cutoff) - output["_publish_time"]
    ).dt.total_seconds() / 86400.0
    output["latest_publish_age_bucket"] = np.select(
        [
            age_days.isna(),
            age_days < 0,
            age_days <= 7,
            age_days <= 30,
            age_days <= 90,
            age_days <= 365,
        ],
        [
            "missing",
            "latest_after_cutoff",
            "0-7d",
            "8-30d",
            "31-90d",
            "91-365d",
        ],
        default="366d+",
    )
    return output.drop(columns=["_publish_time"])


def _rule_attribution(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    membership: pd.DataFrame,
) -> dict[str, dict[str, float | int]]:
    keys = ["profile_id", "content_id"]
    normalized_truth = truth[
        [*keys, "completion_gain", "watch_time_gain"]
    ].copy()
    normalized_truth[keys] = normalized_truth[keys].astype(str)
    hits = normalized_truth[keys].merge(
        candidates[keys].drop_duplicates(), on=keys, how="inner"
    )
    hits["_hit"] = True
    normalized_truth = normalized_truth.merge(hits, on=keys, how="left")
    normalized_truth["_hit"] = normalized_truth["_hit"].eq(True)

    rules = membership[["content_id", "rule_id"]].drop_duplicates().astype(str)
    rules = rules[~rules["rule_id"].isin(SPECIAL_RULES)]
    expanded = normalized_truth.merge(rules, on="content_id", how="inner")
    membership_counts = expanded.groupby(keys, observed=True)["rule_id"].transform(
        "count"
    )
    expanded["_fraction"] = 1.0 / membership_counts.clip(lower=1)
    output: dict[str, dict[str, float | int]] = {}
    for rule_id, group in expanded.groupby(
        "rule_id", observed=True, sort=True
    ):
        hits = group[group["_hit"]]
        fractional_truth = float(group["_fraction"].sum())
        fractional_hits = float(hits["_fraction"].sum())
        row: dict[str, float | int] = {
            "truth_pair_incidences": int(len(group)),
            "truth_hit_incidences": int(len(hits)),
            "candidate_recall": len(hits) / max(len(group), 1),
            "fractional_truth_mass": fractional_truth,
            "fractional_hit_mass": fractional_hits,
            "fractional_miss_mass": fractional_truth - fractional_hits,
        }
        for gain_column in ("completion_gain", "watch_time_gain"):
            total = float((group[gain_column] * group["_fraction"]).sum())
            hit = float((hits[gain_column] * hits["_fraction"]).sum())
            row[f"fractional_{gain_column}_sum"] = total
            row[f"fractional_{gain_column}_hit_sum"] = hit
            row[f"fractional_{gain_column}_recall"] = (
                hit / total if total > 0 else 0.0
            )
        output[str(rule_id)] = row
    return output


def render_markdown(payload: dict[str, Any]) -> str:
    attribution = payload["attribution"]
    lines = [
        f"# Retrieval Attribution: {payload['source_run']}",
        "",
        f"- Split: `{payload['evaluation_split']}`",
        f"- Truth pairs: {attribution['truth_pairs']}",
        f"- Final-pool hits: {attribution['truth_hits']}",
        f"- Candidate Recall: {attribution['candidate_recall']:.6f}",
        "",
        "## Source Attribution",
        "",
        "Inclusive witnesses overlap and must not be added together. Primary source",
        "is fixed-priority provenance, not causal credit. Fractional credit is additive.",
        "",
        "| Source | Primary hits | Inclusive witnesses | Exclusive hits | Fractional credit |",
        "|---|---:|---:|---:|---:|",
    ]
    sources = sorted(
        set(attribution["primary_source"])
        | set(attribution["inclusive_witness"])
        | set(attribution["exclusive_source"])
    )
    for source in sources:
        lines.append(
            f"| {source} | "
            f"{attribution['primary_source'].get(source, {}).get('truth_hits', 0)} | "
            f"{attribution['inclusive_witness'].get(source, {}).get('truth_hits', 0)} | "
            f"{attribution['exclusive_source'].get(source, {}).get('truth_hits', 0)} | "
            f"{attribution['fractional_credit'].get(source, {}).get('truth_hit_credit', 0.0):.2f} |"
        )

    for cohort_name in (
        "observed_history_profile_oov",
        "observed_history_content_oov",
        "activity_bucket",
        "historical_popularity_bucket",
        "latest_publish_age_bucket",
        "content_family",
        "type_id",
    ):
        values = attribution.get("cohorts", {}).get(cohort_name)
        if not values:
            continue
        lines.extend(
            [
                "",
                f"## Cohort: {cohort_name}",
                "",
                "| Value | Truth | Hits | Recall | Completion Gain Recall | Watch Gain Recall |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for value, row in sorted(
            values.items(), key=lambda item: (-int(item[1]["truth_pairs"]), item[0])
        ):
            lines.append(
                f"| {value} | {row['truth_pairs']} | {row['truth_hits']} | "
                f"{row['candidate_recall']:.6f} | "
                f"{row.get('completion_gain_recall', 0.0):.6f} | "
                f"{row.get('watch_time_gain_recall', 0.0):.6f} |"
            )

    rule_rows = payload.get("rule_attribution", {})
    if rule_rows:
        lines.extend(
            [
                "",
                "## Rule Attribution",
                "",
                "Truth can belong to multiple rules. Fractional mass assigns each pair",
                "a total weight of one across its eligible memberships.",
                "",
                "| Rule | Truth incidences | Recall | Fractional miss mass | Completion Gain Recall | Watch Gain Recall |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for rule_id, row in sorted(
            rule_rows.items(),
            key=lambda item: (-int(item[1]["truth_pair_incidences"]), item[0]),
        ):
            lines.append(
                f"| {rule_id} | {row['truth_pair_incidences']} | "
                f"{row['candidate_recall']:.6f} | "
                f"{row['fractional_miss_mass']:.3f} | "
                f"{row['fractional_completion_gain_recall']:.6f} | "
                f"{row['fractional_watch_time_gain_recall']:.6f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation Guard",
            "",
            "- Truth contains novel, catalog-eligible future observed engagement only.",
            "- `observed_history_content_oov=true` means unseen in observed pre-cutoff",
            "  history; it is not model-vocabulary OOV and not an unavailable item.",
            "- `latest_publish_age_bucket` uses mutable latest canonical metadata.",
            "  `latest_after_cutoff` is not proof that content was unavailable at cutoff.",
            "- This is retrieval recall, not CTR and not an online lift estimate.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.project_root.resolve()
    run_dir = root / "data" / "data" / "metrics" / args.source_run / args.domain
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = str(manifest.get("config", {}).get("evaluation_split", ""))
    if split not in {"validation", "test"}:
        raise ValueError(f"source run has unsupported evaluation split: {split!r}")

    candidate_path = run_dir / "retrieval_candidates.parquet"
    truth_path = run_dir / "evaluation_truth.parquet"
    candidate_columns = _available_columns(
        candidate_path,
        (
            "profile_id",
            "content_id",
            "retrieval_source",
            "retrieval_sources",
            "is_incumbent_candidate",
            "is_external_candidate",
            "is_safety_anchor",
            "activity_bucket",
        ),
    )
    candidates = pd.read_parquet(candidate_path, columns=candidate_columns)
    truth = pd.read_parquet(truth_path)
    candidate_hash = pair_keys_hash(candidates)
    truth_hash = pair_keys_hash(truth)
    expected_hashes = manifest.get("hashes", {})
    if candidate_hash != expected_hashes.get("candidate_pairs"):
        raise ValueError("candidate artifact hash does not match source manifest")
    if truth_hash != expected_hashes.get("evaluation_truth_pairs"):
        raise ValueError("truth artifact hash does not match source manifest")
    candidate_artifact_hash = sha256_file(candidate_path)
    truth_artifact_hash = sha256_file(truth_path)
    expected_candidate_artifact = expected_hashes.get(
        "retrieval_candidates_artifact"
    )
    expected_truth_artifact = expected_hashes.get("evaluation_truth_artifact")
    if (
        expected_candidate_artifact is not None
        and candidate_artifact_hash != expected_candidate_artifact
    ):
        raise ValueError("candidate artifact bytes do not match source manifest")
    if (
        expected_truth_artifact is not None
        and truth_artifact_hash != expected_truth_artifact
    ):
        raise ValueError("truth artifact bytes do not match source manifest")

    temporal = root / "data" / "data" / "canonical" / args.domain / "temporal"
    attributes, seen_metrics = _truth_attributes(temporal, split, truth)
    history_end = manifest.get("split", {}).get(split, {}).get("history_end")
    if not history_end:
        raise ValueError("source run manifest is missing split history_end")
    canonical = root / "data" / "data" / "canonical" / args.domain
    attributes = _add_content_diagnostic_attributes(
        attributes,
        _observed_history_pairs(temporal, split),
        pd.read_parquet(canonical / "contents.parquet"),
        history_end,
    )
    if "activity_bucket" in candidates:
        profile_activity = candidates[
            ["profile_id", "activity_bucket"]
        ].drop_duplicates("profile_id", keep="last")
        profile_activity["profile_id"] = profile_activity["profile_id"].astype(str)
        attributes = attributes.drop(columns=["activity_bucket"], errors="ignore").merge(
            profile_activity,
            on="profile_id",
            how="left",
            validate="many_to_one",
        )
    attribution = retrieval_truth_attribution(candidates, truth, attributes)
    membership_path = (
        root
        / "data"
        / "data"
        / "canonical"
        / args.domain
        / "content_rule_membership.parquet"
    )
    rule_attribution = _rule_attribution(
        candidates,
        truth,
        pd.read_parquet(membership_path),
    )
    evaluation_metrics = json.loads(
        (run_dir / "evaluation_metrics.json").read_text(encoding="utf-8")
    )
    data_metrics = json.loads(
        (run_dir / "data_metrics.json").read_text(encoding="utf-8")
    )
    canonical_manifest_path = (
        root / "data" / "data" / "canonical" / args.domain / "source_manifest.json"
    )
    return {
        "source_run": args.source_run,
        "domain": args.domain,
        "evaluation_split": split,
        "run_manifest_sha256": sha256_file(manifest_path),
        "canonical_source_manifest_sha256": (
            sha256_file(canonical_manifest_path)
            if canonical_manifest_path.exists()
            else None
        ),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidate_pairs_hash": candidate_hash,
        "candidate_artifact_sha256": candidate_artifact_hash,
        "evaluation_truth_pairs_hash": truth_hash,
        "evaluation_truth_artifact_sha256": truth_artifact_hash,
        "evaluation_truth_gain_hash": expected_hashes.get("evaluation_truth_gain"),
        "dirty_worktree": bool(manifest.get("dirty_worktree", False)),
        "seen_history": seen_metrics,
        "truth_funnel": {
            key: evaluation_metrics.get(key, data_metrics.get(key))
            for key in (
                "raw_positive_pairs",
                "repeat_positive_pairs_excluded",
                "novel_positive_pairs",
                "unavailable_absent_from_as_of_catalog_pairs",
                "unavailable_without_eligible_rule_pairs",
                "retrievable_positive_pairs",
                "profiles_with_zero_positive_candidate",
            )
        },
        "candidate_recall_curve": {
            key: evaluation_metrics.get(key)
            for key in (
                "candidate_recall@100",
                "candidate_recall@200",
                "candidate_recall@500",
                "candidate_recall@1000",
            )
        },
        "attribution": attribution,
        "rule_attribution": rule_attribution,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--domain", default="all")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(args)
    output = args.output
    if not output.is_absolute():
        output = args.project_root.resolve() / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    markdown_output = args.markdown_output or output.with_suffix(".md")
    if not markdown_output.is_absolute():
        markdown_output = args.project_root.resolve() / markdown_output
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(render_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "markdown_output": str(markdown_output),
                "candidate_recall": payload["attribution"]["candidate_recall"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
