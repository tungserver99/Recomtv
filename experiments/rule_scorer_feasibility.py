from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from recsys_vnpt.evaluation import evaluate_scored_candidates
from recsys_vnpt.metrics import paired_bootstrap_confidence_interval
from recsys_vnpt.pipeline_v2 import _read_canonical_interactions
from recsys_vnpt.retrieval import (
    historical_content_popularity,
    profile_rule_affinity_scores,
)
from recsys_vnpt.rule_assembly import SPECIAL_RULES
from recsys_vnpt.rule_scorer import (
    FEATURE_COLUMNS,
    GAIN_COLUMNS,
    history_features as _history_features,
    membership_and_rules_as_of as _membership_and_rules_as_of,
    rank_score as _rank_score,
    window_targets as _window_targets,
)
from recsys_vnpt.temporal import load_temporal_windows


CORE_METRICS = (
    "completion_gain_rule_ndcg@20",
    "watch_time_gain_rule_ndcg@20",
    "completion_gain_macro_content_ndcg@10",
    "watch_time_gain_macro_content_ndcg@10",
    "completion_gain_all_rule_macro_content_ndcg@10",
    "watch_time_gain_all_rule_macro_content_ndcg@10",
    "overall_recall@200",
    "completion_gain_overall_ndcg@200",
    "watch_time_gain_overall_ndcg@200",
    "exact_20_normal_rule_rate",
    "exact_10_content_rate",
)


def _normal_membership(canonical: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = pd.read_parquet(
        canonical / "content_rule_membership.parquet",
        columns=["content_id", "rule_id"],
    ).drop_duplicates()
    full = full.astype({"content_id": str, "rule_id": str})
    normal = full[~full["rule_id"].isin(SPECIAL_RULES)].copy()
    return full, normal







def _paired_intervals(
    incumbent: dict[str, Any], challenger: dict[str, Any]
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for metric in (
        "completion_gain_rule_ndcg@20",
        "watch_time_gain_rule_ndcg@20",
        "completion_gain_macro_content_ndcg@10",
        "watch_time_gain_macro_content_ndcg@10",
        "completion_gain_all_rule_macro_content_ndcg@10",
        "watch_time_gain_all_rule_macro_content_ndcg@10",
        "overall_recall@200",
    ):
        baseline = {
            profile: float(values.get(metric, 0.0))
            for profile, values in incumbent["per_profile"].items()
        }
        candidate = {
            profile: float(values.get(metric, 0.0))
            for profile, values in challenger["per_profile"].items()
        }
        output[metric] = paired_bootstrap_confidence_interval(
            baseline, candidate, seed=42
        )
    return output


def _metric_summary(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(metrics[key]) for key in CORE_METRICS if key in metrics}


def _cohort_comparison(
    incumbent: dict[str, Any],
    challenger: dict[str, Any],
    profile_cohorts: pd.DataFrame,
) -> dict[str, dict[str, dict[str, float | int]]]:
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    metrics = (
        "completion_gain_rule_ndcg@20",
        "watch_time_gain_rule_ndcg@20",
        "completion_gain_macro_content_ndcg@10",
        "watch_time_gain_macro_content_ndcg@10",
    )
    for column in ("profile_oov", "activity_bucket"):
        if column not in profile_cohorts:
            continue
        by_value: dict[str, dict[str, float | int]] = {}
        for value, cohort in profile_cohorts.groupby(
            column, dropna=False, observed=True, sort=True
        ):
            profiles = [
                profile
                for profile in cohort["profile_id"].astype(str)
                if profile in incumbent["per_profile"]
                and profile in challenger["per_profile"]
            ]
            row: dict[str, float | int] = {"profiles": len(profiles)}
            for metric in metrics:
                baseline = np.asarray(
                    [incumbent["per_profile"][p].get(metric, 0.0) for p in profiles],
                    dtype=float,
                )
                candidate = np.asarray(
                    [challenger["per_profile"][p].get(metric, 0.0) for p in profiles],
                    dtype=float,
                )
                row[f"{metric}.baseline"] = float(baseline.mean()) if len(baseline) else 0.0
                row[f"{metric}.candidate"] = float(candidate.mean()) if len(candidate) else 0.0
                row[f"{metric}.delta"] = float((candidate - baseline).mean()) if len(candidate) else 0.0
            by_value[str(value)] = row
        output[column] = by_value
    return output


def _cohort_quality_deltas(
    cohort_comparison: dict[str, dict[str, dict[str, float | int]]],
) -> list[float]:
    return [
        float(value)
        for by_value in cohort_comparison.values()
        for row in by_value.values()
        if int(row["profiles"]) > 0
        for metric, value in row.items()
        if metric.endswith(
            ("rule_ndcg@20.delta", "macro_content_ndcg@10.delta")
        )
    ]


def _production_provenance_gate(
    *,
    quality_gate_has_eligible_variant: bool,
    evaluation_scope_metrics: dict[str, Any],
    training_scope_metrics: dict[str, dict[str, Any]],
) -> tuple[dict[str, bool], list[str]]:
    blockers: list[str] = []
    historical_rule_membership_available = not bool(
        evaluation_scope_metrics.get("rule_membership_backfill", False)
    )
    if not historical_rule_membership_available:
        blockers.append(
            "historical rule membership is unavailable; current membership is used as a reported backfill"
        )

    metadata_backfill_rates = [
        float(evaluation_scope_metrics.get("metadata_backfill_rate", 0.0)),
        *[
            float(scope.get("metadata_backfill_rate", 0.0))
            for scope in training_scope_metrics.values()
        ],
    ]
    historical_mutable_metadata_available = (
        max(metadata_backfill_rates, default=0.0) == 0.0
    )
    if not historical_mutable_metadata_available:
        blockers.append(
            "historical mutable content metadata is unavailable; current metadata is used as a reported backfill"
        )

    gate = {
        "experiment_quality_gate_has_eligible_variant": bool(
            quality_gate_has_eligible_variant
        ),
        "historical_rule_membership_available": (
            historical_rule_membership_available
        ),
        "historical_mutable_metadata_available": (
            historical_mutable_metadata_available
        ),
    }
    gate["passed"] = all(gate.values())
    return gate, blockers


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    root = args.project_root.resolve()
    canonical = root / "data" / "data" / "canonical" / args.domain
    run_dir = root / "data" / "data" / "metrics" / args.source_run / args.domain
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    manifest_split = str(manifest.get("config", {}).get("evaluation_split", ""))
    if manifest_split != args.split:
        raise ValueError(
            f"source run split {manifest_split!r} does not match --split {args.split!r}"
        )
    manifest_cutoff = pd.Timestamp(
        manifest.get("split", {}).get(args.split, {}).get("history_end")
    )
    cutoff = (
        manifest_cutoff
        if args.cutoff is None
        else pd.Timestamp(args.cutoff)
    )
    if cutoff != manifest_cutoff:
        raise ValueError(
            f"source run cutoff {manifest_cutoff} does not match --cutoff {cutoff}"
        )
    full_membership, membership = _normal_membership(canonical)
    rule_index = pd.DataFrame(
        {"rule_id": sorted(membership["rule_id"].astype(str).unique())}
    )
    rule_index["rule_index"] = np.arange(len(rule_index), dtype=np.int16)
    (
        evaluation_membership,
        evaluation_rules,
        evaluation_scope_metrics,
        retrieval_ids,
    ) = _membership_and_rules_as_of(
        canonical, cutoff, membership, rule_index
    )

    truth = pd.read_parquet(run_dir / "evaluation_truth.parquet")
    candidates = pd.read_parquet(run_dir / "retrieval_candidates.parquet")
    required_candidate_columns = {
        "profile_id",
        "content_id",
        "type_id",
        "score",
    }
    missing_candidate_columns = sorted(
        required_candidate_columns - set(candidates)
    )
    if missing_candidate_columns:
        raise ValueError(
            "source candidate artifact is not replayable; missing "
            f"{missing_candidate_columns}"
        )
    profiles = candidates["profile_id"].astype(str).drop_duplicates()
    positive_keys = set(
        truth[["profile_id", "content_id"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    candidates["label"] = [
        int((str(profile), str(content)) in positive_keys)
        for profile, content in candidates[["profile_id", "content_id"]].itertuples(
            index=False, name=None
        )
    ]
    profile_cohorts = (
        candidates[
            [
                column
                for column in ("profile_id", "profile_oov", "activity_bucket")
                if column in candidates
            ]
        ]
        .drop_duplicates("profile_id", keep="first")
        .copy()
    )

    artifact_windows = load_temporal_windows(canonical / "temporal")
    training_windows = list(artifact_windows[:3])
    if args.include_validation_training:
        if args.split != "test":
            raise ValueError(
                "validation labels may only be added when evaluating the test split"
            )
        training_windows.append(artifact_windows[3])
    window_scopes: dict[
        str, tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], set[str]]
    ] = {}
    feature_content_ids = set(retrieval_ids)
    for window in training_windows:
        scope = _membership_and_rules_as_of(
            canonical, window.history_end, membership, rule_index
        )
        window_scopes[window.name] = scope
        feature_content_ids.update(scope[3])
    interactions = _read_canonical_interactions(
        canonical / "interactions.parquet",
        content_ids=feature_content_ids,
        columns=["profile_id", "content_id", "create_date", "duration"],
    )
    evaluation_interactions = interactions[
        interactions["content_id"].astype(str).isin(retrieval_ids)
    ].copy()

    train_parts: list[pd.DataFrame] = []
    training_rows_by_window: dict[str, int] = {}
    scope_metrics_by_window: dict[str, dict[str, Any]] = {}
    for window in training_windows:
        window_membership, window_rules, scope_metrics, _ = window_scopes[
            window.name
        ]
        targets = _window_targets(
            canonical / "temporal" / f"{window.name}.parquet",
            window_membership,
            window_rules,
        )
        features = _history_features(
            interactions,
            window.history_end,
            targets["profile_id"],
            window_membership,
            window_rules,
        )
        part = features.merge(
            targets[["profile_id", "rule_id", *GAIN_COLUMNS]],
            on=["profile_id", "rule_id"],
            how="inner",
        )
        part["window"] = window.name
        train_parts.append(part)
        training_rows_by_window[window.name] = len(part)
        scope_metrics_by_window[window.name] = scope_metrics
    training = pd.concat(train_parts, ignore_index=True)

    categorical_mask = [column == "rule_index" for column in FEATURE_COLUMNS]
    model_config = {
        "loss": "squared_error",
        "learning_rate": args.learning_rate,
        "max_iter": args.max_iter,
        "max_leaf_nodes": args.max_leaf_nodes,
        "min_samples_leaf": args.min_samples_leaf,
        "l2_regularization": args.l2_regularization,
        "categorical_features": categorical_mask,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 8,
        "random_state": args.seed,
    }
    models: dict[str, HistGradientBoostingRegressor] = {}
    model_metrics: dict[str, Any] = {}
    x_train = training[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    for gain in GAIN_COLUMNS:
        if args.target_normalization == "profile_share":
            denominators = training.groupby(
                ["window", "profile_id"], sort=False
            )[gain].transform("sum").clip(lower=1e-12)
            target = (
                training[gain] / denominators
            ).to_numpy(dtype=np.float32)
        else:
            target = np.log1p(training[gain].to_numpy(dtype=np.float32))
        weights = np.where(target > 0, args.observed_weight, 1.0).astype(
            np.float32
        )
        model = HistGradientBoostingRegressor(**model_config)
        model_started = time.perf_counter()
        model.fit(x_train, target, sample_weight=weights)
        models[gain] = model
        model_metrics[gain] = {
            "fit_seconds": time.perf_counter() - model_started,
            "iterations": int(model.n_iter_),
            "target_positive_rate": float(np.mean(target > 0)),
            "target_mean": float(np.mean(target)),
        }
    del x_train, training, train_parts
    gc.collect()

    validation_features = _history_features(
        interactions,
        cutoff,
        profiles,
        evaluation_membership,
        evaluation_rules,
    )
    x_validation = validation_features[list(FEATURE_COLUMNS)].to_numpy(
        dtype=np.float32
    )
    for gain, model in models.items():
        validation_features[f"predicted_{gain}"] = model.predict(x_validation)
    validation_features["learned_rule_score"] = (
        _rank_score(validation_features, "predicted_completion_gain")
        + _rank_score(validation_features, "predicted_watch_time_gain")
    )

    popularity = historical_content_popularity(
        evaluation_interactions, cutoff=cutoff
    )
    incumbent_scores = profile_rule_affinity_scores(
        pd.DataFrame({"profile_id": profiles}),
        full_membership,
        popularity,
        interactions=evaluation_interactions,
        cutoff=cutoff,
    ).rename(columns={"rule_score": "incumbent_rule_score"})
    rule_scores = validation_features.merge(
        incumbent_scores,
        on=["profile_id", "rule_id"],
        how="left",
    )
    rule_scores["incumbent_rule_score"] = rule_scores[
        "incumbent_rule_score"
    ].fillna(0.0)
    rule_scores["incumbent_rrf"] = _rank_score(
        rule_scores, "incumbent_rule_score"
    )
    rule_scores["learned_rrf"] = _rank_score(
        rule_scores, "learned_rule_score"
    )

    variants: dict[str, pd.Series] = {
        "incumbent_count_affinity": rule_scores["incumbent_rule_score"],
        "learned_graded_rule_scorer": rule_scores["learned_rule_score"],
    }
    for weight in args.fusion_weights:
        variants[f"incumbent_plus_learned_rrf_{weight:g}"] = (
            rule_scores["incumbent_rrf"] + weight * rule_scores["learned_rrf"]
        )

    saved_metrics = json.loads(
        (run_dir / "evaluation_metrics.json").read_text(encoding="utf-8")
    )
    evaluated: dict[str, dict[str, Any]] = {}
    for name, values in variants.items():
        profile_rule_scores = rule_scores[["profile_id", "rule_id"]].copy()
        profile_rule_scores["rule_score"] = values.to_numpy(dtype=np.float64)
        metrics, _ = evaluate_scored_candidates(
            candidates,
            full_membership,
            popularity,
            ground_truth_frame=truth,
            profile_rule_scores=profile_rule_scores,
            score_semantics="rank_score",
        )
        evaluated[name] = metrics
        gc.collect()

    incumbent = evaluated["incumbent_count_affinity"]
    parity = {
        metric: float(incumbent[metric]) - float(saved_metrics[metric])
        for metric in CORE_METRICS
        if metric in incumbent and metric in saved_metrics
    }
    missing_parity_metrics = sorted(
        set(CORE_METRICS) - set(parity)
    )
    maximum_parity_delta = max(map(abs, parity.values()), default=float("inf"))
    parity_valid = not missing_parity_metrics and maximum_parity_delta <= 1e-12
    summaries: dict[str, Any] = {}
    for name, metrics in evaluated.items():
        summary: dict[str, Any] = {"metrics": _metric_summary(metrics)}
        if name != "incumbent_count_affinity":
            summary["paired_delta_vs_incumbent_95"] = _paired_intervals(
                incumbent, metrics
            )
            summary["cohort_comparison"] = _cohort_comparison(
                incumbent, metrics, profile_cohorts
            )
        summaries[name] = summary

    positive_ci_variants = []
    experiment_gate_eligible_variants = []
    for name, summary in summaries.items():
        if name == "incumbent_count_affinity":
            continue
        intervals = summary["paired_delta_vs_incumbent_95"]
        positive_ci = all(
            intervals[metric]["lower"] > 0
            for metric in (
                "completion_gain_rule_ndcg@20",
                "watch_time_gain_rule_ndcg@20",
            )
        )
        if positive_ci:
            positive_ci_variants.append(name)
        candidate_metrics = summary["metrics"]
        relative_rule_gains = {
            metric: candidate_metrics[metric]
            / max(float(incumbent[metric]), 1e-12)
            - 1.0
            for metric in (
                "completion_gain_rule_ndcg@20",
                "watch_time_gain_rule_ndcg@20",
            )
        }
        summary["relative_rule_improvement"] = relative_rule_gains
        relative_content_gains = {
            metric: candidate_metrics[metric]
            / max(float(incumbent[metric]), 1e-12)
            - 1.0
            for metric in (
                "completion_gain_macro_content_ndcg@10",
                "watch_time_gain_macro_content_ndcg@10",
            )
        }
        summary["relative_hierarchical_content_improvement"] = (
            relative_content_gains
        )
        output_contract = all(
            candidate_metrics[metric] == 1.0
            for metric in (
                "exact_20_normal_rule_rate",
                "exact_10_content_rate",
            )
        )
        recall_retained = candidate_metrics["overall_recall@200"] >= (
            float(incumbent["overall_recall@200"]) * 0.995
        )
        all_rule_content_retained = all(
            candidate_metrics[metric]
            >= float(incumbent[metric]) * 0.995
            for metric in (
                "completion_gain_all_rule_macro_content_ndcg@10",
                "watch_time_gain_all_rule_macro_content_ndcg@10",
            )
        )
        cohort_quality_deltas = _cohort_quality_deltas(
            summary["cohort_comparison"]
        )
        summary["minimum_cohort_quality_delta"] = min(
            cohort_quality_deltas, default=0.0
        )
        summary["promotion_gate"] = {
            "incumbent_replay_parity": parity_valid,
            "paired_rule_ci_lower_above_zero": positive_ci,
            "paired_hierarchical_content_ci_lower_above_zero": all(
                intervals[metric]["lower"] > 0
                for metric in (
                    "completion_gain_macro_content_ndcg@10",
                    "watch_time_gain_macro_content_ndcg@10",
                )
            ),
            "both_rule_ndcg_relative_improvement_at_least_3pct": all(
                value >= 0.03 for value in relative_rule_gains.values()
            ),
            "both_hierarchical_content_ndcg_relative_improvement_at_least_3pct": all(
                value >= 0.03 for value in relative_content_gains.values()
            ),
            "overall_recall_retained_within_0_5pct": recall_retained,
            "all_rule_content_ndcg_retained_within_0_5pct": (
                all_rule_content_retained
            ),
            "seen_cold_activity_rule_and_content_cohorts_do_not_drop_over_2pct_absolute": (
                summary["minimum_cohort_quality_delta"] >= -0.02
            ),
            "output_contract_100pct": output_contract,
        }
        if all(summary["promotion_gate"].values()):
            experiment_gate_eligible_variants.append(name)

    production_gate, provenance_blockers = _production_provenance_gate(
        quality_gate_has_eligible_variant=bool(
            experiment_gate_eligible_variants
        ),
        evaluation_scope_metrics=evaluation_scope_metrics,
        training_scope_metrics=scope_metrics_by_window,
    )
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    return {
        "experiment": f"{args.split}_graded_rule_scorer_feasibility_v2",
        "source_run": args.source_run,
        "source_manifest_hash": hashlib.sha256(
            (run_dir / "run_manifest.json").read_bytes()
        ).hexdigest(),
        "script_sha256": script_hash,
        "source_hashes": manifest.get("hashes", {}),
        "domain": args.domain,
        "cutoff": str(cutoff),
        "seed": args.seed,
        "feature_columns": list(FEATURE_COLUMNS),
        "training_rows": int(sum(training_rows_by_window.values())),
        "training_rows_by_window": training_rows_by_window,
        "training_scope_by_window": scope_metrics_by_window,
        "evaluation_scope": evaluation_scope_metrics,
        "include_validation_training": args.include_validation_training,
        "normal_rules": int(len(evaluation_rules)),
        "evaluation_profiles": int(len(profiles)),
        "interaction_rows_loaded_for_training_features": int(len(interactions)),
        "interaction_rows_loaded_for_evaluation_features": int(
            len(evaluation_interactions)
        ),
        "model_config": {
            key: value
            for key, value in model_config.items()
            if key != "categorical_features"
        }
        | {
            "observed_weight": args.observed_weight,
            "target_normalization": args.target_normalization,
        },
        "model_fit": model_metrics,
        "incumbent_parity_delta": parity,
        "missing_incumbent_parity_metrics": missing_parity_metrics,
        "maximum_absolute_parity_delta": maximum_parity_delta,
        "incumbent_replay_parity_valid": parity_valid,
        "variants": summaries,
        "positive_rule_ci_variants": positive_ci_variants,
        "experiment_gate_eligible_variants": experiment_gate_eligible_variants,
        "production_gate": production_gate,
        "provenance_blockers": provenance_blockers,
        "production_change_recommended": bool(production_gate["passed"]),
        "elapsed_seconds": time.perf_counter() - started,
        "notes": [
            "Targets are summed future gains, matching the evaluator's rule relevance.",
            "Interaction events and catalog availability are cutoff-scoped.",
            "Rule assignments are current-membership backfills because historical mappings are absent.",
            "This is a feasibility experiment, not a production promotion run.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--domain", default="all")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--source-run",
        required=True,
        help="Schema-v2 metrics run whose split, truth, and candidates are replayed.",
    )
    parser.add_argument(
        "--cutoff",
        default=None,
        help="Optional assertion; by default the cutoff is read from the source manifest.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf", type=int, default=50)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument("--observed-weight", type=float, default=3.0)
    parser.add_argument(
        "--target-normalization",
        choices=("raw_log1p", "profile_share"),
        default="profile_share",
    )
    parser.add_argument("--include-validation-training", action="store_true")
    parser.add_argument(
        "--fusion-weights", type=float, nargs="+", default=[1.25]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/RULE_SCORER_FEASIBILITY.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(args)
    output = args.output
    if not output.is_absolute():
        output = args.project_root.resolve() / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    concise = {
        "output": str(output),
        "maximum_absolute_parity_delta": result[
            "maximum_absolute_parity_delta"
        ],
        "positive_rule_ci_variants": result["positive_rule_ci_variants"],
        "experiment_gate_eligible_variants": result[
            "experiment_gate_eligible_variants"
        ],
        "production_change_recommended": result[
            "production_change_recommended"
        ],
        "provenance_blockers": result["provenance_blockers"],
        "elapsed_seconds": result["elapsed_seconds"],
        "variants": {
            name: {
                key: value
                for key, value in payload["metrics"].items()
                if key.endswith("rule_ndcg@20") or key == "overall_recall@200"
            }
            for name, payload in result["variants"].items()
        },
    }
    print(json.dumps(concise, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
