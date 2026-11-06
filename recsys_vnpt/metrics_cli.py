from __future__ import annotations

import argparse
import json
from pathlib import Path

from recsys_vnpt.metrics import paired_bootstrap_confidence_interval
from recsys_vnpt.metrics_store import MetricsStore, evaluate_gate, flattened_bundle, render_markdown
from recsys_vnpt.paths import model_artifact_path


COMPARISON_SUMMARY_METRICS = (
    "data.evaluation_profile_selection_hash",
    "data.evaluation_truth_gain_hash",
    "data.evaluation_truth_pairs_hash",
    "evaluation.evaluation_profile_selection_hash",
    "evaluation.evaluation_truth_gain_hash",
    "evaluation.evaluation_truth_pairs_hash",
    "evaluation.candidate_pairs_hash",
    "evaluation.candidate_profiles",
    "evaluation.retrievable_truth_profiles",
    "evaluation.retrievable_positive_pairs",
    "evaluation.candidate_recall@100",
    "evaluation.candidate_recall@200",
    "evaluation.candidate_recall@500",
    "evaluation.candidate_recall@1000",
    "evaluation.candidate_completion_gain_recall@1000",
    "evaluation.candidate_watch_time_gain_recall@1000",
    "evaluation.rule_ndcg@20",
    "evaluation.macro_content_ndcg@10",
    "evaluation.overall_recall@200",
    "evaluation.completion_gain_rule_ndcg@20",
    "evaluation.completion_gain_macro_content_ndcg@10",
    "evaluation.completion_gain_all_rule_macro_content_ndcg@10",
    "evaluation.watch_time_gain_rule_ndcg@20",
    "evaluation.watch_time_gain_macro_content_ndcg@10",
    "evaluation.watch_time_gain_all_rule_macro_content_ndcg@10",
    "evaluation.candidate_rule_groups",
    "evaluation.eligible_candidate_rule_groups",
    "evaluation.eligible_candidate_rule_group_rate",
    "evaluation.mean_eligible_candidate_rules_per_profile",
    "evaluation.loaded_interaction_rows",
    "evaluation.serving_interaction_rows",
    "evaluation.truth_audit_only_interaction_rows",
    "evaluation.truth_audit_only_interaction_contents",
    "evaluation.retrieval_truth_attribution.cohorts.observed_history_profile_oov.true.candidate_recall",
    "evaluation.retrieval_truth_attribution.cohorts.observed_history_content_oov.true.candidate_recall",
    "evaluation.retrieval_truth_attribution.cohorts.activity_bucket.cold.candidate_recall",
    "evaluation.exact_20_normal_rule_rate",
    "evaluation.exact_10_content_rate",
    "evaluation.external_candidate_mode",
    "evaluation.external_candidate_slots",
    "evaluation.external_replacement_requires_witness",
    "evaluation.external_witness_rejected_candidates",
    "evaluation.external_witness_rejected_profiles",
    "evaluation.preblend_incumbent_truth_hits",
    "evaluation.blend_gained_truth_hits",
    "evaluation.blend_lost_truth_hits",
    "evaluation.blend_net_truth_hits",
    "evaluation.blend_net_completion_gain",
    "evaluation.blend_net_watch_time_gain",
    "evaluation.mean_incumbent_candidate_retention_rate",
    "evaluation.minimum_safety_anchor_retention_rate",
    "training.training_time_seconds",
    "training.peak_gpu_memory_bytes",
    "inference.evaluation_retrieval_time_seconds",
    "inference.evaluation_ranking_time_seconds",
    "inference.evaluation_end_to_end_time_seconds",
    "inference.inference_pairs",
    "inference.profiles_per_second",
    "inference.pairs_per_second",
)

PROMOTABLE_PHASES = frozenset(
    {
        "sprint-2",
        "sprint-3",
        "sprint-4",
        "label-l2",
        "label-l3",
        "label-l4",
        "label-l5",
    }
)


def _store(args: argparse.Namespace) -> MetricsStore:
    return MetricsStore(project_root=args.root, metrics_root=args.metrics_root)


def _aggregate_metrics(
    bundle,
    *,
    include_per_profile: bool,
) -> dict[str, object]:
    values = flattened_bundle(bundle)
    if include_per_profile:
        return values
    return {
        key: value
        for key, value in values.items()
        if not key.startswith("evaluation.per_profile.")
    }


def _has_test_evaluation_split(bundle) -> bool:
    split = bundle.manifest.split
    return "test" in split or split.get("evaluation_split") == "test"


def _retrieval_profile_intervals(
    store: MetricsStore,
    baseline,
    candidate,
) -> dict[str, dict[str, float | int]]:
    import pandas as pd

    baseline_path = (
        store.run_dir(baseline.manifest.run_id, baseline.manifest.domain)
        / "retrieval_profile_metrics.parquet"
    )
    candidate_path = (
        store.run_dir(candidate.manifest.run_id, candidate.manifest.domain)
        / "retrieval_profile_metrics.parquet"
    )
    if not baseline_path.exists() or not candidate_path.exists():
        return {}
    before = pd.read_parquet(baseline_path)
    after = pd.read_parquet(candidate_path)
    if "profile_id" not in before or "profile_id" not in after:
        return {}
    before["profile_id"] = before["profile_id"].astype(str)
    after["profile_id"] = after["profile_id"].astype(str)
    intervals: dict[str, dict[str, float | int]] = {}
    for metric in (
        "candidate_recall",
        "candidate_completion_gain_recall",
        "candidate_watch_time_gain_recall",
    ):
        if metric not in before or metric not in after:
            continue
        baseline_values = dict(
            zip(before["profile_id"], before[metric].astype(float), strict=True)
        )
        candidate_values = dict(
            zip(after["profile_id"], after[metric].astype(float), strict=True)
        )
        interval = paired_bootstrap_confidence_interval(
            baseline_values,
            candidate_values,
        )
        shared = sorted(set(baseline_values) & set(candidate_values))
        differences = [
            candidate_values[profile_id] - baseline_values[profile_id]
            for profile_id in shared
        ]
        interval.update(
            {
                "profiles_improved": sum(value > 1e-12 for value in differences),
                "profiles_regressed": sum(value < -1e-12 for value in differences),
                "profiles_tied": sum(abs(value) <= 1e-12 for value in differences),
            }
        )
        intervals[metric] = interval
    return intervals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read, compare, gate, and report RecSys-VNPT metrics.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--metrics-root")
    sub = parser.add_subparsers(dest="command", required=True)

    show = sub.add_parser("show")
    show.add_argument("--run", default="latest")
    show.add_argument("--domain")
    show.add_argument("--include-per-profile", action="store_true")

    compare = sub.add_parser("compare")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--domain", default="all")
    compare.add_argument("--output")
    compare.add_argument("--all-metrics", action="store_true")
    compare.add_argument("--include-per-profile", action="store_true")

    gate = sub.add_parser("gate")
    gate.add_argument("--run", required=True)
    gate.add_argument("--baseline")
    gate.add_argument("--domain", default="all")
    gate.add_argument("--phase", required=True)
    gate.add_argument("--promote", action="store_true")

    report = sub.add_parser("report")
    report.add_argument("--run", default="latest")
    report.add_argument("--baseline")
    report.add_argument("--domain")
    report.add_argument("--phase")
    report.add_argument("--output", default="reports/METRICS_LATEST.md")

    args = parser.parse_args(argv)
    store = _store(args)
    if args.command == "show":
        bundle = store.read_bundle(args.run, args.domain)
        print(
            json.dumps(
                {
                    "manifest": bundle.manifest.to_dict(),
                    "metrics": _aggregate_metrics(
                        bundle,
                        include_per_profile=args.include_per_profile,
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "compare":
        baseline = store.read_bundle(args.baseline, args.domain)
        candidate = store.read_bundle(args.candidate, args.domain)
        before = _aggregate_metrics(
            baseline,
            include_per_profile=args.include_per_profile,
        )
        after = _aggregate_metrics(
            candidate,
            include_per_profile=args.include_per_profile,
        )
        metric_names = sorted(set(before) | set(after))
        if not args.all_metrics and not args.include_per_profile:
            important = set(COMPARISON_SUMMARY_METRICS)
            metric_names = [key for key in metric_names if key in important]
        rows = {
            key: {"baseline": before.get(key), "candidate": after.get(key)}
            for key in metric_names
        }
        paired = _retrieval_profile_intervals(store, baseline, candidate)
        baseline_profiles = baseline.evaluation.get("per_profile", {})
        candidate_profiles = candidate.evaluation.get("per_profile", {})
        paired_metrics = (
            "rule_ndcg@20",
            "macro_content_ndcg@10",
            "all_rule_macro_content_ndcg@10",
            "overall_recall@200",
            "completion_gain_rule_ndcg@20",
            "completion_gain_macro_content_ndcg@10",
            "completion_gain_all_rule_macro_content_ndcg@10",
            "watch_time_gain_rule_ndcg@20",
            "watch_time_gain_macro_content_ndcg@10",
            "watch_time_gain_all_rule_macro_content_ndcg@10",
        )
        for metric in paired_metrics:
            shared_profiles = sorted(set(baseline_profiles) & set(candidate_profiles))
            comparable_profiles = [
                profile
                for profile in shared_profiles
                if metric in baseline_profiles[profile]
                and metric in candidate_profiles[profile]
            ]
            if not comparable_profiles:
                continue
            paired[metric] = paired_bootstrap_confidence_interval(
                {
                    profile: float(baseline_profiles[profile][metric])
                    for profile in comparable_profiles
                },
                {
                    profile: float(candidate_profiles[profile][metric])
                    for profile in comparable_profiles
                },
            )
        payload = {"metrics": rows, "paired_bootstrap_95": paired}
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                "# Metrics Comparison",
                "",
                f"- Baseline: {baseline.manifest.run_id}",
                f"- Candidate: {candidate.manifest.run_id}",
                "",
                "| Metric | Baseline | Candidate |",
                "|---|---:|---:|",
            ]
            lines.extend(
                f"| {metric} | {rows.get(metric, {}).get('baseline')} | "
                f"{rows.get(metric, {}).get('candidate')} |"
                for metric in COMPARISON_SUMMARY_METRICS
                if metric in rows
            )
            if paired:
                lines.extend(
                    [
                        "",
                        "## Paired Bootstrap 95%",
                        "",
                        "| Metric | Mean Delta | Lower | Upper | Profiles |",
                        "|---|---:|---:|---:|---:|",
                    ]
                )
                lines.extend(
                    f"| {metric} | {values['mean']} | {values['lower']} | "
                    f"{values['upper']} | {values['paired_profiles']} |"
                    for metric, values in paired.items()
                )
            output.write_text("\n".join(lines) + "\n", encoding="utf-8")
            payload["output"] = str(output)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "gate":
        bundle = store.read_bundle(args.run, args.domain)
        baseline = store.read_bundle(args.baseline, args.domain) if args.baseline else None
        if args.promote:
            if args.phase not in PROMOTABLE_PHASES:
                print("Promotion is only valid for ranking/retrieval model gates.")
                return 2
            if args.phase != bundle.manifest.phase:
                print("Artifact was not promoted because the requested phase does not match its manifest.")
                return 2
            if not _has_test_evaluation_split(bundle):
                print("Artifact was not promoted because it lacks a test evaluation split.")
                return 2
        result = evaluate_gate(bundle, args.phase, baseline)
        directory = store.run_dir(bundle.manifest.run_id, bundle.manifest.domain)
        (directory / "gate_result.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        if args.promote:
            if not result.passed:
                print("Artifact was not promoted because the metric gate failed.")
            else:
                pointer = (
                    model_artifact_path(
                        args.domain,
                        "retrieval_promoted.json",
                        project_root=args.root,
                    )
                    if args.phase == "sprint-4"
                    else model_artifact_path(
                        args.domain, "promoted.json", project_root=args.root
                    )
                )
                pointer.parent.mkdir(parents=True, exist_ok=True)
                temporary = pointer.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(
                        {
                            "run_id": bundle.manifest.run_id,
                            "domain": bundle.manifest.domain,
                            "model_family": bundle.manifest.model_family,
                            "phase": args.phase,
                            **(
                                {
                                    "candidate_pairs_hash": bundle.evaluation.get(
                                        "candidate_pairs_hash"
                                    ),
                                    "evaluation_truth_pairs_hash": bundle.evaluation.get(
                                        "evaluation_truth_pairs_hash"
                                    ),
                                }
                                if args.phase == "sprint-4"
                                else {}
                            ),
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                temporary.replace(pointer)
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.passed else 2
    bundle = store.read_bundle(args.run, args.domain)
    baseline = store.read_bundle(args.baseline, bundle.manifest.domain) if args.baseline else None
    phase = args.phase or bundle.manifest.phase
    gate_result = (
        evaluate_gate(bundle, phase, baseline)
        if phase.startswith("sprint-") or phase.startswith("label-l")
        else None
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(bundle, gate_result), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
