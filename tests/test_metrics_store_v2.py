import json

import pandas as pd
import pytest

from recsys_vnpt.metrics import hierarchical_ranking_metrics
from recsys_vnpt.metrics_cli import main as metrics_cli_main
from recsys_vnpt.metrics_schema import MetricBundle
from recsys_vnpt.metrics_store import (
    MetricsStore,
    create_manifest,
    evaluate_gate,
    render_markdown,
)


def test_hierarchical_content_metric_penalizes_missing_relevant_rule():
    recommendations = {
        "p1": [{"rule_id": "1", "content_ids": ["c1"]}],
    }
    truth = {"p1": {"c1", "c2"}}
    membership = {"c1": {"1"}, "c2": {"2"}}
    metrics = hierarchical_ranking_metrics(recommendations, truth, membership)
    assert metrics["rule_recall@20"] == 0.5
    assert metrics["macro_content_hit_rate@10"] == 0.5
    assert metrics["overall_recall@200"] == 0.5


def test_metrics_report_surfaces_blend_net_effect(tmp_path) -> None:
    bundle = MetricBundle(
        manifest=create_manifest(tmp_path, "all", "label-l3", "retrieval"),
        evaluation={
            "candidate_recall@1000": 0.8,
            "preblend_incumbent_candidate_recall": 0.79,
            "external_candidate_recall@1000": 0.57,
            "external_incremental_truth_hits": 80,
            "external_oracle_union_truth_hits": 900,
            "blend_gained_truth_hits": 14,
            "blend_lost_truth_hits": 7,
            "blend_net_truth_hits": 7,
            "blend_net_completion_gain": 3.0,
            "blend_net_watch_time_gain": 5.0,
        },
    )

    report = render_markdown(bundle)

    assert "| Preblend Candidate Recall@1,000 | 0.79 |" in report
    assert "| Blend gained truth hits | 14 |" in report
    assert "| Blend lost truth hits | 7 |" in report
    assert "| Blend net truth hits | 7 |" in report


def test_metrics_store_writes_schema_v2_files_and_domain_latest(tmp_path):
    store = MetricsStore(tmp_path)
    all_bundle = MetricBundle(
        manifest=create_manifest(tmp_path, "all", "sprint-1", "per_rule_popularity"),
        output={
            "exact_20_normal_rule_rate": 1,
            "exact_10_content_rate": 1,
            "special_rule_complete_rate": 1,
            "cross_file_consistency_rate": 1,
            "unmatched_rule_rate": 0,
            "invalid_content_rate": 0,
            "duplicate_within_rule_rate": 0,
        },
    )
    all_dir = store.write_bundle(all_bundle)
    movie_bundle = MetricBundle(
        manifest=create_manifest(tmp_path, "movie", "sprint-0", "canonical"),
        data={"duplicate_canonical_ids": 0},
    )
    store.write_bundle(movie_bundle)

    assert (all_dir / "run_manifest.json").exists()
    assert (all_dir / "evaluation_metrics.json").exists()
    pointer = json.loads((store.root / "latest.json").read_text(encoding="utf-8"))
    assert pointer["domains"]["all"] == all_bundle.manifest.run_id
    assert store.read_bundle("latest", "all").manifest.run_id == all_bundle.manifest.run_id
    assert evaluate_gate(all_bundle, "sprint-1").passed


def test_metrics_cli_hides_profile_payload_and_compares_retrieval_profiles(
    tmp_path, capsys
):
    store = MetricsStore(tmp_path)
    bundles = []
    for family, value in (("baseline", 0.25), ("candidate", 0.75)):
        bundle = MetricBundle(
            manifest=create_manifest(tmp_path, "all", "label-l3", family),
            evaluation={
                "candidate_recall@1000": value,
                "per_profile": {
                    "p1": {
                        "rule_ndcg@20": value,
                        "overall_recall@200": value,
                    },
                    "p2": {
                        "rule_ndcg@20": value,
                        "overall_recall@200": value,
                    },
                },
            },
        )
        directory = store.write_bundle(bundle)
        pd.DataFrame(
            {
                "profile_id": ["p1", "p2"],
                "candidate_recall": [value, value],
                "candidate_completion_gain_recall": [value, value],
                "candidate_watch_time_gain_recall": [value, value],
            }
        ).to_parquet(directory / "retrieval_profile_metrics.parquet", index=False)
        bundles.append(bundle)

    result = metrics_cli_main(
        [
            "--root",
            str(tmp_path),
            "compare",
            "--baseline",
            bundles[0].manifest.run_id,
            "--candidate",
            bundles[1].manifest.run_id,
            "--domain",
            "all",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert not any(
        key.startswith("evaluation.per_profile.") for key in payload["metrics"]
    )
    assert len(payload["metrics"]) < 50
    interval = payload["paired_bootstrap_95"]["candidate_recall"]
    assert interval["mean"] == 0.5
    assert interval["lower"] == 0.5
    assert interval["profiles_improved"] == 2
    assert interval["profiles_regressed"] == 0
    assert payload["paired_bootstrap_95"]["overall_recall@200"]["mean"] == 0.5
    assert "macro_content_ndcg@10" not in payload["paired_bootstrap_95"]


@pytest.mark.parametrize(
    ("manifest_phase", "split", "command_phase"),
    [
        ("sprint-5", {"test": {}}, "label-l3"),
        ("label-l3", {"validation": {}}, "label-l3"),
    ],
)
def test_metrics_cli_promotion_rejects_manifest_phase_mismatch_and_non_test_split(
    tmp_path, capsys, manifest_phase, split, command_phase
):
    store = MetricsStore(tmp_path)
    bundle = MetricBundle(
        manifest=create_manifest(
            tmp_path,
            "all",
            manifest_phase,
            "retrieval",
            split=split,
        ),
        evaluation={
            "candidate_recall@1000": 0.99,
            "candidate_completion_gain_recall@1000": 0.99,
            "candidate_watch_time_gain_recall@1000": 0.99,
        },
    )
    store.write_bundle(bundle)
    pointer = tmp_path / "data" / "data" / "model" / "all" / "promoted.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text('{"run_id": "existing"}', encoding="utf-8")
    original_pointer = pointer.read_bytes()
    gate_result = store.run_dir(bundle.manifest.run_id, "all") / "gate_result.json"
    gate_result.write_text('{"passed": "existing"}', encoding="utf-8")
    original_gate_result = gate_result.read_bytes()

    result = metrics_cli_main(
        [
            "--root",
            str(tmp_path),
            "gate",
            "--run",
            bundle.manifest.run_id,
            "--domain",
            "all",
            "--phase",
            command_phase,
            "--promote",
        ]
    )

    assert result == 2
    assert pointer.read_bytes() == original_pointer
    assert gate_result.read_bytes() == original_gate_result
    assert "not promoted" in capsys.readouterr().out


def test_metrics_cli_promotion_rejects_unsupported_passing_phase_before_gate_write(
    tmp_path, capsys
):
    store = MetricsStore(tmp_path)
    bundle = MetricBundle(
        manifest=create_manifest(
            tmp_path,
            "all",
            "sprint-5",
            "per_rule_popularity",
            split={"test": {}},
        ),
        inference={
            "processing_error_count": 0,
            "forecast_error_rate": 0.05,
            "ramp_warm_repetitions": 3,
            "ramp_deterministic_warm_hash_rate": True,
        },
        output={
            "profile_output_coverage": 1,
            "exact_20_normal_rule_rate": 1,
            "exact_10_content_rate": 1,
            "special_rule_complete_rate": 1,
            "cross_file_consistency_rate": 1,
            "invalid_content_rate": 0,
            "unmatched_rule_rate": 0,
        },
    )
    store.write_bundle(bundle)
    pointer = tmp_path / "data" / "data" / "model" / "all" / "promoted.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text('{"run_id": "existing"}', encoding="utf-8")
    gate_result = store.run_dir(bundle.manifest.run_id, "all") / "gate_result.json"
    gate_result.write_text('{"passed": "existing"}', encoding="utf-8")

    result = metrics_cli_main(
        [
            "--root",
            str(tmp_path),
            "gate",
            "--run",
            bundle.manifest.run_id,
            "--domain",
            "all",
            "--phase",
            "sprint-5",
            "--promote",
        ]
    )

    assert result == 2
    assert pointer.read_bytes() == b'{"run_id": "existing"}'
    assert gate_result.read_bytes() == b'{"passed": "existing"}'
    assert "only valid" in capsys.readouterr().out


def test_sprint5_gate_uses_explicit_ramp_warm_reference(tmp_path):
    bundle = MetricBundle(
        manifest=create_manifest(tmp_path, "all", "sprint-5", "embedding_fcn"),
        inference={
            "processing_error_count": 0,
            "forecast_error_rate": 0.05,
            "warm_repetitions": 0,
            "deterministic_warm_hash_rate": False,
            "ramp_warm_repetitions": 3,
            "ramp_deterministic_warm_hash_rate": True,
        },
        output={
            "profile_output_coverage": 1,
            "exact_20_normal_rule_rate": 1,
            "exact_10_content_rate": 1,
            "special_rule_complete_rate": 1,
            "cross_file_consistency_rate": 1,
            "invalid_content_rate": 0,
            "unmatched_rule_rate": 0,
        },
    )

    gate = evaluate_gate(bundle, "sprint-5")

    assert gate.passed
    assert gate.checks[-2].metric == "inference.ramp_warm_repetitions"
    assert gate.checks[-1].metric == "inference.ramp_deterministic_warm_hash_rate"


def test_neural_gate_requires_same_candidate_universe_and_evidence_floor(tmp_path):
    common = {
        "rule_ndcg@20": 0.5,
        "macro_content_ndcg@10": 0.2,
        "overall_recall@200": 0.4,
        "candidate_recall@1000": 0.99,
        "candidate_max_actual_k": 1000,
        "evaluation_truth_pairs_hash": "truth",
        "retrievable_truth_profiles": 500,
        "retrievable_positive_pairs": 700,
        "consumed_candidate_overlap": 0,
        "exact_20_normal_rule_rate": 1,
        "exact_10_content_rate": 1,
        "per_profile": {
            "p1": {"rule_ndcg@20": 0.5, "macro_content_ndcg@10": 0.2}
        },
    }
    baseline = MetricBundle(
        manifest=create_manifest(tmp_path, "all", "sprint-1", "popularity"),
        evaluation={**common, "candidate_pairs_hash": "baseline"},
    )
    candidate = MetricBundle(
        manifest=create_manifest(tmp_path, "all", "sprint-3", "fcn_official"),
        evaluation={
            **common,
            "rule_ndcg@20": 0.6,
            "macro_content_ndcg@10": 0.3,
            "candidate_pairs_hash": "different",
        },
    )
    gate = evaluate_gate(candidate, "sprint-3", baseline)
    hash_check = next(
        check
        for check in gate.checks
        if check.metric == "evaluation.candidate_pairs_hash"
    )
    assert not hash_check.passed

    candidate.evaluation["candidate_pairs_hash"] = "baseline"
    candidate.evaluation["retrievable_truth_profiles"] = 399
    floor_gate = evaluate_gate(candidate, "sprint-3", baseline)
    floor_check = next(
        check
        for check in floor_gate.checks
        if check.metric == "evaluation.retrievable_truth_profiles"
    )
    assert not floor_check.passed


def test_retrieval_gate_allows_new_candidates_but_requires_quality_and_speed(tmp_path):
    common = {
        "rule_ndcg@20": 0.8,
        "macro_content_ndcg@10": 0.1,
        "overall_recall@200": 0.33,
        "candidate_recall@1000": 0.99,
        "candidate_max_actual_k": 1000,
        "evaluation_truth_pairs_hash": "truth",
        "candidate_profiles": 500,
        "retrievable_truth_profiles": 500,
        "retrievable_positive_pairs": 700,
        "consumed_candidate_overlap": 0,
        "exact_20_normal_rule_rate": 1,
        "exact_10_content_rate": 1,
    }
    baseline = MetricBundle(
        manifest=create_manifest(tmp_path, "all", "sprint-1", "popularity"),
        evaluation={**common, "candidate_pairs_hash": "old"},
        inference={"evaluation_end_to_end_time_seconds": 50.0},
    )
    candidate = MetricBundle(
        manifest=create_manifest(tmp_path, "all", "sprint-4", "retrieval"),
        evaluation={
            **common,
            "rule_ndcg@20": 0.796,
            "macro_content_ndcg@10": 0.0995,
            "candidate_pairs_hash": "new",
        },
        inference={"evaluation_end_to_end_time_seconds": 8.0},
    )

    gate = evaluate_gate(candidate, "sprint-4", baseline)
    assert gate.passed
    assert not any(
        check.metric == "evaluation.candidate_pairs_hash" for check in gate.checks
    )

    candidate.evaluation["evaluation_truth_pairs_hash"] = "different-truth"
    truth_gate = evaluate_gate(candidate, "sprint-4", baseline)
    truth_check = next(
        check
        for check in truth_gate.checks
        if check.metric == "evaluation.evaluation_truth_pairs_hash"
    )
    assert not truth_check.passed

    candidate.evaluation["evaluation_truth_pairs_hash"] = "truth"
    candidate.inference["evaluation_end_to_end_time_seconds"] = 20.0
    slow_gate = evaluate_gate(candidate, "sprint-4", baseline)
    speed_check = next(
        check
        for check in slow_gate.checks
        if check.metric == "inference.evaluation_profiles_per_second_relative"
    )
    assert not speed_check.passed


def test_label_l0_gate_requires_playback_contract_without_duplicate_leakage(tmp_path):
    bundle = MetricBundle(
        manifest=create_manifest(tmp_path, "all", "label-l0", "engagement_v1"),
        data={
            "duplicate_leakage": 0,
            "duplicate_event_hashes": 0,
            "playback_events": 10,
            "label_contract_hash": "contract-hash",
            "target_version": "engagement_v1",
            "source_interaction_rows": 100,
            "playback_source_rows": 40,
            "playback_source_rate": 0.4,
            "interaction_catalog_audit": {
                "by_catalog_class": {
                    "eligible": {"rows": 40},
                    "historical_version_only": {"rows": 30},
                    "unknown": {"rows": 30},
                },
                "eligible_missing_start_time_rate": 1.0,
                "eligible_duration_over_24h_rows": 1,
            },
        },
    )

    result = evaluate_gate(bundle, "label-l0")

    assert result.passed


def test_label_l4_gate_is_watch_time_primary_with_completion_diagnostic(tmp_path):
    baseline_values = {
        "completion_gain_rule_ndcg@20": 0.5,
        "completion_gain_macro_content_ndcg@10": 0.2,
        "completion_gain_all_rule_macro_content_ndcg@10": 0.3,
        "watch_time_gain_rule_ndcg@20": 0.5,
        "watch_time_gain_macro_content_ndcg@10": 0.2,
        "watch_time_gain_all_rule_macro_content_ndcg@10": 0.3,
        "overall_recall@200": 0.4,
        "exact_20_normal_rule_rate": 1.0,
        "exact_10_content_rate": 1.0,
    }
    # watch_time improves >=3%; completion REGRESSES. watch_time is the primary
    # family-fair gate, so this must still pass and completion must not block.
    candidate_values = {
        **baseline_values,
        "watch_time_gain_rule_ndcg@20": 0.52,
        "watch_time_gain_macro_content_ndcg@10": 0.21,
        "completion_gain_rule_ndcg@20": 0.40,
        "completion_gain_macro_content_ndcg@10": 0.10,
    }
    profile_metrics = (
        "watch_time_gain_rule_ndcg@20",
        "watch_time_gain_macro_content_ndcg@10",
    )
    baseline = MetricBundle(
        manifest=create_manifest(tmp_path, "all", "label-l2", "baseline"),
        evaluation={
            **baseline_values,
            "per_profile": {
                "p1": {metric: baseline_values[metric] for metric in profile_metrics},
                "p2": {metric: baseline_values[metric] for metric in profile_metrics},
            },
        },
    )
    candidate = MetricBundle(
        manifest=create_manifest(tmp_path, "all", "label-l4", "candidate"),
        evaluation={
            **candidate_values,
            "per_profile": {
                "p1": {metric: candidate_values[metric] for metric in profile_metrics},
                "p2": {metric: candidate_values[metric] for metric in profile_metrics},
            },
        },
    )

    passing = evaluate_gate(candidate, "label-l4", baseline)
    assert passing.passed
    # completion_gain is a reported diagnostic, not a gate check
    assert not any("completion_gain" in check.metric for check in passing.checks)

    # watch_time all-rule retention regression still blocks
    candidate.evaluation["watch_time_gain_all_rule_macro_content_ndcg@10"] = 0.29
    retention_gate = evaluate_gate(candidate, "label-l4", baseline)
    retention_check = next(
        check
        for check in retention_gate.checks
        if check.metric
        == "evaluation.watch_time_gain_all_rule_macro_content_ndcg@10"
    )
    assert not retention_check.passed

    candidate.evaluation.update(candidate_values)
    candidate.evaluation.pop("per_profile")
    evidence_gate = evaluate_gate(candidate, "label-l4", baseline)
    assert not evidence_gate.passed
    assert any(
        check.metric.startswith("evaluation.paired_bootstrap")
        and not check.passed
        for check in evidence_gate.checks
    )


def test_label_l4_content_gate_requires_content_gain_and_rule_retention(tmp_path):
    """A content-ordering change cannot pass label-l4, which demands a rule gain.

    The content scorer reorders content inside a rule and leaves rule ordering
    byte-identical on purpose, so label-l4 fails it by construction rather than
    by any regression. This phase asks for the opposite pair:
    content must rise, rule ordering must be retained.
    """
    baseline_values = {
        "watch_time_gain_rule_ndcg@20": 0.5,
        "watch_time_gain_macro_content_ndcg@10": 0.2,
        "watch_time_gain_all_rule_macro_content_ndcg@10": 0.3,
        "overall_recall@200": 0.4,
        "exact_20_normal_rule_rate": 1.0,
        "exact_10_content_rate": 1.0,
    }
    # Content up 10%, rule ordering untouched: the content scorer's shape.
    candidate_values = {
        **baseline_values,
        "watch_time_gain_macro_content_ndcg@10": 0.22,
        "watch_time_gain_all_rule_macro_content_ndcg@10": 0.33,
        "overall_recall@200": 0.44,
    }
    profile_metrics = ("watch_time_gain_macro_content_ndcg@10",)

    def bundle(phase, values):
        return MetricBundle(
            manifest=create_manifest(tmp_path, "all", phase, phase),
            evaluation={
                **values,
                "per_profile": {
                    name: {metric: values[metric] for metric in profile_metrics}
                    for name in ("p1", "p2")
                },
            },
        )

    baseline = bundle("label-l2", baseline_values)
    candidate = bundle("label-l4-content", candidate_values)

    assert evaluate_gate(candidate, "label-l4-content", baseline).passed
    # The same evidence must NOT pass the rule-oriented phase.
    assert not evaluate_gate(candidate, "label-l4", baseline).passed

    # Trading rule ordering away for content is what retention has to block.
    candidate.evaluation["watch_time_gain_rule_ndcg@20"] = 0.45
    regressed = evaluate_gate(candidate, "label-l4-content", baseline)
    assert not regressed.passed
    assert not next(
        check
        for check in regressed.checks
        if check.metric == "evaluation.watch_time_gain_rule_ndcg@20"
    ).passed

    candidate.evaluation.update(candidate_values)
    candidate.evaluation.pop("per_profile")
    assert not evaluate_gate(candidate, "label-l4-content", baseline).passed
