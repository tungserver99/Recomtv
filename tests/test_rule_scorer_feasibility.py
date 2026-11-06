from experiments.rule_scorer_feasibility import (
    _cohort_quality_deltas,
    _production_provenance_gate,
)


def test_rule_scorer_cohort_gate_includes_rule_and_content_quality() -> None:
    comparison = {
        "activity_bucket": {
            "cold": {
                "profiles": 10,
                "completion_gain_rule_ndcg@20.delta": 0.03,
                "watch_time_gain_rule_ndcg@20.delta": 0.02,
                "completion_gain_macro_content_ndcg@10.delta": -0.008,
                "watch_time_gain_macro_content_ndcg@10.delta": -0.007,
            },
            "empty": {
                "profiles": 0,
                "completion_gain_rule_ndcg@20.delta": -1.0,
                "completion_gain_macro_content_ndcg@10.delta": -1.0,
            },
        }
    }

    deltas = _cohort_quality_deltas(comparison)

    assert min(deltas) == -0.008
    assert -1.0 not in deltas


def test_rule_scorer_provenance_gate_reads_evaluation_and_training_scopes() -> None:
    gate, blockers = _production_provenance_gate(
        quality_gate_has_eligible_variant=True,
        evaluation_scope_metrics={
            "rule_membership_backfill": True,
            "metadata_backfill_rate": 0.0,
        },
        training_scope_metrics={
            "train_1": {"metadata_backfill_rate": 1.0},
        },
    )

    assert gate == {
        "experiment_quality_gate_has_eligible_variant": True,
        "historical_rule_membership_available": False,
        "historical_mutable_metadata_available": False,
        "passed": False,
    }
    assert len(blockers) == 2

    clean_gate, clean_blockers = _production_provenance_gate(
        quality_gate_has_eligible_variant=True,
        evaluation_scope_metrics={
            "rule_membership_backfill": False,
            "metadata_backfill_rate": 0.0,
        },
        training_scope_metrics={
            "train_1": {"metadata_backfill_rate": 0.0},
        },
    )

    assert clean_gate["passed"] is True
    assert clean_blockers == []
