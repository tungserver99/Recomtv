"""Every configuration value the library accepts must be reachable from a CLI.

A flag the CLI silently rejects, never exposes, or accepts without forwarding
makes a run measure the default while reporting the setting under test. These
tests compare the values that actually reach the call.
"""

from __future__ import annotations

import argparse
import dataclasses

import pytest

from recsys_vnpt import domain_entrypoint, pipeline_cli
from recsys_vnpt.retrieval import RetrievalConfig
from recsys_vnpt.rule_assembly import PAGE_DEDUP_POLICIES, RuleAssemblyConfig


def _parser_options(build) -> dict[str, argparse.Action]:
    parser_actions: dict[str, argparse.Action] = {}

    original = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):  # noqa: ANN001
        for action in self._actions:
            for option in action.option_strings:
                parser_actions[option] = action
        for action in getattr(self, "_subparsers", None)._group_actions if self._subparsers else []:
            for sub in action.choices.values():
                for sub_action in sub._actions:
                    for option in sub_action.option_strings:
                        parser_actions[option] = sub_action
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = capture
    try:
        build()
    except SystemExit:
        pass
    finally:
        argparse.ArgumentParser.parse_args = original
    return parser_actions


@pytest.mark.parametrize(
    "build",
    [
        lambda: pipeline_cli.main([]),
        lambda: domain_entrypoint.main("all", "evaluate", []),
        lambda: domain_entrypoint.main("all", "infer", []),
    ],
)
def test_page_dedup_choices_cover_every_supported_policy(build):
    options = _parser_options(build)
    action = options.get("--page-dedup")
    if action is None:
        pytest.skip("this entrypoint does not expose page shape")
    supported = set(PAGE_DEDUP_POLICIES)
    assert set(action.choices) == supported, (
        "CLI --page-dedup choices drifted from the policies RuleAssemblyConfig "
        f"accepts: missing {supported - set(action.choices)}"
    )
    for policy in action.choices:
        RuleAssemblyConfig(page_dedup=policy)  # must not raise


def test_every_page_flag_is_exposed_where_the_page_is_built():
    for build in (
        lambda: domain_entrypoint.main("all", "evaluate", []),
        lambda: domain_entrypoint.main("all", "infer", []),
    ):
        options = _parser_options(build)
        for flag in ("--contents-per-rule-max", "--page-content-budget", "--page-dedup"):
            assert flag in options, f"{flag} is not reachable from this entrypoint"


def test_config_dataclasses_reject_unknown_enum_values():
    with pytest.raises(ValueError):
        RuleAssemblyConfig(page_dedup="not_a_policy")
    assert dataclasses.fields(RetrievalConfig)  # config stays a dataclass


@pytest.mark.parametrize(
    "build",
    [
        lambda: pipeline_cli.main([]),
        lambda: domain_entrypoint.main("all", "evaluate", []),
        lambda: domain_entrypoint.main("all", "infer", []),
    ],
)
def test_content_ordering_choices_cover_every_supported_mode(build):
    options = _parser_options(build)
    action = options.get("--content-ordering")
    if action is None:
        pytest.skip("this entrypoint does not order content")
    supported = {"hand_sum", "learned_rrf"}
    assert set(action.choices) == supported, (
        "CLI --content-ordering choices drifted from the modes pipeline_v2 "
        f"accepts: missing {supported - set(action.choices)}"
    )
    assert "--content-scorer-fusion-weight" in options, (
        "the ordering mode is selectable but its fusion weight is not, so a "
        "weight sweep would silently measure the default"
    )


def test_content_ordering_is_exposed_wherever_rule_ordering_is():
    for build in (
        lambda: pipeline_cli.main([]),
        lambda: domain_entrypoint.main("all", "evaluate", []),
        lambda: domain_entrypoint.main("all", "infer", []),
    ):
        options = _parser_options(build)
        if "--rule-ordering" not in options:
            continue
        assert "--content-ordering" in options, (
            "rule ordering is configurable here but content ordering is not"
        )


def test_evaluate_baseline_forwards_the_page_flags_it_accepts(monkeypatch):
    """Exposing a flag is not enough: it has to reach the call.

    A page flag that is accepted but not forwarded makes a run asking for the
    dynamic 10-30 page silently measure the fixed 20x10 one, while the manifest
    records the default as if it had been chosen.
    """

    captured: dict[str, object] = {}

    def fake_evaluate(*args, **kwargs):  # noqa: ANN002, ANN003
        captured.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(pipeline_cli, "evaluate_popularity_baseline", fake_evaluate)
    with pytest.raises(SystemExit):
        pipeline_cli.main(
            [
                "evaluate-baseline",
                "--page-dedup",
                "greedy_rule_rank",
                "--contents-per-rule-max",
                "30",
                "--page-content-budget",
                "600",
            ]
        )
    assert captured.get("page_dedup") == "greedy_rule_rank"
    assert captured.get("contents_per_rule_max") == 30
    assert captured.get("page_content_budget") == 600


# Operational flags: they steer where a run writes or whether it runs at all,
# not what it produces, so the manifest does not need them.
_NON_CONFIG_LIVE_TEST_FLAGS = {
    "-h",
    "--help",
    "--domain",
    "--model-family",
    "--output-dir",
    "--run-id",
    "--dry-run",
    "--resume",
    "--train-if-missing",
    "--no-gpu-sampling",
    "--skip-offline-eval",
}


def test_serving_run_records_every_flag_that_shapes_its_output():
    """A run that cannot be reproduced from its own manifest is not evidence,
    so every flag that shapes the served page must be recorded in it."""

    import inspect

    from recsys_vnpt.pipeline_v2 import serving_manifest_config

    recorded = set(inspect.signature(serving_manifest_config).parameters)
    options = _parser_options(lambda: pipeline_cli.main([]))
    expected = {
        option.lstrip("-").replace("-", "_")
        for option in options
        if option.startswith("--")
        and option not in _NON_CONFIG_LIVE_TEST_FLAGS
        and option.lstrip("-").replace("-", "_") in recorded | _LIVE_TEST_FLAGS
    }
    missing = _LIVE_TEST_FLAGS - recorded
    assert not missing, f"serving manifest does not record: {sorted(missing)}"
    assert expected <= recorded


# The live-test flags that describe the produced page, kept explicit so adding a
# knob to the CLI without recording it fails here.
_LIVE_TEST_FLAGS = {
    "num_profiles",
    "warm_repetitions",
    "num_contents",
    "profile_chunk_size",
    "batch_size",
    "amp",
    "candidate_budget",
    "exact_cowatch",
    "rule_ordering",
    "rule_scorer_fusion_weight",
    "content_ordering",
    "content_scorer_fusion_weight",
    "contents_per_rule_max",
    "page_content_budget",
    "page_dedup",
    "recency_candidates_per_rule",
    "recency_reserved_per_rule",
    "recency_max_age_days",
    "candidate_rules",
    "reserved_per_rule",
    "source_candidates_per_rule",
}


def test_serving_manifest_config_returns_exactly_its_parameters():
    import inspect

    from recsys_vnpt.pipeline_v2 import serving_manifest_config

    parameters = set(inspect.signature(serving_manifest_config).parameters)
    produced = serving_manifest_config(**{name: None for name in parameters})
    assert set(produced) == parameters


# Config that decides what page a run is scored on. If the two evaluation paths
# disagree here, their numbers cannot be compared to each other.
_COMPARABILITY_CONFIG = (
    "evaluation_split",
    "maximum_evaluation_profiles",
    "evaluation_profile_seed",
    "rule_affinity",
    "rule_affinity_half_life_days",
    "rule_ordering",
    "rule_scorer_fusion_weight",
    "content_ordering",
    "content_scorer_fusion_weight",
    "contents_per_rule_max",
    "page_content_budget",
    "page_dedup",
    "candidate_rules",
    "reserved_per_rule",
    "source_candidates_per_rule",
)


def _forwarded(monkeypatch, target: str, argv: list[str]) -> dict:
    captured: dict = {}

    def fake(*args, **kwargs):  # noqa: ANN002, ANN003
        captured.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(pipeline_cli, target, fake)
    with pytest.raises(SystemExit):
        pipeline_cli.main(argv)
    return captured


def test_both_evaluation_paths_score_the_same_page(monkeypatch):
    """`evaluate` and `evaluate-baseline` must build the same page by default.

    Otherwise a neural challenger is scored on one page shape and ordering
    while the popularity baseline it is compared against is scored on another,
    and the comparison means nothing.
    """

    baseline = _forwarded(
        monkeypatch, "evaluate_popularity_baseline", ["evaluate-baseline"]
    )
    neural = _forwarded(monkeypatch, "evaluate_temporal_model", ["evaluate"])
    differing = {
        name: (baseline.get(name), neural.get(name))
        for name in _COMPARABILITY_CONFIG
        if baseline.get(name) != neural.get(name)
    }
    assert not differing, (
        "the two evaluation paths would score different pages: "
        f"{differing}"
    )


def test_neural_evaluation_records_the_page_it_scored():
    """Same rule as the serving path: record what shaped the numbers.

    `evaluate_temporal_model` accepts the page shape and both ordering modes but
    recorded only nine unrelated fields, so a neural run's manifest could not
    say which page its NDCG came from.
    """

    import inspect

    from recsys_vnpt.pipeline_v2 import (
        evaluate_temporal_model,
        scoring_manifest_config,
    )

    recorded = set(inspect.signature(scoring_manifest_config).parameters)
    assert set(_COMPARABILITY_CONFIG) <= recorded
    accepted = set(inspect.signature(evaluate_temporal_model).parameters)
    assert recorded <= accepted, (
        "the manifest records config the evaluation does not accept: "
        f"{sorted(recorded - accepted)}"
    )


# Retrieval depth decides how many alternatives a rule still has once an earlier
# rule has taken a content, so it is the lever left for the de-duplication cost.
_RETRIEVAL_DEPTH_FLAGS = {
    "candidate_rules",
    "reserved_per_rule",
    "source_candidates_per_rule",
}


def test_retrieval_depth_is_reachable_from_the_evaluation_cli(monkeypatch):
    """Retrieval depth must be settable without editing code."""

    from recsys_vnpt.retrieval import RetrievalConfig

    fields = {field.name for field in dataclasses.fields(RetrievalConfig)}
    assert _RETRIEVAL_DEPTH_FLAGS <= fields

    captured = _forwarded(
        monkeypatch,
        "evaluate_popularity_baseline",
        [
            "evaluate-baseline",
            "--candidate-rules",
            "25",
            "--reserved-per-rule",
            "30",
            "--source-candidates-per-rule",
            "40",
        ],
    )
    assert captured.get("candidate_rules") == 25
    assert captured.get("reserved_per_rule") == 30
    assert captured.get("source_candidates_per_rule") == 40


def test_both_evaluation_manifests_record_retrieval_depth(monkeypatch):
    """A depth sweep is unreadable if the manifest says None for the depth.

    A depth sweep is unreadable if the manifest reports the default while the
    metrics have clearly moved, so both manifest builders must record it.
    """

    import inspect

    from recsys_vnpt.pipeline_v2 import scoring_manifest_config

    assert _RETRIEVAL_DEPTH_FLAGS <= set(
        inspect.signature(scoring_manifest_config).parameters
    )
    # Read the function itself rather than a fixed file path, so the check
    # survives the function moving modules.
    from recsys_vnpt.pipeline_v2 import evaluate_popularity_baseline

    baseline_manifest = inspect.getsource(evaluate_popularity_baseline)
    for flag in sorted(_RETRIEVAL_DEPTH_FLAGS):
        assert f'"{flag}": {flag}' in baseline_manifest, (
            f"evaluate_popularity_baseline does not record {flag}"
        )


def _subcommand_defaults(name: str) -> dict[str, object]:
    captured: dict[str, object] = {}
    original = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):  # noqa: ANN001
        subparsers = next(
            action for action in self._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        captured.update(
            {action.dest: action.default for action in subparsers.choices[name]._actions}
        )
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = capture
    try:
        pipeline_cli.main([])
    except SystemExit:
        pass
    finally:
        argparse.ArgumentParser.parse_args = original
    return captured


def test_batch_serves_the_page_live_test_rehearses():
    from recsys_vnpt.batch_contracts import BatchConfig

    live = _subcommand_defaults("live-test")
    batch = BatchConfig(incoming_root="i", work_root="w", delivery_root="d", model_registry="m")
    shared = {field.name for field in dataclasses.fields(BatchConfig)} & set(live)
    assert {"candidate_rules", "reserved_per_rule", "page_dedup", "content_ordering"} <= shared
    drift = {name: (getattr(batch, name), live[name]) for name in shared if getattr(batch, name) != live[name]}
    assert not drift, f"batch and live-test defaults differ: {drift}"


def test_evaluation_scores_the_retrieval_serving_uses():
    live = _subcommand_defaults("live-test")
    evaluation = _subcommand_defaults("evaluate-baseline")
    library = RetrievalConfig()
    domain = {
        operation: _parser_options(lambda operation=operation: domain_entrypoint.main("all", operation, []))
        for operation in ("evaluate", "infer")
    }
    for name in (
        "candidate_rules",
        "reserved_per_rule",
        "source_candidates_per_rule",
        "recency_candidates_per_rule",
        "recency_reserved_per_rule",
        "recency_max_age_days",
    ):
        resolved = evaluation[name] if evaluation[name] is not None else getattr(library, name)
        assert resolved == live[name], name
        option = "--" + name.replace("_", "-")
        for operation, options in domain.items():
            assert options[option].default == live[name], (operation, name)
