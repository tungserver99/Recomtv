"""Every measured combination is reachable from config and builds a page."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import recsys_vnpt.batch_runner as runner
from recsys_vnpt import orderings
from recsys_vnpt.batch_contracts import BatchConfig, BatchError, load_config
from recsys_vnpt.export import read_current_delivery
from recsys_vnpt.output_validation import validate_client_output
from recsys_vnpt.smoke_fixture import write_smoke_batch

ROOTS = {"incoming_root": "i", "work_root": "w", "delivery_root": "d", "model_registry": "m"}


def _pairs():
    return pd.DataFrame(
        {
            "profile_id": ["p1"] * 4 + ["p2"] * 2,
            "content_id": ["a", "b", "c", "d", "a", "c"],
            "candidate_rule_id": ["1", "1", "2", "2", "1", "2"],
            "rule_score": [1.0, 1.0, 0.5, 0.5, 1.0, 0.5],
            "retrieval_score": [1.4, 1.2, 0.9, 0.6, 1.3, 0.8],
        }
    )


@pytest.mark.parametrize("name", sorted(orderings.COMBINATIONS))
def test_every_measured_combination_is_a_valid_batch_config(tmp_path, name):
    path = tmp_path / "batch.json"
    path.write_text(json.dumps({**ROOTS, "combination": name}), encoding="utf-8")
    config = load_config(path)
    for key, value in orderings.COMBINATIONS[name].items():
        assert getattr(config, key) == value


def test_a_combination_that_contradicts_the_config_is_refused(tmp_path):
    path = tmp_path / "batch.json"
    path.write_text(json.dumps({**ROOTS, "combination": "GxGp", "content_rank_scope": "rule"}), encoding="utf-8")
    with pytest.raises(BatchError, match="content_rank_scope"):
        load_config(path)
    path.write_text(json.dumps({**ROOTS, "combination": "nope"}), encoding="utf-8")
    with pytest.raises(BatchError, match="unknown combination"):
        load_config(path)


def test_the_default_is_the_production_combination():
    config = BatchConfig(**ROOTS)
    for key, value in orderings.COMBINATIONS["GxGp"].items():
        assert getattr(config, key) == value


def test_profile_scope_drops_the_rule_offset_and_ranks_across_rules():
    pairs = _pairs()
    signal = np.array([0.0, 0.0, 0.0, 9.0, 0.0, 0.0])
    profile = orderings.rrf_fuse(pairs, [signal], "profile", 5.0)
    rule = orderings.rrf_fuse(pairs, [signal], "rule", 5.0)
    rr = lambda rank: 1.0 / (60.0 + rank)
    # d has p1's lowest hand-built score, in its rule and in the profile.
    assert profile[3] == pytest.approx(rr(4) + 5.0 * rr(1))
    assert rule[3] == pytest.approx(0.5 + rr(2) + 5.0 * rr(1))
    # Ranked over the profile, the signal lifts d above every other content.
    assert profile[3] == profile[:4].max()


def _fake_fcn(monkeypatch, scores):
    class Contract:
        categorical_features = {"profile_id": type("V", (), {"vocabulary": ["p1"]})()}

    monkeypatch.setattr(orderings, "score_fcn", lambda fcn, scope, pairs, outputs: {o: scores for o in outputs})
    return {"fcn_official": orderings.FcnModel("fcn_official", None, Contract())}


def test_routed_ordering_sends_only_profiles_in_the_fcn_vocabulary_to_fcn(monkeypatch):
    pairs = _pairs()
    fcn_scores = np.array([0.0, 0.0, 0.0, 9.0, 9.0, 0.0], dtype=np.float32)
    parts = orderings.OrderingParts(fcn=_fake_fcn(monkeypatch, fcn_scores))
    monkeypatch.setattr(
        "recsys_vnpt.content_scorer.apply_content_scorer",
        lambda model, scope, frame, weight, rank_scope: (frame.assign(retrieval_score=-1.0), {}),
    )
    config = orderings.OrderingConfig(
        rule_ordering="count", content_ordering="routed", content_rank_scope="profile"
    )
    ordered, _, metrics = orderings.apply_orderings(
        config, parts, pairs["profile_id"].drop_duplicates(), pairs, pd.DataFrame(), pd.DataFrame()
    )
    expected = orderings.rrf_fuse(pairs, [fcn_scores], "profile", 5.0)
    assert ordered["retrieval_score"].iloc[:4].tolist() == pytest.approx(expected[:4].tolist())
    assert ordered["retrieval_score"].iloc[4:].tolist() == [-1.0, -1.0]
    assert metrics["routed_profile_rate"] == 0.5


def test_segment_ordering_reorders_only_cold_profiles():
    pairs = _pairs()
    scope = orderings.SegmentScope(
        warm=frozenset({"p1"}),
        attributes=pd.DataFrame({"package_code": ["x", "x"]}, index=["p1", "p2"]),
        contents={"package_code": pd.DataFrame({"package_code": ["x"], "content_id": ["c"], "norm": [1.0]})},
        rules={"package_code": pd.Series([1.0], index=pd.MultiIndex.from_tuples([("x", "2")]))},
        rule_global=pd.Series({"1": 0.2, "2": 0.0}),
        global_norm=pd.Series({"a": 0.0, "c": 1.0}),
    )
    scores = orderings.segment_content_scores(scope, pairs)
    assert scores[:4].tolist() == pairs["retrieval_score"].iloc[:4].tolist()
    assert scores[5] > scores[4]  # p2 is cold: its segment's favourite comes first
    rules = orderings.segment_rule_scores(
        scope,
        pd.DataFrame({"profile_id": ["p1", "p1", "p2", "p2"], "rule_id": ["1", "2", "1", "2"], "rule_score": [0.9, 0.1, 0.9, 0.1]}),
    )
    assert rules["rule_score"].tolist()[:2] == [0.9, 0.1]
    assert rules["rule_score"].tolist()[2:] == pytest.approx([0.2, 1.0])


def _config(tmp_path, **settings) -> BatchConfig:
    return BatchConfig(
        incoming_root=str(tmp_path / "incoming"),
        work_root=str(tmp_path / "work"),
        delivery_root=str(tmp_path / "delivery"),
        model_registry=str(tmp_path / "registry"),
        min_free_bytes=0,
        input_wait_seconds=0,
        num_shards=2,
        **settings,
    )


def _published_pages_are_valid(config) -> None:
    current = read_current_delivery(config.delivery_root)
    directory = Path(current["directory"])
    names = sorted(name for name in current["files"] if name.startswith("f_"))
    validation = validate_client_output(
        json.loads((directory / names[0]).read_text(encoding="utf-8")),
        (directory / names[1]).read_text(encoding="utf-8"),
        config=config.assembly,
    )
    assert validation.passed, validation.errors[:3]


def test_the_batch_retrains_and_serves_both_fcn_models(tmp_path):
    """FCN rows, and contents from the GBDT + FCN ensemble ranked over the
    profile: both FCN families and both GBDT scorers in one bundle."""
    write_smoke_batch(tmp_path / "incoming", "b1")
    config = _config(
        tmp_path, rule_ordering="fcn", content_ordering="ensemble", gbdt_backend="sklearn", fcn_epochs=1
    )
    retrained = runner.run_retrain_batch(config, "b1")
    assert retrained["status"] == "promoted"
    manifest = json.loads((Path(retrained["bundle"]) / "bundle.json").read_text(encoding="utf-8"))
    assert sorted(manifest["fcn"]) == ["fcn_official", "hierarchical_fcn"]
    assert manifest["ordering"]["content_ordering"] == "ensemble"

    report = runner.run_inference_batch(config, "b1")
    assert report["status"] == "published"
    assert report["model_sha256"] == manifest["bundle_sha256"]
    _published_pages_are_valid(config)


def test_a_bundle_without_the_fcn_the_ordering_needs_is_refused(tmp_path):
    write_smoke_batch(tmp_path / "incoming", "b1")
    runner.run_retrain_batch(_config(tmp_path, gbdt_backend="sklearn"), "b1")
    with pytest.raises(BatchError, match="MODEL_UNAVAILABLE"):
        runner.run_inference_batch(_config(tmp_path, content_ordering="fcn_rrf", gbdt_backend="sklearn"), "b1")


def test_the_segment_heuristic_serves_without_a_bundle(tmp_path):
    write_smoke_batch(tmp_path / "incoming", "b1")
    config = _config(tmp_path, rule_ordering="segment", content_ordering="segment")
    assert runner.run_inference_batch(config, "b1")["status"] == "published"
    _published_pages_are_valid(config)
