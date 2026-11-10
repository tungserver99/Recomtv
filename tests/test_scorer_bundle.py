import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from recsys_vnpt.content_scorer import ContentScorerModel
from recsys_vnpt.rule_scorer import RuleScorerModel
from recsys_vnpt.scorer_bundle import (
    BundleError,
    load_bundle,
    promote,
    promoted_bundle,
    save_bundle,
)


def _head(seed):
    x = np.arange(40, dtype=np.float32).reshape(20, 2)
    return HistGradientBoostingRegressor(max_iter=5, random_state=seed).fit(x, x[:, 0])


def _models():
    rule = RuleScorerModel(
        heads={"completion_gain": _head(1)},
        rule_index=pd.DataFrame({"rule_id": ["1", "2"], "rule_index": [0, 1]}),
        metrics={"rule_scorer_training_rows": 20},
    )
    content = ContentScorerModel(
        heads={"completion_gain": _head(2)},
        level_maps={"type_index": {"1": 0}},
        metrics={"content_scorer_training_rows": 20},
    )
    return rule, content


def test_bundle_round_trips_and_predicts_identically(tmp_path):
    rule, content = _models()
    bundle = save_bundle(tmp_path, rule, content, {"cutoff": "2025-01-31"})

    loaded_rule, loaded_content, manifest = load_bundle(bundle)

    x = np.ones((3, 2), dtype=np.float32)
    assert np.array_equal(
        loaded_rule.heads["completion_gain"].predict(x),
        rule.heads["completion_gain"].predict(x),
    )
    assert loaded_content.level_maps == content.level_maps
    assert manifest["cutoff"] == "2025-01-31"
    assert manifest["bundle_id"] == bundle.name


def test_corrupt_or_foreign_bundle_is_refused(tmp_path):
    bundle = save_bundle(tmp_path, *_models(), {})
    (bundle / "scorers.joblib").write_bytes(b"tampered")
    with pytest.raises(BundleError, match="corrupt"):
        load_bundle(bundle)

    other = save_bundle(tmp_path / "other", *_models(), {})
    manifest = json.loads((other / "bundle.json").read_text())
    manifest["libraries"]["scikit-learn"] = "0.0.0"
    (other / "bundle.json").write_text(json.dumps(manifest))
    with pytest.raises(BundleError, match="retrain"):
        load_bundle(other)


def test_promotion_keeps_the_previous_pointer_for_rollback(tmp_path):
    registry = tmp_path / "registry"
    assert promoted_bundle(registry) is None
    first = save_bundle(tmp_path / "models", *_models(), {})
    second = save_bundle(tmp_path / "models2", *_models(), {})

    promote(registry, first, {"passed": True})
    promote(registry, second, {"passed": True})

    assert promoted_bundle(registry) == second.resolve()
    previous = json.loads((registry / "previous_model.json").read_text())
    assert previous["bundle_path"] == str(first.resolve())


def test_bundle_copies_and_verifies_optional_fcn_artifacts(tmp_path):
    checkpoint = tmp_path / "best_model_v2.pth"
    contract = tmp_path / "feature_contract_v2.json"
    checkpoint.write_bytes(b"synthetic checkpoint artifact")
    contract.write_text('{"model_family":"fcn_official"}', encoding="utf-8")

    bundle = save_bundle(
        tmp_path / "models",
        *_models(),
        {"scoring_config": {"preset": "GxEp"}},
        fcn_checkpoint_path=checkpoint,
        fcn_contract_path=contract,
    )

    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    artifacts = manifest["fcn_artifacts"]
    assert (bundle / artifacts["checkpoint_file"]).read_bytes() == checkpoint.read_bytes()
    assert (bundle / artifacts["contract_file"]).read_bytes() == contract.read_bytes()
    assert len(artifacts["checkpoint_sha256"]) == 64
    assert len(artifacts["contract_sha256"]) == 64
    assert load_bundle(bundle)[2]["scoring_config"]["preset"] == "GxEp"

    (bundle / artifacts["contract_file"]).write_text("{}", encoding="utf-8")
    with pytest.raises(BundleError, match="FCN artifact.*corrupt"):
        load_bundle(bundle)


def test_bundle_rejects_only_one_fcn_artifact(tmp_path):
    checkpoint = tmp_path / "best_model_v2.pth"
    checkpoint.write_bytes(b"synthetic checkpoint artifact")
    with pytest.raises(BundleError, match="both FCN checkpoint and contract"):
        save_bundle(
            tmp_path / "models",
            *_models(),
            {},
            fcn_checkpoint_path=checkpoint,
        )


def test_fcn_checkpoint_is_part_of_the_bundle_identity(tmp_path):
    contract = tmp_path / "feature_contract_v2.json"
    contract.write_text('{"model_family":"fcn_official"}', encoding="utf-8")
    checkpoint_a = tmp_path / "checkpoint-a.pth"
    checkpoint_b = tmp_path / "checkpoint-b.pth"
    checkpoint_a.write_bytes(b"weights-a")
    checkpoint_b.write_bytes(b"weights-b")
    rule, content = _models()

    bundle_a = save_bundle(
        tmp_path / "models-a", rule, content, {"scoring_config": {"preset": "GxEp"}},
        fcn_checkpoint_path=checkpoint_a, fcn_contract_path=contract,
    )
    bundle_b = save_bundle(
        tmp_path / "models-b", rule, content, {"scoring_config": {"preset": "GxEp"}},
        fcn_checkpoint_path=checkpoint_b, fcn_contract_path=contract,
    )
    manifest_a = json.loads((bundle_a / "bundle.json").read_text(encoding="utf-8"))
    manifest_b = json.loads((bundle_b / "bundle.json").read_text(encoding="utf-8"))

    assert manifest_a["model_sha256"] == manifest_b["model_sha256"]
    assert manifest_a["bundle_sha256"] != manifest_b["bundle_sha256"]
    assert bundle_a.name != bundle_b.name
