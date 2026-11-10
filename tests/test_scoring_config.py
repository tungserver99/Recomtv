import pytest

from recsys_vnpt.scoring_config import resolve_scoring_config


def test_legacy_config_keeps_per_rule_fusion_and_old_components():
    config = resolve_scoring_config()

    assert config.content_backends == ("heuristic", "gbdt")
    assert config.rank_scope == "per_rule"
    assert config.weights == {"heuristic": 1.0, "gbdt": 5.0, "fcn": 1.0}


def test_presets_only_change_the_content_components_and_rank_scope():
    gxgp = resolve_scoring_config(preset="GxGp")
    gxep = resolve_scoring_config(preset="GxEp")

    assert gxgp.content_backends == ("heuristic", "gbdt")
    assert gxgp.rank_scope == "profile"
    assert gxep.content_backends == ("heuristic", "gbdt", "fcn")
    assert gxep.rank_scope == "profile"


def test_custom_components_and_scope_are_validated():
    config = resolve_scoring_config(
        preset="custom",
        content_backends=("heuristic", "fcn"),
        rank_scope="profile",
        weights={"heuristic": 1.0, "fcn": 0.5},
    )
    assert config.content_backends == ("heuristic", "fcn")
    assert config.weights["fcn"] == 0.5

    with pytest.raises(ValueError, match="content backend"):
        resolve_scoring_config(preset="custom", content_backends=("mystery",))
    with pytest.raises(ValueError, match="rank scope"):
        resolve_scoring_config(preset="custom", content_backends=("heuristic",), rank_scope="global")
