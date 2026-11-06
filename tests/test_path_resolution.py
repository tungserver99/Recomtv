
from recsys_vnpt.paths import (
    data_artifact_path,
    data_root,
    model_artifact_path,
    rule_info_path,
    source_artifact_path,
    source_root,
)


def test_data_root_defaults_to_nested_data_data_when_present(tmp_path, monkeypatch):
    monkeypatch.delenv("RECSYS_VNPT_DATA_ROOT", raising=False)
    nested = tmp_path / "data" / "data"
    nested.mkdir(parents=True)

    assert data_root(tmp_path) == nested
    assert data_artifact_path("duration", project_root=tmp_path) == nested / "duration"
    assert model_artifact_path("clip", "best_model.pth", project_root=tmp_path) == nested / "model" / "clip" / "best_model.pth"


def test_data_root_can_be_overridden_by_environment(tmp_path, monkeypatch):
    custom = tmp_path / "external" / "mytv"
    monkeypatch.setenv("RECSYS_VNPT_DATA_ROOT", str(custom))

    assert data_root(tmp_path) == custom


def test_source_root_follows_data_root_unless_separated(tmp_path, monkeypatch):
    monkeypatch.delenv("RECSYS_VNPT_SOURCE_ROOT", raising=False)
    monkeypatch.setenv("RECSYS_VNPT_DATA_ROOT", str(tmp_path / "artifacts"))
    assert source_root(tmp_path) == data_root(tmp_path)

    monkeypatch.setenv("RECSYS_VNPT_SOURCE_ROOT", str(tmp_path / "incoming"))
    assert source_artifact_path("duration", project_root=tmp_path) == tmp_path / "incoming" / "duration"
    assert data_artifact_path("canonical", project_root=tmp_path) == tmp_path / "artifacts" / "canonical"


def test_rule_info_prefers_data_root_but_falls_back_to_project_root(tmp_path, monkeypatch):
    monkeypatch.delenv("RECSYS_VNPT_DATA_ROOT", raising=False)
    monkeypatch.delenv("RECSYS_VNPT_SOURCE_ROOT", raising=False)
    nested = tmp_path / "data" / "data"
    nested.mkdir(parents=True)
    fallback = tmp_path / "rule_info.parquet"
    fallback.write_text("placeholder", encoding="utf-8")

    assert rule_info_path(tmp_path) == fallback

    preferred = nested / "rule_info.parquet"
    preferred.write_text("placeholder", encoding="utf-8")

    assert rule_info_path(tmp_path) == preferred
