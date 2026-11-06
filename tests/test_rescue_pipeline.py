import json
from pathlib import Path

import pandas as pd
import pytest

import recsys_vnpt.export as client_export
from recsys_vnpt.audit import audit_project
from recsys_vnpt.audit_cli import write_audit_report
from recsys_vnpt.baselines import PopularityBaseline
from recsys_vnpt.config import DOMAIN_CONFIGS, get_domain_config
from recsys_vnpt.data_loading import load_json_records
from recsys_vnpt.export import build_client_exports, rank_result
from recsys_vnpt.features import FeatureSchema
from recsys_vnpt.inference import filter_recommendable_items
from recsys_vnpt.labels import build_labeled_interactions, finalize_merged_interactions
from recsys_vnpt.metrics import ranking_metrics
from recsys_vnpt.pipeline import build_training_frame
from recsys_vnpt.training import prepare_feature_matrix, profile_group_split


def test_domain_configs_capture_label_policy():
    assert get_domain_config("clip").watch_percent_threshold == 0.3
    assert get_domain_config("movie").repeat_watch_positive is True
    assert get_domain_config("music").watch_percent_threshold == 1.0
    assert get_domain_config("all").uses_family_label_policy is True
    assert set(DOMAIN_CONFIGS) == {"clip", "movie", "music", "all"}


def test_build_labeled_interactions_aggregates_rows_and_records_reasons():
    interactions = pd.DataFrame(
        [
            {"profile_id": 1, "username": "u1", "content_id": "c1", "duration": 20},
            {"profile_id": 1, "username": "u1", "content_id": "c1", "duration": 25},
            {"profile_id": 1, "username": "u1", "content_id": "c2", "duration": 5},
            {"profile_id": 2, "username": "u2", "content_id": "c3", "duration": 100},
        ]
    )
    items = pd.DataFrame(
        [
            {"content_id": "c1", "content_duration": 100, "content_family": "clip"},
            {"content_id": "c2", "content_duration": 100, "content_family": "clip"},
            {"content_id": "c3", "content_duration": 100, "content_family": "movie"},
        ]
    )

    labeled = build_labeled_interactions(interactions, items, domain="clip")

    c1 = labeled[labeled["content_id"] == "c1"].iloc[0]
    c2 = labeled[labeled["content_id"] == "c2"].iloc[0]
    c3 = labeled[labeled["content_id"] == "c3"].iloc[0]
    assert c1["watch_count"] == 2
    assert c1["total_duration"] == 45
    assert c1["max_watch_percent"] == 0.25
    assert c1["label"] == 0
    assert c1["label_reason"] == "negative_low_watch"
    assert c2["sample_type"] == "observed_negative"
    assert c3["label"] == 1
    assert c3["label_reason"] == "positive_watch_percent"


def test_all_domain_uses_content_family_label_policy():
    interactions = pd.DataFrame(
        [
            {"profile_id": 1, "content_id": "clip1", "duration": 35},
            {"profile_id": 1, "content_id": "movie1", "duration": 35},
            {"profile_id": 1, "content_id": "music1", "duration": 100},
        ]
    )
    items = pd.DataFrame(
        [
            {"content_id": "clip1", "content_duration": 100, "content_family": "clip"},
            {"content_id": "movie1", "content_duration": 100, "content_family": "movie"},
            {"content_id": "music1", "content_duration": 100, "content_family": "music"},
        ]
    )

    labeled = build_labeled_interactions(interactions, items, domain="all")
    by_id = labeled.set_index("content_id")

    assert by_id.loc["clip1", "label"] == 1
    assert by_id.loc["movie1", "label"] == 0
    assert by_id.loc["music1", "label"] == 1


def test_feature_schema_preserves_train_columns_and_fills_missing_in_inference(tmp_path):
    train = pd.DataFrame(
        {
            "profile_id": [1],
            "content_id": ["c1"],
            "label": [1],
            "feature_b": [2.0],
            "feature_a": [1.0],
        }
    )
    schema = FeatureSchema.fit(train)
    assert schema.feature_columns == ["feature_b", "feature_a"]

    path = tmp_path / "schema.json"
    schema.save(path)
    loaded = FeatureSchema.load(path)
    inference = loaded.transform(pd.DataFrame({"feature_a": [5.0], "unused": [99]}))

    assert list(inference.columns) == ["feature_b", "feature_a"]
    assert inference.iloc[0].to_dict() == {"feature_b": 0.0, "feature_a": 5.0}
    assert json.loads(path.read_text())["feature_columns"] == ["feature_b", "feature_a"]


def test_ranking_metrics_report_recommender_quality():
    recommendations = {
        "u1": ["a", "b", "c"],
        "u2": ["d", "e"],
    }
    ground_truth = {
        "u1": {"b", "x"},
        "u2": {"d"},
        "u3": {"z"},
    }

    metrics = ranking_metrics(recommendations, ground_truth, k_values=(1, 2))

    assert metrics["users_evaluated"] == 2
    assert metrics["recall@1"] == 0.5
    assert metrics["hit_rate@1"] == 0.5
    assert metrics["recall@2"] == 0.75
    assert metrics["map@2"] == 0.625
    assert round(metrics["ndcg@2"], 6) == 0.693426


def test_client_exports_keep_existing_shape_and_skip_unmatched_rules():
    result = {
        "p1": {
            "suggested_content": {
                "c1": {"type_id": "12", "rule_id": "132", "score": 0.9},
                "c2": {"type_id": "7", "rule_id": -100, "score": 0.8},
                "c3": {"type_id": "78", "rule_id": "148", "score": 0.7},
                "c4": {"type_id": "12", "rule_id": "132", "score": 0.6},
            },
            "user": {"username": "u1", "profile_id": "p1"},
        }
    }

    rulename, rule_content = build_client_exports(result)

    assert rulename == [{"pid": "p1", "d": "132,148"}]
    assert rule_content == "p1|132,12#c1$12#c4;148,78#c3\n"


def _assembled_recommendations(label):
    return {
        "p1": {
            "normal_rules": [
                {
                    "rule_id": f"{label}-rule-{index}",
                    "contents": [
                        {"type_id": "1", "content_id": f"{label}-{index}-{item}"}
                        for item in range(10)
                    ],
                }
                for index in range(20)
            ],
            "special_rules": ["112", "109", "108", "110"],
        }
    }


def test_client_output_manifest_commits_and_reads_immutable_generations(tmp_path):
    output = tmp_path / "client_output"
    client_export.atomic_write_client_outputs(output, _assembled_recommendations("old"))
    old_snapshot = client_export.read_committed_client_output(output)

    client_export.atomic_write_client_outputs(output, _assembled_recommendations("new"))
    new_snapshot = client_export.read_committed_client_output(output)
    reloaded_old = client_export.read_client_output_generation(output, old_snapshot.manifest)

    assert old_snapshot.homepage[0]["d"].startswith("old-rule-")
    assert new_snapshot.homepage[0]["d"].startswith("new-rule-")
    assert reloaded_old.homepage == old_snapshot.homepage
    assert set(new_snapshot.manifest["files"]) == {"rulename.json", "rule_content.txt"}
    assert (output / "rulename.json").read_text(encoding="utf-8") == json.dumps(
        new_snapshot.homepage, indent=2, ensure_ascii=False
    )


@pytest.mark.parametrize("failure_point", ["generation", "mirror", "manifest"])
def test_failed_client_publication_keeps_prior_manifest_generation_readable(
    tmp_path, monkeypatch, failure_point
):
    output = tmp_path / "client_output"
    client_export.atomic_write_client_outputs(output, _assembled_recommendations("old"))
    previous = client_export.read_committed_client_output(output)
    real_replace = client_export.os.replace

    def fail_before_commit(source, destination):
        source_name = Path(source).name
        if (
            failure_point == "generation" and source_name.endswith(".generation.stage")
        ) or (
            failure_point == "mirror" and ".mirror.stage" in source_name
        ) or (
            failure_point == "manifest"
            and source_name.startswith("client_output_manifest.json.")
            and source_name.endswith(".stage")
        ):
            raise OSError(f"injected {failure_point} failure")
        return real_replace(source, destination)

    monkeypatch.setattr(client_export.os, "replace", fail_before_commit)

    with pytest.raises(OSError, match=f"injected {failure_point} failure"):
        client_export.atomic_write_client_outputs(output, _assembled_recommendations("new"))

    assert client_export.read_committed_client_output(output) == previous


def test_rank_result_sorts_each_profile_and_keeps_top_n():
    data = {
        "p1": {
            "suggested_content": {
                "low": {"content_name": "l", "tag_names": "", "type_id": "1", "score": 0.1},
                "high": {"content_name": "h", "tag_names": "", "type_id": "1", "score": 0.9},
            },
            "user": {"profile_id": "p1"},
        }
    }

    ranked = rank_result(data, 1)

    assert list(ranked["p1"]["suggested_content"]) == ["high"]


def test_filter_recommendable_items_excludes_invalid_or_consumed_items():
    items = pd.DataFrame(
        [
            {"content_id": "ok", "content_status": "1", "tag_names": "Sport", "recommendable": True},
            {"content_id": "inactive", "content_status": "0", "tag_names": "Sport", "recommendable": True},
            {"content_id": "no_tags", "content_status": "1", "tag_names": "", "recommendable": True},
            {"content_id": "excluded", "content_status": "1", "tag_names": "Sport", "recommendable": False},
            {"content_id": "seen", "content_status": "1", "tag_names": "Sport", "recommendable": True},
        ]
    )

    filtered = filter_recommendable_items(items, consumed_content_ids={"seen"})

    assert filtered["content_id"].tolist() == ["ok"]


def test_audit_project_reports_missing_artifacts_and_rule_summary(tmp_path):
    rule_info = pd.DataFrame(
        [
            {
                "PARENT_RULE_ID": "32",
                "PARENT_RULE_NAME": "Kids",
                "RULE_ID": "144",
                "RULE_NAME": "Kids",
                "LIST_TAGS_STR": "Kids",
                "LIST_TAGS_ID": "3",
                "RULE_TYPE": "1",
                "FILENAME_DATE": "2025-06-11",
            }
        ]
    )
    rule_info.to_parquet(tmp_path / "rule_info.parquet", index=False)

    report = audit_project(tmp_path)

    assert "duration" in report.missing_artifacts
    assert report.rule_info_rows == 1
    assert report.rule_id_count == 1
    assert report.content_status["clip"].status == "excluded"
    assert report.content_status["clip"].reason == "missing content source"


def test_write_audit_report_creates_markdown_file(tmp_path):
    output = tmp_path / "AUDIT_REPORT.md"

    write_audit_report(root=tmp_path, output=output)

    text = output.read_text(encoding="utf-8")
    assert "# Current State Audit" in text
    assert "`duration`" in text
    assert "`clip`: excluded" in text


def test_load_json_records_reports_skipped_files_without_silent_pass(tmp_path):
    good = tmp_path / "good.json"
    bad = tmp_path / "bad.json"
    good.write_text(json.dumps([{"content_id": "c1"}]), encoding="utf-8")
    bad.write_text("{broken", encoding="utf-8")

    frame, skipped = load_json_records([good, bad])

    assert frame.to_dict("records") == [{"content_id": "c1"}]
    assert len(skipped) == 1
    assert skipped[0].path == bad
    assert "Expecting" in skipped[0].reason or "property name" in skipped[0].reason


def test_build_training_frame_merges_users_items_and_labels():
    interactions = pd.DataFrame(
        [{"profile_id": 1, "username": "u1", "content_id": "c1", "duration": 40}]
    )
    users = pd.DataFrame([{"username": "u1", "age": 30, "package_code_MYTV021": 1}])
    items = pd.DataFrame(
        [{"content_id": "c1", "content_duration": 100, "type_id_12": 1, "content_family": "clip"}]
    )

    frame = build_training_frame(interactions, users, items, domain="clip")

    assert frame.loc[0, "label"] == 1
    assert frame.loc[0, "age"] == 30
    assert frame.loc[0, "type_id_12"] == 1


def test_finalize_merged_interactions_preserves_features_and_aggregates_labels():
    merged = pd.DataFrame(
        [
            {
                "profile_id": 1,
                "username": "u1",
                "content_id": "c1",
                "duration": 35,
                "content_duration": 100,
                "content_family": "clip",
                "age": 30,
                "type_id_12": 1,
            },
            {
                "profile_id": 1,
                "username": "u1",
                "content_id": "c1",
                "duration": 5,
                "content_duration": 100,
                "content_family": "clip",
                "age": 30,
                "type_id_12": 1,
            },
        ]
    )

    labeled = finalize_merged_interactions(merged, domain="all")

    assert len(labeled) == 1
    assert labeled.loc[0, "watch_count"] == 2
    assert labeled.loc[0, "total_duration"] == 40
    assert labeled.loc[0, "label"] == 1
    assert labeled.loc[0, "label_reason"] == "positive_watch_percent"
    assert labeled.loc[0, "age"] == 30
    assert labeled.loc[0, "type_id_12"] == 1


def test_prepare_feature_matrix_round_trips_schema_and_labels():
    frame = pd.DataFrame(
        {
            "profile_id": [1, 2],
            "content_id": ["c1", "c2"],
            "label": [1, 0],
            "feature": [0.5, 1.5],
        }
    )

    matrix, labels, schema = prepare_feature_matrix(frame)

    assert matrix.shape == (2, 1)
    assert labels.tolist() == [1.0, 0.0]
    assert schema.feature_columns == ["feature"]


def test_profile_group_split_keeps_profiles_in_one_side():
    frame = pd.DataFrame(
        {
            "profile_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "content_id": list("abcdefgh"),
            "label": [1, 0, 1, 0, 1, 0, 1, 0],
        }
    )

    train, val = profile_group_split(frame, test_size=0.5, random_state=7)

    assert set(train["profile_id"]).isdisjoint(set(val["profile_id"]))
    assert len(train) + len(val) == len(frame)


def test_popularity_baseline_recommends_unconsumed_items_by_positive_count():
    training = pd.DataFrame(
        [
            {"content_id": "a", "label": 1},
            {"content_id": "a", "label": 1},
            {"content_id": "b", "label": 1},
            {"content_id": "c", "label": 0},
        ]
    )
    candidates = pd.DataFrame([{"content_id": "a"}, {"content_id": "b"}, {"content_id": "c"}])

    model = PopularityBaseline().fit(training)
    recs = model.recommend(candidates, consumed_content_ids={"a"}, top_n=2)

    assert recs == ["b", "c"]
