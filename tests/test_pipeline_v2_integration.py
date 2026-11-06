import json

import pandas as pd
import pytest

import recsys_vnpt.pipeline_v2 as pipeline_v2
import recsys_vnpt.serving as serving
from recsys_vnpt.output_validation import OutputValidationResult
from recsys_vnpt.pipeline_v2 import (
    _evaluation_seen_entities,
    _partition_serving_interaction_scopes,
    build_temporal_project,
    evaluate_popularity_baseline,
    popularity_live_test,
    prepare_canonical_project,
)


def test_evaluation_seen_entities_use_observed_history_and_validation_for_test(
    tmp_path,
) -> None:
    temporal = tmp_path / "temporal"
    temporal.mkdir()
    pd.DataFrame(
        {
            "profile_id": ["train-profile", "sampled-profile"],
            "content_id": ["train-content", "sampled-content"],
            "sample_type": [
                "observed_engagement",
                "unobserved_uniform_negative",
            ],
        }
    ).to_parquet(temporal / "train_1.parquet", index=False)
    pd.DataFrame(
        {
            "profile_id": ["validation-profile", "validation-sampled"],
            "content_id": ["validation-content", "validation-negative"],
            "sample_type": [
                "observed_engagement",
                "unobserved_same_rule_hard_negative",
            ],
        }
    ).to_parquet(temporal / "validation.parquet", index=False)

    validation_profiles, validation_contents, validation_metrics = (
        _evaluation_seen_entities(temporal, "validation")
    )
    test_profiles, test_contents, test_metrics = _evaluation_seen_entities(
        temporal, "test"
    )

    assert validation_profiles == {"train-profile"}
    assert validation_contents == {"train-content"}
    assert validation_metrics["seen_history_windows"] == ["train_1"]
    assert test_profiles == {"train-profile", "validation-profile"}
    assert test_contents == {"train-content", "validation-content"}
    assert test_metrics["seen_history_windows"] == ["train_1", "validation"]
    assert "sampled-profile" not in test_profiles
    assert "validation-negative" not in test_contents


def test_truth_audit_interactions_do_not_enter_serving_scope() -> None:
    loaded = pd.DataFrame(
        {
            "profile_id": ["p1", "p1", "p2"],
            "content_id": ["serving", "future-truth-only", "serving"],
            "duration": [10.0, 999.0, 20.0],
        }
    )

    serving, audit, metrics = _partition_serving_interaction_scopes(
        loaded, {"serving"}
    )

    assert set(serving["content_id"]) == {"serving"}
    assert set(audit["content_id"]) == {"serving", "future-truth-only"}
    assert metrics["truth_audit_only_interaction_rows"] == 1
    assert metrics["truth_audit_only_interaction_contents"] == 1
    assert metrics["serving_interaction_rows"] == 2


def _write_synthetic_raw(root):
    data = root / "data" / "data"
    data.mkdir(parents=True)
    pd.DataFrame(
        {
            "username": ["u1", "u2"],
            "province": ["01", "02"],
            "tentinh": ["A", "B"],
            "sex": [1, 0],
            "birthday": ["19900101", "19950101"],
            "package_code": ["P1", "P2"],
        }
    ).to_parquet(data / "month_mytv_info.parquet", index=False)

    duration = data / "duration"
    duration.mkdir()
    rows = []
    dates = ["2024-12-20", "2025-01-05", "2025-01-15", "2025-01-25", "2025-02-05", "2025-02-15"]
    for profile_index, profile_id in enumerate(("p1", "p2")):
        for date_index, date in enumerate(dates):
            rows.append(
                {
                    "profile_id": profile_id,
                    "username": f"u{profile_index + 1}",
                    "member_id": f"m{profile_index + 1}",
                    "content_id": f"c{date_index + profile_index}",
                    "duration": 10,
                    "create_date": f"{date} 12:00:00",
                    "filename_date": date.replace("-", ""),
                }
            )
    pd.DataFrame(rows).to_csv(duration / "duration1.csv", index=False)

    content_dir = data / "mytv_vmp_content" / "20250101"
    content_dir.mkdir(parents=True)
    content_rows = [
        {
            "content_id": f"c{index}",
            "content_name": f"Content {index}",
            "type_id": 2,
            "content_duration": 10,
            "content_status": 1,
            "content_publish_year": 2025,
            "VOD_CODE": "AVOD",
        }
        for index in range(24)
    ]
    (content_dir / "content_movie_20250101_000000.json").write_text(
        json.dumps(content_rows), encoding="utf-8"
    )
    (content_dir / "album_20250101_000000.json").write_text(
        json.dumps([{"content_id": "unsupported-album"}]), encoding="utf-8"
    )

    tag_dir = data / "tags" / "20250311"
    tag_dir.mkdir(parents=True)
    (tag_dir / "TAG.json").write_text(
        json.dumps([{"TAG_ID": 1, "TAG_NAME": "eligible", "STATUS": 1}]),
        encoding="utf-8",
    )
    content_tag_dir = data / "content_tags" / "20250311"
    content_tag_dir.mkdir(parents=True)
    (content_tag_dir / "CONTENT_TO_TAG.json").write_text(
        json.dumps([{"CONTENT_ID": f"c{index}", "TAG_ID": 1} for index in range(24)]),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "RULE_ID": [str(index) for index in range(1, 21)],
            "LIST_TAGS_ID": ["1"] * 20,
        }
    ).to_parquet(root / "rule_info.parquet", index=False)


def test_synthetic_raw_to_temporal_metrics_and_client_output(tmp_path):
    _write_synthetic_raw(tmp_path)
    paths, data_metrics = prepare_canonical_project(
        tmp_path,
        "all",
        include_behavior=False,
    )
    assert all(path.exists() for path in paths.values())
    assert data_metrics["duplicate_canonical_ids"] == 0
    assert data_metrics["eligible_normal_rules"] == 20
    assert data_metrics["silent_skipped_files"] == 0
    assert data_metrics["reported_skipped_files"] == 1
    assert data_metrics["member_ids"] == 2
    assert data_metrics["profiles_with_multiple_member_ids"] == 0
    manifest = json.loads(
        (paths["contents"].parent / "source_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["skipped"][0]["reason"] == "unsupported_content_family"

    temporal = build_temporal_project(tmp_path, "all")
    assert temporal["test_candidates"].exists()
    baseline = evaluate_popularity_baseline(tmp_path, "all")
    assert "rule_ndcg@20" in baseline.evaluation
    assert baseline.inference["inference_pairs"] > 0
    assert baseline.data["target_version"] == "engagement_v1"
    assert baseline.data["evaluation_truth_pairs_hash"]
    assert baseline.data["evaluation_profile_selection_hash"]
    assert baseline.manifest.hashes["evaluation_truth_pairs"]
    assert baseline.manifest.hashes["candidate_pairs"]
    baseline_metrics_dir = (
        tmp_path
        / "data"
        / "data"
        / "metrics"
        / baseline.manifest.run_id
        / "all"
    )
    assert (baseline_metrics_dir / "retrieval_candidates.parquet").exists()
    assert (baseline_metrics_dir / "evaluation_truth.parquet").exists()
    assert (baseline_metrics_dir / "retrieval_profile_metrics.parquet").exists()
    assert (baseline_metrics_dir / "retrieval_preblend_lost.parquet").exists()
    assert (baseline_metrics_dir / "retrieval_blend_profiles.parquet").exists()
    assert baseline.manifest.hashes["retrieval_preblend_lost_artifact"]
    assert baseline.manifest.hashes["retrieval_blend_profiles_artifact"]
    assert baseline.manifest.hashes["preblend_incumbent_pair_set"]
    assert baseline.manifest.config["external_replacement_requires_witness"] is False
    persisted_data_metrics = json.loads(
        (baseline_metrics_dir / "data_metrics.json").read_text(encoding="utf-8")
    )
    assert persisted_data_metrics == baseline.data
    persisted_candidates = pd.read_parquet(
        baseline_metrics_dir / "retrieval_candidates.parquet"
    )
    assert {
        "retrieval_score",
        "external_fusion_score",
        "type_id",
        "base_score",
        "score",
    }.issubset(persisted_candidates)
    blend_profiles = pd.read_parquet(
        baseline_metrics_dir / "retrieval_blend_profiles.parquet"
    )
    blend_lost = pd.read_parquet(
        baseline_metrics_dir / "retrieval_preblend_lost.parquet"
    )
    assert len(blend_profiles) == baseline.evaluation["candidate_profiles"]
    assert blend_profiles["lost_safety_anchor_count"].sum() == 0
    assert blend_lost.empty
    assert baseline.training["score_source"] == "serving_personalized_retrieval_score"
    temporal_metrics = json.loads(
        (paths["contents"].parent / "temporal" / "data_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        baseline.evaluation["novel_positive_pairs"]
        == temporal_metrics["test_novel_positives"]
    )
    assert baseline.evaluation["model_features_materialized"] is False
    assert baseline.evaluation["serving_candidate_columns"] <= 20

    live = popularity_live_test(
        tmp_path,
        "all",
        num_profiles=2,
        warm_repetitions=0,
        output_dir=tmp_path / "client_output",
        num_contents=220,
        profile_chunk_size=1,
    )
    assert live.output["exact_20_normal_rule_rate"] == 1.0
    assert live.output["exact_10_content_rate"] == 1.0
    assert 0 < live.inference["inference_pairs"] <= 48
    assert (
        live.inference["inference_pairs"]
        == live.inference["candidate_pairs_actual"]
    )
    assert live.inference["deterministic_warm_hash_rate"] is False
    assert live.manifest.phase == "sprint-5"
    gate = json.loads(
        (
            tmp_path
            / "data"
            / "data"
            / "metrics"
            / live.manifest.run_id
            / "all"
            / "gate_result.json"
        ).read_text(encoding="utf-8")
    )
    assert gate["phase"] == "sprint-5"
    assert (tmp_path / "client_output" / "rulename.json").exists()

    resumed = popularity_live_test(
        tmp_path,
        "all",
        num_profiles=2,
        warm_repetitions=0,
        output_dir=tmp_path / "client_output",
        num_contents=220,
        profile_chunk_size=1,
        resume=True,
    )
    assert resumed.inference["resume_chunks_loaded"] == 2
    assert resumed.inference["output_sha256"] == live.inference["output_sha256"]


def test_invalid_live_output_does_not_replace_existing_client_files(tmp_path, monkeypatch):
    _write_synthetic_raw(tmp_path)
    prepare_canonical_project(tmp_path, "all", include_behavior=False)
    output = tmp_path / "client_output"
    output.mkdir()
    homepage_path = output / "rulename.json"
    content_path = output / "rule_content.txt"
    homepage_path.write_text('[{"pid": "previous"}]', encoding="utf-8")
    content_path.write_text("previous|1,1#previous\n", encoding="utf-8")
    previous_homepage = homepage_path.read_bytes()
    previous_content = content_path.read_bytes()
    # Patch where the serving stage resolves it.
    monkeypatch.setattr(
        serving,
        "validate_client_output",
        lambda *args, **kwargs: OutputValidationResult(False, ["forced invalid output"]),
    )

    with pytest.raises(ValueError, match="Client output validation failed"):
        popularity_live_test(
            tmp_path,
            "all",
            num_profiles=2,
            warm_repetitions=0,
            output_dir=output,
            num_contents=220,
            profile_chunk_size=1,
        )

    assert homepage_path.read_bytes() == previous_homepage
    assert content_path.read_bytes() == previous_content
