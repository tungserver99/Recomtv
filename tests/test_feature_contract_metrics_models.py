import json

import numpy as np
import pandas as pd
import pytest
import torch

from recsys_vnpt.feature_contract import FeatureContract
from recsys_vnpt.metrics_io import (
    build_model_run_metrics,
    compare_model_runs,
    read_metrics_json,
    write_metrics_json,
)
from recsys_vnpt.metrics import output_health_metrics
from recsys_vnpt.models import EmbeddingDCNv3, EmbeddingFCN, EmbeddingModelConfig
from recsys_vnpt.timing import measure_seconds, runtime_metrics


def test_feature_contract_handles_train_infer_mismatch_and_unseen_values(tmp_path):
    train = pd.DataFrame(
        {
            "age": [25, 40],
            "watch_score": [1.5, 2.5],
            "package_code": ["MYTV021", "VASC011"],
            "tag_names": ["sports,movie", "kids"],
        }
    )
    contract = FeatureContract.fit(
        domain="clip",
        frame=train,
        label_policy={"watch_percent_threshold": 0.3},
        model_family="embedding_dcnv3",
        model_config={"embedding_dim": 4, "hidden_dims": [8]},
        numeric_columns=["age", "watch_score"],
        categorical_columns=["package_code"],
        multivalue_columns=["tag_names"],
    )
    path = tmp_path / "feature_contract.json"
    contract.save(path)

    loaded = FeatureContract.load(path)
    inference = pd.DataFrame(
        {
            "package_code": ["BRAND_NEW", None],
            "tag_names": ["sports,brand_new_tag", ""],
            "extra_inference_only": [99, 100],
        }
    )

    batch = loaded.transform(inference)

    assert batch.numeric_feature_names == ["age", "watch_score"]
    assert np.allclose(batch.numeric_values, [[0.0, 0.0], [0.0, 0.0]])
    package_feature = loaded.categorical_features["package_code"]
    assert batch.categorical_values["package_code"].tolist() == [
        package_feature.unknown_id,
        package_feature.missing_id,
    ]
    tag_feature = loaded.multivalue_features["tag_names"]
    assert batch.multivalue_values["tag_names"][0] == [
        tag_feature.vocabulary["sports"],
        tag_feature.unknown_id,
    ]
    assert batch.multivalue_values["tag_names"][1] == [tag_feature.missing_id]
    assert "extra_inference_only" not in batch.numeric_feature_names


def test_feature_contract_rejects_checkpoint_config_mismatch():
    frame = pd.DataFrame({"age": [1], "package_code": ["A"], "tag_names": ["x"]})
    contract = FeatureContract.fit(
        domain="clip",
        frame=frame,
        label_policy={"watch_percent_threshold": 0.3},
        model_family="embedding_dcnv3",
        model_config={"embedding_dim": 4},
        numeric_columns=["age"],
        categorical_columns=["package_code"],
        multivalue_columns=["tag_names"],
    )

    contract.assert_model_compatible("embedding_dcnv3", {"embedding_dim": 4})
    with pytest.raises(ValueError, match="model family"):
        contract.assert_model_compatible("embedding_fcn", {"embedding_dim": 4})
    with pytest.raises(ValueError, match="model config"):
        contract.assert_model_compatible("embedding_dcnv3", {"embedding_dim": 8})


def test_encoded_batch_preserves_training_order_and_torch_shapes():
    frame = pd.DataFrame(
        {
            "age": [25, 40],
            "watch_score": [1.5, 2.5],
            "package_code": ["MYTV021", "VASC011"],
            "tag_names": ["sports,movie", "kids"],
        }
    )
    contract = FeatureContract.fit(
        domain="clip",
        frame=frame,
        label_policy={"watch_percent_threshold": 0.3},
        model_family="embedding_fcn",
        model_config={"embedding_dim": 4},
        numeric_columns=["age", "watch_score"],
        categorical_columns=["package_code"],
        multivalue_columns=["tag_names"],
    )

    torch_batch = contract.transform(frame).to_torch()

    assert torch_batch["numeric"].shape == (2, 2)
    assert torch_batch["categorical"].shape == (2, 1)
    assert torch_batch["categorical_feature_names"] == ["package_code"]
    assert torch_batch["multivalue"]["tag_names"]["offsets"].tolist() == [0, 2]
    assert torch_batch["multivalue"]["tag_names"]["values"].dtype == torch.long


def test_numeric_contract_normalizes_nullable_integer_columns():
    frame = pd.DataFrame(
        {"count": pd.Series([1, pd.NA, 3], dtype="Int64")}
    )
    contract = FeatureContract.fit(
        domain="all",
        frame=frame,
        label_policy={},
        model_family="embedding_fcn",
        model_config={},
        numeric_columns=["count"],
        categorical_columns=[],
        multivalue_columns=[],
    )
    values = contract.transform(frame).numeric_values[:, 0]
    assert np.isfinite(values).all()
    assert values[1] == 0.0


def test_runtime_metrics_use_seconds_and_compute_per_pair_latency():
    ticks = iter([10.0, 12.5])
    result, elapsed = measure_seconds(lambda: "trained", clock=lambda: next(ticks))

    metrics = runtime_metrics(
        model_family="embedding_dcnv3",
        training_time_seconds=elapsed,
        inference_time_seconds=4.0,
        inference_pairs=8,
    )

    assert result == "trained"
    assert metrics["model_family"] == "embedding_dcnv3"
    assert metrics["training_time_seconds"] == 2.5
    assert metrics["inference_time_seconds"] == 4.0
    assert metrics["inference_pairs"] == 8
    assert metrics["inference_seconds_per_pair"] == 0.5
    with pytest.raises(ValueError, match="nonnegative"):
        runtime_metrics(model_family="embedding_fcn", training_time_seconds=-0.1)


def test_output_health_metrics_count_coverage_and_unmatched_rules():
    result = {
        "p1": {
            "suggested_content": {
                "c1": {"rule_id": "10"},
                "c2": {"rule_id": "-100"},
            }
        },
        "p2": {
            "suggested_content": {
                "c2": {"rule_id": "11"},
                "c3": {"rule_id": "-100"},
            }
        },
    }

    metrics = output_health_metrics(result, candidate_content_count=6, top_n=200)

    assert metrics == {
        "profiles_scored": 2,
        "recommendations_returned": 4,
        "candidate_contents": 6,
        "distinct_recommended_contents": 3,
        "catalog_coverage": 0.5,
        "average_recommendations_per_profile": 2.0,
        "top_n": 200,
        "unmatched_rule_count": 2,
        "unmatched_rule_rate": 0.5,
    }


def test_metrics_json_and_model_comparison_include_quality_rule_and_timing(tmp_path):
    dense = build_model_run_metrics(
        model_family="dense_mimic_baseline",
        quality_metrics={"recall@10": 0.2},
        runtime_metrics={"training_time_seconds": 10.0, "inference_time_seconds": 4.0},
        rule_metrics={"unmatched_rule_rate": 0.0912},
    )
    embedding = build_model_run_metrics(
        model_family="embedding_dcnv3",
        quality_metrics={"recall@10": 0.25},
        runtime_metrics={"training_time_seconds": 12.0, "inference_time_seconds": 5.0},
        rule_metrics={"unmatched_rule_rate": 0.07},
    )
    output = tmp_path / "training_metrics.json"

    write_metrics_json(output, dense)
    loaded = read_metrics_json(output)
    comparison = compare_model_runs([dense, embedding], primary_metric="recall@10")

    assert loaded["model_family"] == "dense_mimic_baseline"
    assert json.loads(output.read_text(encoding="utf-8"))["runtime"]["training_time_seconds"] == 10.0
    assert comparison["best_model_family"] == "embedding_dcnv3"
    assert [run["model_family"] for run in comparison["runs"]] == [
        "dense_mimic_baseline",
        "embedding_dcnv3",
    ]


def test_embedding_models_accept_contract_encoded_batches():
    frame = pd.DataFrame(
        {
            "age": [25, 40],
            "watch_score": [1.5, 2.5],
            "package_code": ["MYTV021", "VASC011"],
            "tag_names": ["sports,movie", "kids"],
        }
    )
    contract = FeatureContract.fit(
        domain="clip",
        frame=frame,
        label_policy={"watch_percent_threshold": 0.3},
        model_family="embedding_dcnv3",
        model_config={"embedding_dim": 4, "numeric_projection_dim": 4, "hidden_dims": [8]},
        numeric_columns=["age", "watch_score"],
        categorical_columns=["package_code"],
        multivalue_columns=["tag_names"],
    )
    batch = contract.transform(frame).to_torch()
    config = EmbeddingModelConfig.from_contract(
        contract,
        embedding_dim=4,
        numeric_projection_dim=4,
        hidden_dims=(8,),
    )

    fcn_output = EmbeddingFCN(config)(batch)
    dcn_output = EmbeddingDCNv3(config)(batch)

    assert fcn_output["y_pred"].shape == (2,)
    assert dcn_output["y_pred"].shape == (2,)
    assert set(dcn_output) == {"y_pred", "y_d", "y_s"}
    assert torch.all((dcn_output["y_pred"] >= 0.0) & (dcn_output["y_pred"] <= 1.0))
