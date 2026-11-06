import json

import numpy as np
import pandas as pd
import pytest
import torch

from recsys_vnpt.model_runner import (
    TrainingConfig,
    _grouped_pairwise_loss,
    _hierarchical_slate_batches,
    _loss,
    _rule_listwise_loss,
    infer_feature_roles,
    load_v2_checkpoint,
    train_model,
)
from recsys_vnpt.models import (
    EmbeddingModelConfig,
    FieldAlignedFeatureEncoder,
    OfficialCrossNetwork,
    OfficialFCN,
)


def _batch():
    return {
        "numeric": torch.tensor([[1.0], [2.0]]),
        "categorical": torch.tensor([[2], [3]]),
        "categorical_feature_names": ["category"],
        "multivalue": {},
    }


def test_multihead_field_encoder_and_official_fcn_shapes():
    config = EmbeddingModelConfig(
        numeric_feature_names=("numeric",),
        categorical_vocab_sizes={"category": 4},
        multivalue_vocab_sizes={},
        embedding_dim=4,
        num_heads=2,
        deep_cross_layers=2,
        shallow_cross_layers=1,
        batch_norm=False,
    )
    encoder = FieldAlignedFeatureEncoder(config)
    encoded = encoder(_batch())
    assert encoded.shape == (2, 2, 8)
    output = OfficialFCN(config)(_batch())
    assert set(output) == {"y_pred", "y_d", "y_s"}
    assert output["y_pred"].shape == (2,)


def test_official_ecn_and_lcn_recurrences_match_reference_equations():
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    ecn = OfficialCrossNetwork(
        4, 1, True, 0.0, 1, layer_norm=False, batch_norm=False
    )
    with torch.no_grad():
        ecn.projections[0].weight.copy_(
            torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        )
        ecn.biases[0].fill_(0.25)
    hidden = ecn.projections[0](x)
    interaction = torch.cat([hidden, hidden * torch.relu(hidden)], dim=-1)
    expected_ecn = x * (interaction + 0.25) + x
    assert torch.allclose(ecn.forward_features(x), expected_ecn)

    lcn = OfficialCrossNetwork(
        4, 2, False, 0.0, 1, layer_norm=False, batch_norm=False
    )
    with torch.no_grad():
        for projection in lcn.projections:
            projection.weight.copy_(
                torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
            )
        for bias in lcn.biases:
            bias.fill_(0.1)
    current = x
    for projection in lcn.projections:
        hidden = projection(current)
        interaction = torch.cat([hidden, hidden * torch.relu(hidden)], dim=-1)
        current = x * (interaction + 0.1) + current
    assert torch.allclose(lcn.forward_features(x), current)


def test_tri_bce_is_finite_and_not_below_fused_bce():
    labels = torch.tensor([1.0, 0.0])
    outputs = {
        "y_pred": torch.tensor([0.8, 0.2]),
        "y_d": torch.tensor([0.7, 0.3]),
        "y_s": torch.tensor([0.6, 0.4]),
    }
    fused = torch.nn.functional.binary_cross_entropy(outputs["y_pred"], labels)
    loss = _loss(outputs, labels, "fcn_official")
    assert torch.isfinite(loss)
    assert loss >= fused


def test_rule_listwise_target_sums_gain_and_preserves_temporal_context() -> None:
    probabilities = torch.full((3,), 0.5, requires_grad=True)
    loss = _rule_listwise_loss(
        probabilities,
        torch.tensor([0.6, 0.6, 0.9]),
        profiles=["p1", "p1", "p1"],
        rules=["a", "a", "b"],
        windows=["w1", "w1", "w1"],
    )
    loss.backward()

    assert probabilities.grad is not None
    assert probabilities.grad[0] < 0
    assert probabilities.grad[1] < 0
    assert probabilities.grad[2] > 0

    contextual = torch.full((4,), 0.5, requires_grad=True)
    contextual_loss = _rule_listwise_loss(
        contextual,
        torch.tensor([1.0, 0.0, 0.0, 1.0]),
        profiles=["p1"] * 4,
        rules=["a", "b", "a", "b"],
        windows=["w1", "w1", "w2", "w2"],
    )
    contextual_loss.backward()
    assert contextual.grad is not None
    assert contextual.grad[0] < 0 < contextual.grad[1]
    assert contextual.grad[2] > 0 > contextual.grad[3]


def test_hierarchical_pairwise_weights_and_inactive_gradient_are_safe() -> None:
    probabilities = torch.tensor([0.5, 0.5], requires_grad=True)
    labels = torch.tensor([1.0, 0.0])
    groups = [("w", "p", "r"), ("w", "p", "r")]
    full = _grouped_pairwise_loss(
        probabilities, labels, groups, sample_weights=torch.tensor([1.0, 1.0])
    )
    downweighted = _grouped_pairwise_loss(
        probabilities, labels, groups, sample_weights=torch.tensor([1.0, 0.25])
    )
    assert downweighted < full

    inactive = _rule_listwise_loss(
        probabilities,
        labels,
        profiles=["p", "p"],
        rules=["only", "only"],
        windows=["w", "w"],
    )
    inactive.backward()
    assert probabilities.grad is not None


def test_hierarchical_batches_keep_profile_window_slates_intact() -> None:
    frame = pd.DataFrame(
        {
            "temporal_window": ["w1"] * 3 + ["w2"] * 2 + ["w1"] * 2,
            "profile_id": ["p"] * 5 + ["q"] * 2,
        }
    )
    batches, diagnostics = _hierarchical_slate_batches(
        frame, batch_size=4, rng=np.random.default_rng(42)
    )
    row_to_batch = {
        int(row): batch_index
        for batch_index, rows in enumerate(batches)
        for row in rows
    }
    for _, slate in frame.groupby(["temporal_window", "profile_id"]):
        assert len({row_to_batch[int(row)] for row in slate.index}) == 1
    assert diagnostics["slates"] == 3
    assert diagnostics["split_slates"] == 0
    assert diagnostics["maximum_slate_rows"] == 3


def test_hierarchical_rule_id_is_grouping_metadata_not_model_input() -> None:
    frame = pd.DataFrame(
        {
            "profile_id": ["p1", "p2"],
            "content_id": ["c1", "c2"],
            "rule_id": ["1", "2"],
            "type_id": ["movie", "music"],
            "historical_interaction_count": [1, 2],
            "label": [1, 0],
        }
    )

    roles = infer_feature_roles(frame, model_family="hierarchical_fcn")

    assert "rule_id" not in roles.categorical
    assert "type_id" in roles.categorical


def test_schema_v3_checkpoint_rejects_family_mismatch_and_drops_username(tmp_path):
    frame = pd.DataFrame(
        {
            "profile_id": [f"p{i % 4}" for i in range(16)],
            "username": [f"u{i % 4}" for i in range(16)],
            "content_id": [f"c{i % 6}" for i in range(16)],
            "type_id": [str(i % 2) for i in range(16)],
            "historical_interaction_count": list(range(1, 17)),
            "label": [i % 2 for i in range(16)],
        }
    )
    config = TrainingConfig(
        model_family="fcn_official",
        epochs=1,
        patience=1,
        batch_size=8,
        embedding_dim=4,
        numeric_projection_dim=4,
        hidden_dims=(8,),
        deep_cross_layers=1,
        shallow_cross_layers=1,
        batch_norm=False,
    )
    _, contract, metrics = train_model(frame, frame.iloc[:8], "all", tmp_path, config)
    checkpoint = torch.load(
        tmp_path / "best_model_v2.pth", map_location="cpu", weights_only=False
    )
    assert checkpoint["schema_version"] == 3
    assert "username" not in contract.categorical_features
    assert metrics["checkpoint_selection_source"] == "log_loss_fallback"

    payload = json.loads(
        (tmp_path / "feature_contract_v2.json").read_text(encoding="utf-8")
    )
    payload["model_family"] = "ecn_official"
    mismatch = tmp_path / "mismatch.json"
    mismatch.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="model family"):
        load_v2_checkpoint(tmp_path / "best_model_v2.pth", mismatch)


def test_hierarchical_checkpoint_records_rule_semantics_and_complete_slates(
    tmp_path,
) -> None:
    frame = pd.DataFrame(
        {
            "temporal_window": ["w1"] * 8 + ["w2"] * 8,
            "profile_id": ["p1"] * 4 + ["p2"] * 4 + ["p1"] * 4 + ["p2"] * 4,
            "content_id": [f"c{i % 4}" for i in range(16)],
            "rule_id": ["a", "a", "b", "b"] * 4,
            "type_id": ["movie", "music"] * 8,
            "historical_interaction_count": list(range(1, 17)),
            "label": [1, 0, 1, 0] * 4,
            "completion_gain": [0.8, 0.0, 0.4, 0.0] * 4,
            "watch_time_gain": [0.7, 0.0, 0.5, 0.0] * 4,
            "sample_weight": [1.0, 0.25, 1.0, 0.25] * 4,
            "target_semantics": ["engagement_v1"] * 16,
        }
    )
    config = TrainingConfig(
        model_family="hierarchical_fcn",
        epochs=1,
        patience=1,
        batch_size=6,
        embedding_dim=4,
        numeric_projection_dim=4,
        deep_cross_layers=1,
        shallow_cross_layers=1,
        batch_norm=False,
    )

    _, contract, metrics = train_model(
        frame, frame.iloc[:8].copy(), "all", tmp_path, config, device="cpu"
    )
    checkpoint_path = tmp_path / "best_model_v2.pth"
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )

    assert checkpoint["schema_version"] == 4
    assert checkpoint["hierarchical_rule_semantics"] == (
        "actual_membership_contribution_v2"
    )
    assert "rule_id" not in contract.categorical_features
    assert metrics["hierarchical_batching"]["split_slates"] == 0

    checkpoint.pop("hierarchical_rule_semantics")
    obsolete = tmp_path / "obsolete.pth"
    torch.save(checkpoint, obsolete)
    with pytest.raises(ValueError, match="obsolete rule semantics"):
        load_v2_checkpoint(
            obsolete, tmp_path / "feature_contract_v3.json", device="cpu"
        )
