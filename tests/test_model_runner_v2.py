import json

import pandas as pd
import pytest
import torch

from recsys_vnpt.model_runner import (
    TrainingConfig,
    load_v2_checkpoint,
    predict,
    train_model,
)


def _model_frame(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "profile_id": [f"p{index % 4}" for index in range(rows)],
            "content_id": [f"c{index % 6}" for index in range(rows)],
            "type_id": [str(index % 2) for index in range(rows)],
            "package_code": ["A" if index % 2 else "B" for index in range(rows)],
            "tag_names": ["sports,movie" if index % 2 else "kids" for index in range(rows)],
            "historical_interaction_count": [index + 1 for index in range(rows)],
            "content_duration": [100 + index for index in range(rows)],
            "label": [index % 2 for index in range(rows)],
        }
    )


def test_training_initializes_without_checkpoint_and_validates_contract_hash(tmp_path):
    config = TrainingConfig(
        model_family="embedding_fcn",
        epochs=2,
        patience=2,
        batch_size=4,
        hidden_dims=(8,),
        embedding_dim=4,
        numeric_projection_dim=4,
    )
    train = _model_frame(24)
    validation = _model_frame(8)
    model, contract, metrics = train_model(
        train,
        validation,
        "all",
        tmp_path,
        config,
    )
    assert metrics["initialization"] == "random"
    assert metrics["training_rows"] == 24
    assert (tmp_path / "best_model_v2.pth").exists()
    assert contract.numeric_features["historical_interaction_count"].mean != 0

    loaded_model, loaded_contract, _ = load_v2_checkpoint(
        tmp_path / "best_model_v2.pth",
        tmp_path / "feature_contract_v2.json",
    )
    assert len(predict(loaded_model, loaded_contract, validation, "embedding_fcn")) == 8

    payload = json.loads((tmp_path / "feature_contract_v2.json").read_text(encoding="utf-8"))
    payload["categorical_features"][0]["vocabulary"]["tampered"] = 999
    tampered = tmp_path / "tampered_contract.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hashes do not match"):
        load_v2_checkpoint(tmp_path / "best_model_v2.pth", tampered)


def test_training_resumes_optimizer_and_scheduler_after_interruption(
    tmp_path, monkeypatch
):
    config = TrainingConfig(
        model_family="embedding_fcn",
        epochs=3,
        patience=3,
        batch_size=4,
        hidden_dims=(8,),
        embedding_dim=4,
        numeric_projection_dim=4,
    )
    train = _model_frame(24)
    validation = _model_frame(8)
    real_save = torch.save
    interrupted = False

    def save_then_interrupt(payload, path):
        nonlocal interrupted
        real_save(payload, path)
        if not interrupted:
            interrupted = True
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(torch, "save", save_then_interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_model(train, validation, "all", tmp_path, config)

    monkeypatch.setattr(torch, "save", real_save)
    _, _, metrics = train_model(
        train,
        validation,
        "all",
        tmp_path,
        config,
        resume=True,
    )

    assert metrics["initialization"] == "resumed_checkpoint"
    assert metrics["epochs_completed"] == 2
    checkpoint = torch.load(
        tmp_path / "best_model_v2.pth", map_location="cpu", weights_only=False
    )
    assert "optimizer_state_dict" in checkpoint
    assert "scheduler_state_dict" in checkpoint
    assert checkpoint["training_data_hash"] == metrics["training_data_hash"]


def test_resume_rejects_changed_training_rows(tmp_path):
    config = TrainingConfig(
        model_family="embedding_fcn",
        epochs=1,
        patience=1,
        batch_size=4,
        hidden_dims=(8,),
        embedding_dim=4,
        numeric_projection_dim=4,
    )
    train = _model_frame(24)
    validation = _model_frame(8)
    train_model(train, validation, "all", tmp_path, config)
    changed = train.copy()
    changed.loc[0, "label"] = 1 - int(changed.loc[0, "label"])
    with pytest.raises(ValueError, match="training data hash does not match"):
        train_model(
            changed,
            validation,
            "all",
            tmp_path,
            config,
            resume=True,
        )
