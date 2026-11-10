from types import SimpleNamespace

import pytest

from recsys_vnpt import fcn_scorer


def _artifacts(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    contract = tmp_path / "contract.json"
    checkpoint.write_bytes(b"checkpoint")
    contract.write_text("{}", encoding="utf-8")
    return checkpoint, contract


def test_fcn_loader_reuses_the_existing_contract_checked_checkpoint_loader(
    tmp_path, monkeypatch
):
    checkpoint, contract = _artifacts(tmp_path)
    model_object = object()
    contract_object = SimpleNamespace(model_family="fcn_official")
    checkpoint_object = {"epoch": 2}
    calls = []

    def fake_load(checkpoint_path, contract_path, device=None):
        calls.append((checkpoint_path, contract_path, device))
        return model_object, contract_object, checkpoint_object

    monkeypatch.setattr(fcn_scorer, "load_v2_checkpoint", fake_load)
    loaded = fcn_scorer.load_fcn_scorer(checkpoint, contract, device="cpu")

    assert loaded.model is model_object
    assert loaded.contract is contract_object
    assert loaded.checkpoint is checkpoint_object
    assert calls == [(checkpoint, contract, "cpu")]


def test_fcn_loader_requires_official_family_and_both_artifacts(tmp_path):
    checkpoint, contract = _artifacts(tmp_path)

    with pytest.raises(ValueError, match="fcn_official"):
        fcn_scorer.load_fcn_scorer(
            checkpoint, contract, device="cpu", model_family="embedding_fcn"
        )
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        fcn_scorer.load_fcn_scorer(
            tmp_path / "missing.pth", contract, device="cpu"
        )


def test_candidate_scoring_uses_contract_preparation_and_bounded_predict(
    monkeypatch,
):
    scorer = fcn_scorer.FCNScorerModel(
        model=object(), contract=object(), checkpoint={}, model_family="fcn_official", device="cpu"
    )
    frame = object()
    prepared = object()
    calls = []
    monkeypatch.setattr(
        fcn_scorer,
        "prepare_inference_input",
        lambda contract, values, family: calls.append(
            ("prepare", contract, values, family)
        )
        or prepared,
    )

    def fake_predict(*args, **kwargs):
        calls.append(("predict", kwargs))
        return [0.75]

    monkeypatch.setattr(fcn_scorer, "predict", fake_predict)

    assert fcn_scorer.score_fcn_candidates(scorer, frame, batch_size=64) == [0.75]
    assert calls[0] == ("prepare", scorer.contract, frame, "fcn_official")
    assert calls[1][1]["batch_size"] == 64
    assert calls[1][1]["prepared"] is prepared
