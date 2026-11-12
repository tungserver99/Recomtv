"""End-to-end batch runs on the synthetic input batch."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

import recsys_vnpt.batch_runner as runner
from recsys_vnpt.batch_contracts import BatchConfig, BatchError
from recsys_vnpt.export import (
    JsonArrayWriter,
    _json_array_one_record_per_line,
    read_current_delivery,
)
from recsys_vnpt.output_validation import validate_client_output
from recsys_vnpt.scorer_bundle import promoted_bundle
from recsys_vnpt.smoke_fixture import write_smoke_batch


def _config(tmp_path, **overrides) -> BatchConfig:
    settings = {
        "incoming_root": str(tmp_path / "incoming"),
        "work_root": str(tmp_path / "work"),
        "delivery_root": str(tmp_path / "delivery"),
        "model_registry": str(tmp_path / "registry"),
        "min_free_bytes": 0,
        "input_wait_seconds": 0,
        "num_shards": 3,
    }
    return BatchConfig(**{**settings, **overrides})


def _hand_built(tmp_path, **overrides) -> BatchConfig:
    return _config(tmp_path, rule_ordering="count", content_ordering="hand_sum", **overrides)


def _delivered(config) -> dict:
    current = read_current_delivery(config.delivery_root)
    directory = Path(current["directory"])
    return {name: (directory / name).read_bytes() for name in current["files"]}


def test_retrain_then_inference_commits_a_valid_three_file_delivery(tmp_path):
    write_smoke_batch(tmp_path / "incoming", "b1")
    config = _config(tmp_path)

    retrained = runner.run_retrain_batch(config, "b1")
    assert retrained["status"] == "promoted"
    assert promoted_bundle(config.model_registry) == Path(retrained["bundle"]).resolve()

    report = runner.run_inference_batch(config, "b1")
    assert report["status"] == "published"
    assert report["bundle_id"] == Path(retrained["bundle"]).name
    files = _delivered(config)
    names = sorted(name for name in files if name.startswith("f_"))
    assert names == [
        "f_personalized_homepage_rule_ms_all_0_700000_DL_MyTV_20250302.json",
        "f_personalized_rule_content_ms_0_700000_DL_MyTV_20250302.csv",
        "f_profile_grouping_ms_DL_MyTV_0_20250302.json",
    ]
    homepage = json.loads(files[names[0]])
    validation = validate_client_output(
        homepage, files[names[1]].decode("utf-8"), config=config.assembly
    )
    assert validation.passed, validation.errors[:3]
    assert validation.metrics["profiles_emitted"] == 30

    again = runner.run_inference_batch(config, "b1")
    assert again["status"] == "already_done"
    assert read_current_delivery(config.delivery_root)["generation"] == report["generation"]


def test_inference_never_fits_a_missing_model(tmp_path):
    write_smoke_batch(tmp_path / "incoming", "b1")
    config = _config(tmp_path)
    with pytest.raises(BatchError, match="MODEL_UNAVAILABLE") as error:
        runner.run_inference_batch(config, "b1")
    assert error.value.exit_code == 20
    assert read_current_delivery(config.delivery_root) is None


def test_shard_and_worker_counts_do_not_change_delivered_bytes(tmp_path):
    write_smoke_batch(tmp_path / "incoming", "b1")
    one = _hand_built(tmp_path / "one", num_shards=1)
    many = _hand_built(tmp_path / "many", num_shards=4, workers=2)
    one = replace(one, incoming_root=str(tmp_path / "incoming"))
    many = replace(many, incoming_root=str(tmp_path / "incoming"))
    runner.run_inference_batch(one, "b1")
    runner.run_inference_batch(many, "b1")
    first, second = _delivered(one), _delivered(many)
    assert {k: v for k, v in first.items() if k.startswith("f_")} == {
        k: v for k, v in second.items() if k.startswith("f_")
    }


def test_a_rerun_recomputes_only_the_shards_that_do_not_verify(tmp_path, monkeypatch):
    write_smoke_batch(tmp_path / "incoming", "b1")
    config = _hand_built(tmp_path)
    real_merge = runner.merge_shards

    def failing_merge(*args, **kwargs):
        raise OSError("disk went away")

    monkeypatch.setattr(runner, "merge_shards", failing_merge)
    with pytest.raises(BatchError, match="OUTPUT_INVALID"):
        runner.run_inference_batch(config, "b1")
    assert read_current_delivery(config.delivery_root) is None

    shard = Path(config.work_root) / "batches" / "b1" / "inference" / "shards" / "shard_00001"
    (shard / "rule_content.txt").write_text("corrupted\n", encoding="utf-8")
    monkeypatch.setattr(runner, "merge_shards", real_merge)
    report = runner.run_inference_batch(config, "b1")
    assert report["status"] == "published"
    assert report["shards_reused"] == config.num_shards - 1


def test_max_profiles_serves_a_stable_sample(tmp_path):
    write_smoke_batch(tmp_path / "incoming", "b1")
    config = _hand_built(tmp_path, max_profiles=5)
    report = runner.run_inference_batch(config, "b1")
    assert report["status"] == "published"
    assert report["profiles"] == 5
    ids = [f"p{i}" for i in range(30)]
    assert runner.sample_profiles(ids, 5) == runner.sample_profiles(list(reversed(ids)), 5)
    assert runner.sample_profiles(ids, 0) == ids


def test_a_batch_that_does_not_advance_the_cutoff_is_refused(tmp_path):
    write_smoke_batch(tmp_path / "incoming", "b1")
    write_smoke_batch(tmp_path / "incoming", "b1-copy")
    config = _hand_built(tmp_path)
    published = runner.run_inference_batch(config, "b1")
    with pytest.raises(BatchError, match="does not advance"):
        runner.run_inference_batch(config, "b1-copy")
    assert read_current_delivery(config.delivery_root)["generation"] == published["generation"]


def test_a_failed_gate_leaves_the_promoted_model_alone(tmp_path, monkeypatch):
    write_smoke_batch(tmp_path / "incoming", "b1")
    write_smoke_batch(tmp_path / "incoming", "b2")
    config = _config(tmp_path)
    first = runner.run_retrain_batch(config, "b1")

    def worse(config, bundle=None):
        # The bundle being gated scores below the promoted one.
        score = 0.4 if Path(bundle).resolve() != Path(first["bundle"]).resolve() else 0.5
        return {"run_id": "x", "novel_positive_pairs": 10, "rule_ndcg@20": score, "macro_content_ndcg@10": score}

    monkeypatch.setattr(runner, "_evaluate", worse)
    with pytest.raises(BatchError, match="QUALITY_GATE_FAILED") as error:
        runner.run_retrain_batch(config, "b2")
    assert error.value.exit_code == 22
    assert promoted_bundle(config.model_registry) == Path(first["bundle"]).resolve()
    status = json.loads((Path(config.work_root) / "batches/b2/retrain/status.json").read_text())
    assert status["phase"] == "REJECTED"


def test_gate_requires_evidence_and_bounds_regression():
    candidate = {"novel_positive_pairs": 5, "rule_ndcg@20": 0.62, "macro_content_ndcg@10": 0.13}
    assert runner.retrain_gate(candidate, None, 0.02)["passed"]
    promoted = {"rule_ndcg@20": 0.70, "macro_content_ndcg@10": 0.13}
    assert not runner.retrain_gate(candidate, promoted, 0.02)["passed"]
    empty = {**candidate, "novel_positive_pairs": 0}
    assert not runner.retrain_gate(empty, None, 0.02)["passed"]
    # The hand-built ordering is no longer a threshold.
    assert [check["check"] for check in runner.retrain_gate(candidate, None, 0.02)["checks"]] == [
        "test window has novel positives"
    ]


def test_merge_rejects_a_profile_in_two_shards(tmp_path):
    for name in ("a", "b"):
        shard = tmp_path / name
        shard.mkdir()
        (shard / "homepage.jsonl").write_text('{"pid": "p1", "d": "1"}\n')
        (shard / "rule_content.txt").write_text("p1|1,2#c1\n")
        (shard / "grouping.jsonl").write_text('{"pid": "p1", "mid": "p1", "gid": 3}\n')
    names = {"homepage": "h.json", "rule_content": "c.csv", "grouping": "g.json"}
    with pytest.raises(BatchError, match="more than once"):
        runner.merge_shards([tmp_path / "a", tmp_path / "b"], tmp_path / "out", names, {"p1"})


def test_streamed_arrays_match_the_fixture_layout(tmp_path):
    records = [{"pid": "p1", "d": "1,2"}, {"pid": "p2", "d": "3"}]
    for rows in (records, []):
        path = tmp_path / "out.json"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = JsonArrayWriter(handle)
            for row in rows:
                writer.write_line(json.dumps(row, ensure_ascii=False))
            writer.close()
        assert path.read_text(encoding="utf-8") == _json_array_one_record_per_line(rows)


def test_qualification_measures_what_exists_and_refuses_to_extrapolate(tmp_path):
    write_smoke_batch(tmp_path / "incoming", "b1")
    config = _hand_built(tmp_path, qualification_workloads=(10, 30, 1000), qualification_repetitions=2)
    report = runner.qualify_hardware(config, "b1")
    measured = {item["workload"]: item for item in report["results"]}
    assert measured[10]["deterministic"] and measured[30]["deterministic"]
    assert measured[1000]["status"] == "not_measurable"
    assert report["verdict"] == "not_measured"
    assert read_current_delivery(config.delivery_root) is None


def test_a_shard_that_fails_transiently_is_retried_once(tmp_path, monkeypatch):
    write_smoke_batch(tmp_path / "incoming", "b1")
    config = _hand_built(tmp_path)
    real_run_shard = runner._run_shard
    calls = {"shard_1": 0}

    def flaky(context, member_ids, config, directory, profile_ids, identity):
        if identity["shard"] == 1:
            calls["shard_1"] += 1
            if calls["shard_1"] == 1:
                raise OSError("network share dropped")
        return real_run_shard(context, member_ids, config, directory, profile_ids, identity)

    monkeypatch.setattr(runner, "_run_shard", flaky)
    report = runner.run_inference_batch(config, "b1")
    assert report["status"] == "published"
    assert calls["shard_1"] == 2
    log = (Path(config.work_root) / "batches/b1/inference/log.jsonl").read_text()
    assert '"event": "shard_retry"' in log


def test_persistent_or_contract_failures_are_not_papered_over(tmp_path, monkeypatch):
    write_smoke_batch(tmp_path / "incoming", "b1")
    config = _hand_built(tmp_path)
    calls = []

    def broken(*args, **kwargs):
        calls.append(1)
        raise OSError("disk gone")

    monkeypatch.setattr(runner, "_run_shard", broken)
    with pytest.raises(BatchError, match="after 2 attempts"):
        runner.run_inference_batch(config, "b1")

    def invalid(*args, **kwargs):
        calls.append(1)
        raise BatchError("OUTPUT_INVALID", "shard 0: bad page")

    calls.clear()
    monkeypatch.setattr(runner, "_run_shard", invalid)
    with pytest.raises(BatchError, match="OUTPUT_INVALID"):
        runner.run_inference_batch(config, "b1")
    assert len(calls) == 1
    assert read_current_delivery(config.delivery_root) is None
