from pathlib import Path


DOMAINS = ("clip", "movie", "music", "all")


def test_domain_train_scripts_persist_contract_and_training_metrics():
    root = Path(__file__).resolve().parents[1]
    for domain in DOMAINS:
        text = (root / domain / "train.py").read_text(encoding="utf-8")

        assert "FeatureContract" in text
        assert "feature_contract.json" in text
        assert "training_metrics.json" in text
        assert "dense_mimic_baseline" in text


def test_domain_infer_scripts_validate_contract_and_persist_inference_metrics():
    root = Path(__file__).resolve().parents[1]
    for domain in DOMAINS:
        text = (root / domain / "infer.py").read_text(encoding="utf-8")

        assert "FeatureContract.load" in text
        assert "assert_model_compatible" in text
        assert "inference_metrics.json" in text
        assert "inference_seconds_per_pair" in text or "runtime_metrics" in text


def test_domain_infer_scripts_accept_bounded_inference_caps():
    root = Path(__file__).resolve().parents[1]
    content_arg_by_domain = {
        "clip": "num_clip",
        "movie": "num_movie",
        "music": "num_music",
        "all": "num_all",
    }
    for domain in DOMAINS:
        text = (root / domain / "infer.py").read_text(encoding="utf-8")

        assert "--num-users" in text
        assert "--num-contents" in text
        assert "num_user=args.num_users" in text
        assert f"{content_arg_by_domain[domain]}=args.num_contents" in text
