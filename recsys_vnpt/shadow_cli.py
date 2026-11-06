from __future__ import annotations

import argparse
import json
from pathlib import Path

from recsys_vnpt.forecast import fit_runtime_forecast, observation_from_metrics
from recsys_vnpt.pipeline_v2 import popularity_live_test


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run non-publishing local/client shadow workload ramps."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--domain", choices=("all", "clip", "movie", "music"), default="all")
    parser.add_argument(
        "--model-family",
        choices=(
            "per_rule_popularity",
            "dense_repaired_baseline",
            "embedding_mlp_baseline",
            "legacy_embedding_dcnv3",
            "fcn_official",
            "ecn_official",
            "hierarchical_fcn",
            "embedding_fcn",
            "embedding_dcnv3",
        ),
        default="per_rule_popularity",
    )
    parser.add_argument("--workloads", nargs="+", type=int, required=True)
    parser.add_argument("--num-contents", type=int)
    parser.add_argument("--warm-repetitions", type=int, default=3)
    parser.add_argument("--profile-chunk-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--no-gpu-sampling", action="store_true")
    parser.add_argument("--candidate-budget", type=int, default=1000)
    parser.add_argument("--exact-cowatch", action="store_true")
    parser.add_argument("--output", default="reports/SHADOW_LATEST.json")
    args = parser.parse_args(argv)

    if args.require_cuda:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA shadow run requested, but this PyTorch build cannot access CUDA"
            )

    bundles = []
    for profiles in args.workloads:
        bundles.append(
            popularity_live_test(
                args.root,
                args.domain,
                num_profiles=profiles,
                warm_repetitions=args.warm_repetitions,
                model_family=args.model_family,
                num_contents=args.num_contents,
                profile_chunk_size=args.profile_chunk_size,
                batch_size=args.batch_size,
                dry_run=True,
                amp=args.amp,
                sample_gpu=not args.no_gpu_sampling,
                candidate_budget=args.candidate_budget,
                exact_cowatch=args.exact_cowatch,
            )
        )
    observations = [
        observation_from_metrics(
            bundle.manifest.run_id,
            bundle.inference,
            str(bundle.manifest.hardware.get("device", "unknown")),
        )
        for bundle in bundles
    ]
    forecast = fit_runtime_forecast(observations)
    payload = {
        "domain": args.domain,
        "model_family": args.model_family,
        "publishes_to_homepage": False,
        "run_ids": [bundle.manifest.run_id for bundle in bundles],
        "forecast": forecast,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if forecast["holdout"]["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
