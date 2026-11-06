from __future__ import annotations

import argparse
import json
from pathlib import Path

from recsys_vnpt.forecast import fit_runtime_forecast, observation_from_metrics
from recsys_vnpt.metrics_schema import MetricBundle
from recsys_vnpt.metrics_store import (
    MetricsStore,
    create_manifest,
    evaluate_gate,
    render_markdown,
)
from recsys_vnpt.paths import model_artifact_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit validated RecSys-VNPT runtime forecasts.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--domain", default="all")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--targets", nargs="+", type=int, default=[1_000_000, 2_000_000])
    parser.add_argument("--pairs-per-profile", type=int)
    parser.add_argument("--output", default="reports/RUNTIME_FORECAST.json")
    parser.add_argument("--phase", choices=("sprint-5", "sprint-6"))
    args = parser.parse_args(argv)

    store = MetricsStore(args.root)
    observations = []
    bundles = []
    for run_id in args.runs:
        bundle = store.read_bundle(run_id, args.domain)
        bundles.append(bundle)
        observations.append(
            observation_from_metrics(
                bundle.manifest.run_id,
                bundle.inference,
                str(bundle.manifest.hardware.get("device", "unknown")),
            )
        )
    forecast = fit_runtime_forecast(
        observations,
        target_profiles=args.targets,
        pairs_per_profile=args.pairs_per_profile,
    )
    largest = max(bundles, key=lambda bundle: int(bundle.inference["profiles_scored"]))
    measured_profiles = int(largest.inference["profiles_scored"])
    non_gpu_seconds = float(largest.inference.get("rule_assembly_seconds", 0.0)) + float(
        largest.inference.get("serialization_seconds", 0.0)
    )
    measured_total = float(largest.inference.get("p50_seconds", largest.inference["inference_time_seconds"]))
    encode_forward = float(largest.inference.get("encode_forward_seconds", 0.0))
    forecast["a100_unmeasured_scenario"] = {
        "status": "not_an_a100_prediction",
        "source_run_id": largest.manifest.run_id,
        "encode_forward_fraction": encode_forward / measured_total if measured_total else 0.0,
        "non_gpu_stage_fraction": non_gpu_seconds / measured_total if measured_total else 0.0,
        "assumption": (
            "same CPU/serialization scaling; lower bound sets the entire encode+forward stage to zero"
        ),
        "theoretical_best_case": {
            str(target): {
                "seconds": non_gpu_seconds / measured_profiles * target,
                "hours": non_gpu_seconds / measured_profiles * target / 3600,
            }
            for target in args.targets
        },
        "measurement_required": "run the 10K -> 100K -> 1M -> 2M shadow ramp on the client A100",
    }
    gate_payload = None
    metrics_run_id = None
    if args.phase:
        warm_candidates = [
            bundle
            for bundle in bundles
            if int(bundle.inference.get("warm_repetitions", 0)) >= 3
        ]
        warm_reference = max(
            warm_candidates,
            key=lambda bundle: int(bundle.inference["profiles_scored"]),
        ) if warm_candidates else largest
        manifest = create_manifest(
            args.root,
            args.domain,
            args.phase,
            largest.manifest.model_family,
            config={"source_run_ids": args.runs},
        )
        training_metrics = next(
            (dict(bundle.training) for bundle in bundles if bundle.training), {}
        )
        training_metrics_path = model_artifact_path(
            args.domain,
            "v2",
            largest.manifest.model_family,
            "training_metrics.json",
            project_root=args.root,
        )
        if not training_metrics and training_metrics_path.exists():
            training_metrics = json.loads(
                training_metrics_path.read_text(encoding="utf-8")
            )
        inference = dict(largest.inference)
        if int(inference.get("warm_repetitions", 0)) == 0:
            inference["deterministic_warm_hash_rate"] = False
        inference.update(
            {
                "forecast_error_rate": forecast["holdout"]["forecast_error_rate"],
                "ramp_warm_reference_run_id": warm_reference.manifest.run_id,
                "ramp_warm_repetitions": int(
                    warm_reference.inference.get("warm_repetitions", 0)
                ),
                "ramp_deterministic_warm_hash_rate": bool(
                    warm_reference.inference.get("deterministic_warm_hash_rate", False)
                ),
                "workload_run_ids": args.runs,
                "forecast": forecast,
            }
        )
        rehearsal = MetricBundle(
            manifest=manifest,
            data=dict(largest.data),
            training=training_metrics,
            evaluation=dict(largest.evaluation),
            inference=inference,
            output=dict(largest.output),
        )
        directory = store.write_bundle(rehearsal)
        gate = evaluate_gate(rehearsal, args.phase)
        (directory / "gate_result.json").write_text(
            json.dumps(gate.to_dict(), indent=2), encoding="utf-8"
        )
        (directory / "report.md").write_text(
            render_markdown(rehearsal, gate), encoding="utf-8"
        )
        gate_payload = gate.to_dict()
        metrics_run_id = manifest.run_id
        forecast["metrics_run_id"] = metrics_run_id
        forecast["gate"] = gate_payload
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(forecast, indent=2, sort_keys=True), encoding="utf-8")
    if metrics_run_id is not None:
        print(json.dumps({"metrics_run_id": metrics_run_id, "gate": gate_payload}, indent=2))
    print(json.dumps(forecast, indent=2, sort_keys=True))
    if gate_payload is not None:
        return 0 if gate_payload["passed"] else 2
    return 0 if forecast["holdout"]["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
