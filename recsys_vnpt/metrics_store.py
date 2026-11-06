from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from recsys_vnpt.metrics import paired_bootstrap_confidence_interval
from recsys_vnpt.metrics_schema import GateCheck, GateResult, MetricBundle, RunManifest
from recsys_vnpt.paths import data_artifact_path


SECTION_FILES = {
    "data": "data_metrics.json",
    "training": "training_metrics.json",
    "evaluation": "evaluation_metrics.json",
    "inference": "inference_metrics.json",
    "output": "output_metrics.json",
}


def new_run_id(domain: str, model_family: str, now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}_{domain}_{model_family}"


def _git(command: list[str], root: Path) -> str:
    try:
        return subprocess.run(command, cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def hardware_snapshot() -> dict[str, Any]:
    payload: dict[str, Any] = {"platform": platform.platform(), "processor": platform.processor()}
    try:
        import torch

        payload.update(
            {
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "cuda_runtime": torch.version.cuda,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "gpu_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else 0,
            }
        )
    except ImportError:
        payload["device"] = "torch-not-installed"
    return payload


def software_snapshot() -> dict[str, str]:
    values = {"python": platform.python_version()}
    for name in ("pandas", "numpy", "sklearn", "torch", "polars"):
        try:
            module = __import__(name)
            values[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            values[name] = "not-installed"
    return values


def create_manifest(
    root: str | Path,
    domain: str,
    phase: str,
    model_family: str,
    run_id: str | None = None,
    **kwargs: Any,
) -> RunManifest:
    project = Path(root).resolve()
    return RunManifest(
        run_id=run_id or new_run_id(domain, model_family),
        domain=domain,
        phase=phase,
        model_family=model_family,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=_git(["git", "rev-parse", "HEAD"], project),
        dirty_worktree=bool(_git(["git", "status", "--porcelain"], project).strip()),
        hardware=hardware_snapshot(),
        software=software_snapshot(),
        **kwargs,
    )


class MetricsStore:
    def __init__(self, project_root: str | Path = ".", metrics_root: str | Path | None = None) -> None:
        self.project_root = Path(project_root).resolve()
        self.root = Path(metrics_root) if metrics_root is not None else data_artifact_path("metrics", project_root=self.project_root)

    def run_dir(self, run_id: str, domain: str) -> Path:
        return self.root / run_id / domain

    def resolve_run_id(self, run_id: str, domain: str | None = None) -> str:
        if run_id != "latest":
            return run_id
        pointer = self.root / "latest.json"
        if not pointer.exists():
            raise FileNotFoundError(f"No latest metrics pointer at {pointer}")
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        if domain is not None and domain in payload.get("domains", {}):
            return str(payload["domains"][domain])
        return str(payload["run_id"])

    def write_bundle(self, bundle: MetricBundle, make_latest: bool = True) -> Path:
        bundle.validate()
        directory = self.run_dir(bundle.manifest.run_id, bundle.manifest.domain)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "run_manifest.json").write_text(json.dumps(bundle.manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        for section, filename in SECTION_FILES.items():
            (directory / filename).write_text(json.dumps(getattr(bundle, section), indent=2, sort_keys=True), encoding="utf-8")
        if make_latest:
            self.root.mkdir(parents=True, exist_ok=True)
            pointer = self.root / "latest.json"
            current = json.loads(pointer.read_text(encoding="utf-8")) if pointer.exists() else {}
            domains = dict(current.get("domains", {}))
            domains[bundle.manifest.domain] = bundle.manifest.run_id
            (self.root / "latest.json").write_text(
                json.dumps(
                    {
                        "run_id": bundle.manifest.run_id,
                        "domain": bundle.manifest.domain,
                        "domains": domains,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        return directory

    def read_bundle(self, run_id: str, domain: str | None = None) -> MetricBundle:
        if run_id == "legacy":
            if domain is None:
                raise ValueError("domain is required for legacy metrics")
            return self.read_legacy_bundle(domain)
        candidate_path = Path(run_id)
        if candidate_path.suffix.lower() == ".json" and candidate_path.exists():
            if domain is None:
                domain = "all"
            payload = json.loads(candidate_path.read_text(encoding="utf-8"))
            manifest = create_manifest(
                self.project_root,
                domain,
                "legacy",
                str(payload.get("model_family", "legacy_flat_metrics")),
                run_id=f"legacy_{domain}",
            )
            return MetricBundle(manifest=manifest, training=payload)
        resolved = self.resolve_run_id(run_id, domain)
        if domain is None and run_id == "latest":
            domain = json.loads((self.root / "latest.json").read_text(encoding="utf-8"))["domain"]
        if domain is None:
            domains = [path.name for path in (self.root / resolved).iterdir() if path.is_dir()]
            if len(domains) != 1:
                raise ValueError("domain is required when a run contains zero or multiple domains")
            domain = domains[0]
        directory = self.run_dir(resolved, domain)
        manifest = RunManifest.from_dict(json.loads((directory / "run_manifest.json").read_text(encoding="utf-8")))
        sections = {
            section: json.loads((directory / filename).read_text(encoding="utf-8")) if (directory / filename).exists() else {}
            for section, filename in SECTION_FILES.items()
        }
        return MetricBundle(manifest=manifest, **sections)

    def read_legacy_bundle(self, domain: str) -> MetricBundle:
        training_path = data_artifact_path(
            "model", domain, "training_metrics.json", project_root=self.project_root
        )
        inference_candidates = (
            self.project_root / domain / "result" / "inference_metrics.json",
            self.project_root / domain / "result_v2" / "inference_metrics.json",
        )
        training = json.loads(training_path.read_text(encoding="utf-8")) if training_path.exists() else {}
        inference_path = next((path for path in inference_candidates if path.exists()), None)
        inference = json.loads(inference_path.read_text(encoding="utf-8")) if inference_path else {}
        if not training and not inference:
            raise FileNotFoundError(f"No legacy metrics found for domain {domain}")
        family = str(training.get("model_family", inference.get("model_family", "legacy_flat_metrics")))
        manifest = create_manifest(
            self.project_root,
            domain,
            "legacy",
            family,
            run_id=f"legacy_{domain}",
        )
        manifest.hardware = {
            "device": "unknown",
            "provenance": "legacy metrics did not record hardware",
        }
        manifest.config["legacy_imported_at_read_time"] = True
        return MetricBundle(manifest=manifest, training=training, inference=inference)


def _flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            output.update(_flatten(value, name))
        else:
            output[name] = value
    return output


def flattened_bundle(bundle: MetricBundle) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for section in SECTION_FILES:
        values.update(_flatten(getattr(bundle, section), section))
    return values


def _relative(candidate: float, baseline: float) -> float:
    return (candidate - baseline) / abs(baseline) if baseline else (1.0 if candidate > 0 else 0.0)


def evaluate_gate(bundle: MetricBundle, phase: str, baseline: MetricBundle | None = None) -> GateResult:
    current = flattened_bundle(bundle)
    previous = flattened_bundle(baseline) if baseline else {}
    checks: list[GateCheck] = []

    def absolute(metric: str, predicate, expected: str) -> None:
        actual = current.get(metric)
        passed = actual is not None and predicate(float(actual))
        checks.append(GateCheck(metric, passed, actual, expected))

    def relative(metric: str, minimum: float) -> None:
        actual, old = current.get(metric), previous.get(metric)
        passed = actual is not None and old is not None and _relative(float(actual), float(old)) >= minimum
        checks.append(GateCheck(metric, passed, actual, f">= {minimum:.1%} relative vs {old}"))

    def retained(metric: str, minimum_ratio: float) -> None:
        actual, old = current.get(metric), previous.get(metric)
        passed = (
            actual is not None
            and old is not None
            and float(actual) >= float(old) * minimum_ratio
        )
        checks.append(
            GateCheck(
                metric,
                passed,
                actual,
                f">= {minimum_ratio:.1%} of {old}",
            )
        )

    def content_shape(section: str) -> None:
        """Every served rule carries a valid number of contents.

        A fixed 20x10 page satisfies this through exact_10_content_rate; a
        variable page (10..30 contents per rule) satisfies it through
        content_count_within_range_rate. Both express the same contract, so
        either passing is enough.
        """

        exact = current.get(f"{section}.exact_10_content_rate")
        in_range = current.get(f"{section}.content_count_within_range_rate")
        passed = (exact is not None and float(exact) == 1.0) or (
            in_range is not None and float(in_range) == 1.0
        )
        checks.append(
            GateCheck(
                f"{section}.content_count_contract",
                passed,
                exact if in_range is None else in_range,
                "exact_10_content_rate == 1.0 or content_count_within_range_rate == 1.0",
            )
        )

    def present(metric: str) -> None:
        actual = current.get(metric)
        checks.append(GateCheck(metric, actual is not None, actual, "present"))

    def evaluation_profiles_per_second(values: dict[str, Any]) -> float | None:
        explicit = values.get("inference.profiles_per_second")
        if explicit is not None:
            return float(explicit)
        profiles = values.get("evaluation.candidate_profiles")
        seconds = values.get("inference.evaluation_end_to_end_time_seconds")
        if profiles is None or seconds is None or float(seconds) <= 0:
            return None
        return float(profiles) / float(seconds)

    if phase == "label-l0":
        absolute("data.duplicate_leakage", lambda value: value == 0, "== 0")
        absolute("data.duplicate_event_hashes", lambda value: value == 0, "== 0")
        absolute("data.playback_events", lambda value: value > 0, "> 0")
        present("data.label_contract_hash")
        present("data.target_version")
        present("data.source_interaction_rows")
        present("data.playback_source_rows")
        present("data.playback_source_rate")
        present("data.interaction_catalog_audit.by_catalog_class.eligible.rows")
        present(
            "data.interaction_catalog_audit.by_catalog_class.historical_version_only.rows"
        )
        present("data.interaction_catalog_audit.by_catalog_class.unknown.rows")
        present("data.interaction_catalog_audit.eligible_missing_start_time_rate")
        present("data.interaction_catalog_audit.eligible_duration_over_24h_rows")
    elif phase == "label-l1":
        present("evaluation.evaluation_truth_gain_hash")
        for gain_column in ("completion_gain", "watch_time_gain"):
            present(f"evaluation.{gain_column}_rule_ndcg@20")
            present(f"evaluation.{gain_column}_macro_content_ndcg@10")
        actual = current.get("evaluation.classification_metrics_valid")
        checks.append(
            GateCheck(
                "evaluation.classification_metrics_valid",
                actual is False,
                actual,
                "== false",
            )
        )
    elif phase == "label-l2":
        for gain_column in ("completion_gain", "watch_time_gain"):
            absolute(
                f"evaluation.{gain_column}_rule_ndcg@20",
                lambda value: 0 <= value <= 1,
                "in [0, 1]",
            )
            absolute(
                f"evaluation.{gain_column}_macro_content_ndcg@10",
                lambda value: 0 <= value <= 1,
                "in [0, 1]",
            )
        absolute(
            "evaluation.exact_20_normal_rule_rate",
            lambda value: value == 1.0,
            "== 1.0",
        )
        content_shape("evaluation")
    elif phase == "label-l3":
        binary = current.get("evaluation.candidate_recall@1000")
        gain_values = [
            current.get(
                f"evaluation.candidate_{gain_column}_recall@1000"
            )
            for gain_column in ("completion_gain", "watch_time_gain")
        ]
        passed = binary is not None and float(binary) >= 0.95 and (
            float(binary) >= 0.98
            or all(value is not None and float(value) >= 0.98 for value in gain_values)
        )
        checks.append(
            GateCheck(
                "evaluation.candidate_recall_or_gain_recall@1000",
                passed,
                {
                    "binary": binary,
                    "completion_gain": gain_values[0],
                    "watch_time_gain": gain_values[1],
                },
                "binary >= 0.98, or both gains >= 0.98 with binary >= 0.95",
            )
        )
    elif phase in {"label-l4", "label-l5"}:
        # watch_time_gain is the primary, family-fair promotion target. completion_gain
        # is length/family-biased (movie completion median ~0.06 vs music ~1.0), so it
        # is reported as a diagnostic (render_markdown) but does not gate promotion.
        for gain_column in ("watch_time_gain",):
            for metric in ("rule_ndcg@20", "macro_content_ndcg@10"):
                relative(f"evaluation.{gain_column}_{metric}", 0.03)
        if baseline:
            retained("evaluation.overall_recall@200", 0.995)
            for gain_column in ("watch_time_gain",):
                retained(
                    f"evaluation.{gain_column}_all_rule_macro_content_ndcg@10",
                    0.995,
                )
            baseline_profiles = baseline.evaluation.get("per_profile", {})
            candidate_profiles = bundle.evaluation.get("per_profile", {})
            for gain_column in ("watch_time_gain",):
                for suffix in ("rule_ndcg@20", "macro_content_ndcg@10"):
                    metric = f"{gain_column}_{suffix}"
                    if baseline_profiles and candidate_profiles:
                        interval = paired_bootstrap_confidence_interval(
                            {
                                profile: float(values[metric])
                                for profile, values in baseline_profiles.items()
                                if metric in values
                            },
                            {
                                profile: float(values[metric])
                                for profile, values in candidate_profiles.items()
                                if metric in values
                            },
                        )
                        checks.append(
                            GateCheck(
                                f"evaluation.paired_bootstrap.{metric}.lower",
                                float(interval["lower"]) > 0,
                                interval["lower"],
                                "> 0",
                            )
                        )
                    else:
                        checks.append(
                            GateCheck(
                                f"evaluation.paired_bootstrap.{metric}.lower",
                                False,
                                0,
                                "> 0 with paired per-profile evidence",
                            )
                        )
        absolute(
            "evaluation.exact_20_normal_rule_rate",
            lambda value: value == 1.0,
            "== 1.0",
        )
        content_shape("evaluation")
    elif phase == "label-l4-content":
        # The mirror image of label-l4, for a change that reorders content
        # inside a rule without touching which rules are shown. label-l4 demands
        # a rule-NDCG rise, so a content-only change fails it by construction
        # rather than by regressing; here content has to rise and rule ordering
        # has to be retained.
        relative("evaluation.watch_time_gain_macro_content_ndcg@10", 0.03)
        if baseline:
            retained("evaluation.watch_time_gain_rule_ndcg@20", 0.995)
            retained("evaluation.overall_recall@200", 0.995)
            retained(
                "evaluation.watch_time_gain_all_rule_macro_content_ndcg@10", 0.995
            )
            metric = "watch_time_gain_macro_content_ndcg@10"
            baseline_profiles = baseline.evaluation.get("per_profile", {})
            candidate_profiles = bundle.evaluation.get("per_profile", {})
            if baseline_profiles and candidate_profiles:
                interval = paired_bootstrap_confidence_interval(
                    {
                        profile: float(values[metric])
                        for profile, values in baseline_profiles.items()
                        if metric in values
                    },
                    {
                        profile: float(values[metric])
                        for profile, values in candidate_profiles.items()
                        if metric in values
                    },
                )
                checks.append(
                    GateCheck(
                        f"evaluation.paired_bootstrap.{metric}.lower",
                        float(interval["lower"]) > 0,
                        interval["lower"],
                        "> 0",
                    )
                )
            else:
                checks.append(
                    GateCheck(
                        f"evaluation.paired_bootstrap.{metric}.lower",
                        False,
                        0,
                        "> 0 with paired per-profile evidence",
                    )
                )
        absolute(
            "evaluation.exact_20_normal_rule_rate",
            lambda value: value == 1.0,
            "== 1.0",
        )
        content_shape("evaluation")
    elif phase == "label-l6":
        absolute(
            "output.exact_20_normal_rule_rate",
            lambda value: value == 1.0,
            "== 1.0",
        )
        content_shape("output")
        absolute(
            "inference.processing_error_count",
            lambda value: value == 0,
            "== 0",
        )
    elif phase == "sprint-0":
        absolute("data.duplicate_canonical_ids", lambda value: value == 0, "== 0")
        absolute("data.tombstone_leakage", lambda value: value == 0, "== 0")
        absolute("data.missing_required_columns", lambda value: value == 0, "== 0")
        absolute("data.silent_skipped_files", lambda value: value == 0, "== 0")
        absolute("data.eligible_normal_rules", lambda value: value >= 20, ">= 20")
    elif phase == "sprint-1":
        content_shape("output")
        for metric in ("output.exact_20_normal_rule_rate", "output.special_rule_complete_rate", "output.cross_file_consistency_rate"):
            absolute(metric, lambda value: value == 1.0, "== 1.0")
        for metric in ("output.unmatched_rule_rate", "output.invalid_content_rate", "output.duplicate_within_rule_rate"):
            absolute(metric, lambda value: value == 0.0, "== 0.0")
    elif phase == "sprint-2":
        relative("evaluation.rule_ndcg@20", 0.02)
        relative("evaluation.macro_content_ndcg@10", 0.02)
        if baseline:
            old = previous.get("evaluation.overall_recall@200")
            actual = current.get("evaluation.overall_recall@200")
            checks.append(
                GateCheck(
                    "evaluation.overall_recall@200",
                    actual is not None and old is not None and float(actual) >= float(old),
                    actual,
                    f">= {old}",
                )
            )
        content_shape("output")
        for metric in (
            "output.exact_20_normal_rule_rate",
            "output.special_rule_complete_rate",
            "output.cross_file_consistency_rate",
        ):
            absolute(metric, lambda value: value == 1.0, "== 1.0")
        for metric in (
            "output.unmatched_rule_rate",
            "output.invalid_content_rate",
            "output.duplicate_within_rule_rate",
        ):
            absolute(metric, lambda value: value == 0.0, "== 0.0")
    elif phase in {"sprint-3", "sprint-4"}:
        absolute(
            "evaluation.retrievable_truth_profiles",
            lambda value: value >= 400,
            ">= 400",
        )
        absolute(
            "evaluation.retrievable_positive_pairs",
            lambda value: value >= 600,
            ">= 600",
        )
        absolute(
            "evaluation.candidate_recall@1000",
            lambda value: value >= 0.98,
            ">= 0.98",
        )
        absolute(
            "evaluation.consumed_candidate_overlap",
            lambda value: value == 0,
            "== 0",
        )
        absolute(
            "evaluation.candidate_max_actual_k",
            lambda value: value <= 1000,
            "<= 1000",
        )
        if baseline:
            current_truth_hash = current.get(
                "evaluation.evaluation_truth_pairs_hash"
            )
            baseline_truth_hash = previous.get(
                "evaluation.evaluation_truth_pairs_hash"
            )
            checks.append(
                GateCheck(
                    "evaluation.evaluation_truth_pairs_hash",
                    current_truth_hash is not None
                    and baseline_truth_hash is not None
                    and str(current_truth_hash) == str(baseline_truth_hash),
                    current_truth_hash,
                    f"== {baseline_truth_hash}",
                )
            )
        if phase == "sprint-3":
            relative("evaluation.rule_ndcg@20", 0.03)
            relative("evaluation.macro_content_ndcg@10", 0.03)
        else:
            retained("evaluation.rule_ndcg@20", 0.99)
            retained("evaluation.macro_content_ndcg@10", 0.99)
        if baseline and phase == "sprint-3":
            current_hash = current.get("evaluation.candidate_pairs_hash")
            baseline_hash = previous.get("evaluation.candidate_pairs_hash")
            checks.append(
                GateCheck(
                    "evaluation.candidate_pairs_hash",
                    current_hash is not None
                    and baseline_hash is not None
                    and str(current_hash) == str(baseline_hash),
                    current_hash,
                    f"== {baseline_hash}",
                )
            )
            retained("evaluation.overall_recall@200", 0.995)
            baseline_profiles = baseline.evaluation.get("per_profile", {})
            candidate_profiles = bundle.evaluation.get("per_profile", {})
            for metric in ("rule_ndcg@20", "macro_content_ndcg@10"):
                if baseline_profiles and candidate_profiles:
                    interval = paired_bootstrap_confidence_interval(
                        {
                            profile: float(values[metric])
                            for profile, values in baseline_profiles.items()
                            if metric in values
                        },
                        {
                            profile: float(values[metric])
                            for profile, values in candidate_profiles.items()
                            if metric in values
                        },
                    )
                    checks.append(
                        GateCheck(
                            f"evaluation.paired_bootstrap.{metric}.lower",
                            float(interval["lower"]) > 0,
                            interval["lower"],
                            "> 0",
                        )
                    )
        if baseline and phase == "sprint-4":
            retained("evaluation.overall_recall@200", 0.99)
            candidate_throughput = evaluation_profiles_per_second(current)
            baseline_throughput = evaluation_profiles_per_second(previous)
            throughput_ratio = (
                candidate_throughput / baseline_throughput
                if candidate_throughput is not None
                and baseline_throughput is not None
                and baseline_throughput > 0
                else None
            )
            checks.append(
                GateCheck(
                    "inference.evaluation_profiles_per_second_relative",
                    throughput_ratio is not None and throughput_ratio >= 5.0,
                    throughput_ratio,
                    f">= 5.0x vs {baseline_throughput}",
                )
            )
        absolute(
            "evaluation.exact_20_normal_rule_rate",
            lambda value: value == 1.0,
            "== 1.0",
        )
        content_shape("evaluation")
        if bundle.output:
            content_shape("output")
            for metric in (
                "output.exact_20_normal_rule_rate",
                "output.special_rule_complete_rate",
                "output.cross_file_consistency_rate",
            ):
                absolute(metric, lambda value: value == 1.0, "== 1.0")
            for metric in (
                "output.unmatched_rule_rate",
                "output.invalid_content_rate",
                "output.duplicate_within_rule_rate",
            ):
                absolute(metric, lambda value: value == 0.0, "== 0.0")
        if phase == "sprint-3" and bundle.training:
            absolute(
                "training.peak_gpu_memory_bytes",
                lambda value: value < 6 * 1024**3,
                "< 6 GiB",
            )
        if phase == "sprint-3":
            model_throughput = evaluation_profiles_per_second(current)
            checks.append(
                GateCheck(
                    "inference.evaluation_profiles_per_second",
                    model_throughput is not None and model_throughput >= 75,
                    model_throughput,
                    ">= 75",
                )
            )
    elif phase in {"sprint-5", "sprint-6"}:
        absolute("output.profile_output_coverage", lambda value: value == 1.0, "== 1.0")
        absolute("output.exact_20_normal_rule_rate", lambda value: value == 1.0, "== 1.0")
        content_shape("output")
        absolute("output.special_rule_complete_rate", lambda value: value == 1.0, "== 1.0")
        absolute("output.cross_file_consistency_rate", lambda value: value == 1.0, "== 1.0")
        absolute("output.invalid_content_rate", lambda value: value == 0.0, "== 0.0")
        absolute("output.unmatched_rule_rate", lambda value: value == 0.0, "== 0.0")
        absolute("inference.processing_error_count", lambda value: value == 0, "== 0")
        absolute("inference.forecast_error_rate", lambda value: value <= 0.10, "<= 0.10")
        if phase == "sprint-5":
            repetitions_metric = (
                "inference.ramp_warm_repetitions"
                if "inference.ramp_warm_repetitions" in current
                else "inference.warm_repetitions"
            )
            deterministic_metric = (
                "inference.ramp_deterministic_warm_hash_rate"
                if "inference.ramp_deterministic_warm_hash_rate" in current
                else "inference.deterministic_warm_hash_rate"
            )
            absolute(repetitions_metric, lambda value: value >= 3, ">= 3")
            absolute(deterministic_metric, lambda value: value == 1.0, "== 1.0")
    else:
        raise ValueError(f"Unknown metric gate phase: {phase}")
    return GateResult(phase=phase, passed=bool(checks) and all(check.passed for check in checks), checks=checks)


def render_markdown(bundle: MetricBundle, gate: GateResult | None = None) -> str:
    lines = [
        f"# Metrics Report: {bundle.manifest.run_id}",
        "",
        f"- Domain: `{bundle.manifest.domain}`",
        f"- Phase: `{bundle.manifest.phase}`",
        f"- Model: `{bundle.manifest.model_family}`",
        f"- Device: `{bundle.manifest.hardware.get('device', 'unknown')}`",
    ]
    flat = flattened_bundle(bundle)
    summary_metrics = (
        ("Candidate profiles", "evaluation.candidate_profiles"),
        ("Retrievable truth profiles", "evaluation.retrievable_truth_profiles"),
        ("Retrievable positives", "evaluation.retrievable_positive_pairs"),
        ("Candidate Recall@1,000", "evaluation.candidate_recall@1000"),
        (
            "Completion Gain Recall@1,000",
            "evaluation.candidate_completion_gain_recall@1000",
        ),
        (
            "Watch-time Gain Recall@1,000",
            "evaluation.candidate_watch_time_gain_recall@1000",
        ),
        (
            "Preblend Candidate Recall@1,000",
            "evaluation.preblend_incumbent_candidate_recall",
        ),
        (
            "External Candidate Recall@1,000",
            "evaluation.external_candidate_recall@1000",
        ),
        (
            "External incremental truth hits",
            "evaluation.external_incremental_truth_hits",
        ),
        (
            "External oracle-union truth hits",
            "evaluation.external_oracle_union_truth_hits",
        ),
        ("Blend gained truth hits", "evaluation.blend_gained_truth_hits"),
        ("Blend lost truth hits", "evaluation.blend_lost_truth_hits"),
        ("Blend net truth hits", "evaluation.blend_net_truth_hits"),
        (
            "Blend net completion gain",
            "evaluation.blend_net_completion_gain",
        ),
        (
            "Blend net watch-time gain",
            "evaluation.blend_net_watch_time_gain",
        ),
        (
            "Cold-profile Candidate Recall@1,000",
            "evaluation.retrieval_truth_attribution.cohorts.observed_history_profile_oov.true.candidate_recall",
        ),
        (
            "Cold-content Candidate Recall@1,000",
            "evaluation.retrieval_truth_attribution.cohorts.observed_history_content_oov.true.candidate_recall",
        ),
        (
            "Zero-history Candidate Recall@1,000",
            "evaluation.retrieval_truth_attribution.cohorts.activity_bucket.cold.candidate_recall",
        ),
        ("Rule NDCG@20", "evaluation.rule_ndcg@20"),
        ("Macro Content NDCG@10", "evaluation.macro_content_ndcg@10"),
        (
            "Completion Rule NDCG@20",
            "evaluation.completion_gain_rule_ndcg@20",
        ),
        (
            "Completion Content NDCG@10",
            "evaluation.completion_gain_macro_content_ndcg@10",
        ),
        (
            "Completion Eligible-rule-pool Content NDCG@10",
            "evaluation.completion_gain_all_rule_macro_content_ndcg@10",
        ),
        (
            "Watch-time Rule NDCG@20",
            "evaluation.watch_time_gain_rule_ndcg@20",
        ),
        (
            "Watch-time Content NDCG@10",
            "evaluation.watch_time_gain_macro_content_ndcg@10",
        ),
        (
            "Watch-time Eligible-rule-pool Content NDCG@10",
            "evaluation.watch_time_gain_all_rule_macro_content_ndcg@10",
        ),
        (
            "Eligible candidate rule-group rate",
            "evaluation.eligible_candidate_rule_group_rate",
        ),
        ("Hierarchical Recall@200", "evaluation.overall_recall@200"),
        ("Exact 20 normal rules", "evaluation.exact_20_normal_rule_rate"),
        ("Exact 10 content/rule", "evaluation.exact_10_content_rate"),
        ("Training seconds", "training.training_time_seconds"),
        ("Evaluation end-to-end seconds", "inference.evaluation_end_to_end_time_seconds"),
        ("Profiles/second", "inference.profiles_per_second"),
        ("Pairs/second", "inference.pairs_per_second"),
    )
    summary_rows = [
        (label, flat[metric])
        for label, metric in summary_metrics
        if metric in flat
    ]
    if summary_rows:
        lines.extend(
            [
                "",
                "## Summary",
                "",
                "| Metric | Value |",
                "|---|---:|",
                *[
                    f"| {label} | {value} |"
                    for label, value in summary_rows
                ],
            ]
        )
    for section in SECTION_FILES:
        section_payload = getattr(bundle, section)
        values = _flatten(section_payload)
        per_profile = section_payload.get("per_profile")
        if isinstance(per_profile, dict):
            values = {
                key: value
                for key, value in values.items()
                if not key.startswith("per_profile.")
            }
            values["per_profile_rows_in_json"] = len(per_profile)
        lines.extend(["", f"## {section.title()}", "", "| Metric | Value |", "|---|---:|"])
        lines.extend(f"| `{key}` | {value} |" for key, value in sorted(values.items()))
    if gate is not None:
        lines.extend(["", "## Gate", "", f"Overall: **{'PASS' if gate.passed else 'FAIL'}**", "", "| Metric | Result | Actual | Expected |", "|---|---|---:|---|"])
        lines.extend(f"| `{check.metric}` | {'PASS' if check.passed else 'FAIL'} | {check.actual} | {check.expected} |" for check in gate.checks)
    return "\n".join(lines) + "\n"
