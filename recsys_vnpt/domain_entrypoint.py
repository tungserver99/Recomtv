from __future__ import annotations

import argparse
import json
from pathlib import Path

from recsys_vnpt.engagement import (
    DEFAULT_SESSION_GAP_MINUTES,
    DEFAULT_TARGET_VERSION,
)
from recsys_vnpt.gbdt import GBDT_BACKENDS, GBDT_DEVICES
from recsys_vnpt.orderings import add_ordering_arguments, apply_combination
from recsys_vnpt.implicit_baselines import IMPLICIT_BASELINE_FAMILIES
from recsys_vnpt.rule_assembly import PAGE_DEDUP_POLICIES
from recsys_vnpt.paths import data_artifact_path
from recsys_vnpt.pipeline_v2 import (
    build_temporal_project,
    evaluate_popularity_baseline,
    evaluate_temporal_model,
    popularity_live_test,
    prepare_canonical_project,
    train_temporal_model,
)


MODEL_FAMILIES = (
    "dense_repaired_baseline",
    "embedding_mlp_baseline",
    "legacy_embedding_dcnv3",
    "fcn_official",
    "ecn_official",
    "hierarchical_fcn",
    "embedding_fcn",
    "embedding_dcnv3",
)


def _add_raw_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--from-raw", action="store_true")
    parser.add_argument("--max-interaction-rows", type=int)
    parser.add_argument("--max-behavior-rows", type=int)
    parser.add_argument("--max-profiles", type=int)
    parser.add_argument("--max-contents", type=int)
    parser.add_argument("--skip-behavior", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metadata-as-of", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--target-version", default=DEFAULT_TARGET_VERSION)
    parser.add_argument(
        "--session-gap-minutes", type=int, default=DEFAULT_SESSION_GAP_MINUTES
    )
    parser.add_argument(
        "--sequence-mode", choices=("none", "historical"), default="none"
    )
    parser.add_argument(
        "--window-mode", choices=("pinned", "relative"), default="pinned"
    )


def _add_gbdt_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gbdt-backend", choices=GBDT_BACKENDS, default="xgboost")
    parser.add_argument("--gbdt-device", choices=GBDT_DEVICES, default="auto")


def _prepare_if_requested(args: argparse.Namespace, domain: str, root: Path) -> None:
    if args.from_raw:
        prepare_canonical_project(
            root,
            domain,
            max_interaction_rows=args.max_interaction_rows,
            max_behavior_rows=args.max_behavior_rows,
            include_behavior=not args.skip_behavior,
            force=args.force_prepare,
            target_version=args.target_version,
            session_gap_minutes=args.session_gap_minutes,
        )
        build_temporal_project(
            root,
            domain,
            max_profiles=args.max_profiles,
            max_contents=args.max_contents,
            seed=args.seed,
            metadata_as_of=args.metadata_as_of,
            target_version=args.target_version,
            sequence_mode=args.sequence_mode,
            window_mode=args.window_mode,
        )


def main(domain: str, operation: str, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"RecSys-VNPT shared v2 {operation} entrypoint for {domain}."
    )
    parser.add_argument("--root", default=".")

    if operation == "train":
        parser.add_argument("--model-family", choices=MODEL_FAMILIES, default="fcn_official")
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--batch-size", type=int, default=4096)
        parser.add_argument(
            "--checkpoint-selection",
            choices=("hierarchical", "log_loss"),
            default="hierarchical",
        )
        parser.add_argument("--rule-loss-weight", type=float, default=0.4)
        parser.add_argument("--content-loss-weight", type=float, default=0.4)
        parser.add_argument("--auxiliary-loss-weight", type=float, default=0.2)
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--require-cuda", action="store_true")
        parser.add_argument("--amp", action="store_true")
        parser.add_argument("--hybrid-retrieval", action="store_true")
        parser.add_argument("--hybrid-epochs", type=int, default=5)
        parser.add_argument("--hybrid-embedding-dim", type=int, default=64)
        parser.add_argument("--hybrid-holdout-days", type=int)
        parser.add_argument("--sequence-epochs", type=int, default=3)
        _add_raw_options(parser)
    elif operation == "evaluate":
        parser.add_argument(
            "--model-family",
            choices=("per_rule_popularity", *IMPLICIT_BASELINE_FAMILIES, *MODEL_FAMILIES),
            default="fcn_official",
        )
        parser.add_argument("--run-id")
        parser.add_argument("--batch-size", type=int, default=4096)
        parser.add_argument("--amp", action="store_true")
        parser.add_argument(
            "--eval-serving-candidates",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        parser.add_argument("--candidate-budget", type=int, default=1000)
        parser.add_argument("--exact-cowatch", action="store_true")
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--embedding-dim", type=int, default=32)
        parser.add_argument("--fusion-bpr-weight", type=float, default=1.0)
        parser.add_argument("--baseline-batch-size", type=int, default=8192)
        parser.add_argument("--device")
        parser.add_argument("--hybrid-retrieval", action="store_true")
        parser.add_argument("--hybrid-epochs", type=int, default=5)
        parser.add_argument("--hybrid-embedding-dim", type=int, default=64)
        parser.add_argument("--hybrid-holdout-days", type=int)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument(
            "--sequence-mode", choices=("none", "historical"), default="none"
        )
        parser.add_argument("--sequence-epochs", type=int, default=3)
        parser.add_argument(
            "--external-candidate-mode",
            choices=("rerank", "replace"),
            default="rerank",
        )
        parser.add_argument("--external-candidate-slots", type=int, default=100)
        parser.add_argument("--external-rrf-weight", type=float, default=0.25)
        parser.add_argument(
            "--external-replacement-requires-witness",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        parser.add_argument("--content-external-rrf-weight", type=float, default=0.0)
        parser.add_argument("--content-rrf-constant", type=int, default=60)
        parser.add_argument(
            "--rule-affinity", choices=("count", "gain"), default="count"
        )
        parser.add_argument("--rule-affinity-half-life-days", type=float)
        add_ordering_arguments(parser)
        parser.add_argument("--contents-per-rule-max", type=int, default=30)
        parser.add_argument("--page-content-budget", type=int, default=600)
        parser.add_argument(
            "--page-dedup",
            choices=PAGE_DEDUP_POLICIES,
            default="balanced_regret",
        )
        parser.add_argument("--recency-candidates-per-rule", type=int, default=8)
        parser.add_argument("--recency-reserved-per-rule", type=int, default=8)
        parser.add_argument("--recency-max-age-days", type=int, default=30)
        parser.add_argument("--cowatch-candidates-per-rule", type=int, default=0)
        parser.add_argument("--tag-candidates-per-rule", type=int, default=0)
        parser.add_argument("--candidate-rules", type=int, default=30)
        parser.add_argument("--reserved-per-rule", type=int, default=33)
        parser.add_argument("--source-candidates-per-rule", type=int, default=33)
        parser.add_argument(
            "--evaluation-split",
            choices=("validation", "test"),
            default="test",
        )
        parser.add_argument("--max-eval-profiles", type=int)
        parser.add_argument("--eval-profile-seed", type=int, default=42)
        _add_gbdt_options(parser)
    elif operation == "infer":
        parser.add_argument("--model-family", choices=("per_rule_popularity", *MODEL_FAMILIES), default="per_rule_popularity")
        parser.add_argument(
            "--num-profiles",
            "--num-users",
            dest="num_profiles",
            type=int,
            default=1000,
            help="0 serves every profile",
        )
        parser.add_argument("--num-contents", "--num-items", dest="num_contents", type=int)
        parser.add_argument("--top-n", type=int, default=200, help=argparse.SUPPRESS)
        parser.add_argument("--warm-repetitions", type=int, default=3)
        parser.add_argument("--profile-chunk-size", type=int, default=128)
        parser.add_argument("--batch-size", type=int, default=4096)
        parser.add_argument("--amp", action="store_true")
        parser.add_argument("--output-dir")
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--train-if-missing", action="store_true")
        parser.add_argument("--no-gpu-sampling", action="store_true")
        parser.add_argument("--skip-offline-eval", action="store_true")
        parser.add_argument("--run-id")
        parser.add_argument("--candidate-budget", type=int, default=1000)
        parser.add_argument("--exact-cowatch", action="store_true")
        add_ordering_arguments(parser)
        parser.add_argument("--contents-per-rule-max", type=int, default=30)
        parser.add_argument("--page-content-budget", type=int, default=600)
        parser.add_argument(
            "--page-dedup",
            choices=PAGE_DEDUP_POLICIES,
            default="balanced_regret",
        )
        parser.add_argument("--recency-candidates-per-rule", type=int, default=8)
        parser.add_argument("--recency-reserved-per-rule", type=int, default=8)
        parser.add_argument("--recency-max-age-days", type=int, default=30)
        parser.add_argument("--candidate-rules", type=int, default=30)
        parser.add_argument("--reserved-per-rule", type=int, default=33)
        parser.add_argument("--source-candidates-per-rule", type=int, default=33)
        _add_gbdt_options(parser)
        _add_raw_options(parser)
    else:
        raise ValueError(f"Unsupported domain operation: {operation}")

    args = apply_combination(parser.parse_args(argv))
    root = Path(args.root).resolve()
    if operation in {"train", "infer"}:
        _prepare_if_requested(args, domain, root)

    if operation == "train":
        temporal = data_artifact_path("canonical", domain, "temporal", project_root=root)
        if not temporal.exists():
            raise FileNotFoundError(
                f"Missing temporal artifacts at {temporal}. Pass --from-raw or run build-temporal first."
            )
        bundle = train_temporal_model(
            root,
            domain,
            args.model_family,
            resume=args.resume,
            epochs=args.epochs,
            batch_size=args.batch_size,
            require_cuda=args.require_cuda,
            amp=args.amp,
            seed=args.seed,
            checkpoint_selection=args.checkpoint_selection,
            rule_loss_weight=args.rule_loss_weight,
            content_loss_weight=args.content_loss_weight,
            auxiliary_loss_weight=args.auxiliary_loss_weight,
            hybrid_retrieval=args.hybrid_retrieval,
            hybrid_epochs=args.hybrid_epochs,
            hybrid_embedding_dim=args.hybrid_embedding_dim,
            hybrid_holdout_days=args.hybrid_holdout_days,
            sequence_mode=args.sequence_mode,
            sequence_epochs=args.sequence_epochs,
        )
        payload = {"run_id": bundle.manifest.run_id, "training": bundle.training}
    elif operation == "evaluate":
        if args.model_family == "per_rule_popularity" or args.model_family in IMPLICIT_BASELINE_FAMILIES:
            bundle = evaluate_popularity_baseline(
                root,
                domain,
                eval_serving_candidates=args.eval_serving_candidates,
                candidate_budget=args.candidate_budget,
                exact_cowatch=args.exact_cowatch,
                baseline_family=args.model_family,
                epochs=args.epochs,
                embedding_dim=args.embedding_dim,
                batch_size=args.baseline_batch_size,
                fusion_bpr_weight=args.fusion_bpr_weight,
                seed=args.seed,
                device=args.device,
                hybrid_retrieval=args.hybrid_retrieval,
                hybrid_epochs=args.hybrid_epochs,
                hybrid_embedding_dim=args.hybrid_embedding_dim,
                hybrid_holdout_days=args.hybrid_holdout_days,
                sequence_mode=args.sequence_mode,
                sequence_epochs=args.sequence_epochs,
                evaluation_split=args.evaluation_split,
                maximum_evaluation_profiles=args.max_eval_profiles,
                evaluation_profile_seed=args.eval_profile_seed,
                external_candidate_mode=args.external_candidate_mode,
                external_candidate_slots=args.external_candidate_slots,
                external_rrf_weight=args.external_rrf_weight,
                external_replacement_requires_witness=(
                    args.external_replacement_requires_witness
                ),
                content_external_rrf_weight=args.content_external_rrf_weight,
                content_rrf_constant=args.content_rrf_constant,
                recency_candidates_per_rule=args.recency_candidates_per_rule,
                recency_reserved_per_rule=args.recency_reserved_per_rule,
                recency_max_age_days=args.recency_max_age_days,
                cowatch_candidates_per_rule=args.cowatch_candidates_per_rule,
                tag_candidates_per_rule=args.tag_candidates_per_rule,
                candidate_rules=args.candidate_rules,
                reserved_per_rule=args.reserved_per_rule,
                source_candidates_per_rule=args.source_candidates_per_rule,
                rule_affinity=args.rule_affinity,
                rule_affinity_half_life_days=args.rule_affinity_half_life_days,
                rule_ordering=args.rule_ordering,
                rule_scorer_fusion_weight=args.rule_scorer_fusion_weight,
                content_ordering=args.content_ordering,
                content_scorer_fusion_weight=args.content_scorer_fusion_weight,
                contents_per_rule_max=args.contents_per_rule_max,
                page_content_budget=args.page_content_budget,
                page_dedup=args.page_dedup,
                gbdt_backend=args.gbdt_backend,
                gbdt_device=args.gbdt_device,
                content_rank_scope=args.content_rank_scope,
                content_fcn_family=args.content_fcn_family,
                routed_fallback=args.routed_fallback,
                fcn_model_dir=args.fcn_model_dir,
            )
        else:
            bundle = evaluate_temporal_model(
                root,
                domain,
                args.model_family,
                run_id=args.run_id,
                batch_size=args.batch_size,
                amp=args.amp,
                eval_serving_candidates=args.eval_serving_candidates,
                candidate_budget=args.candidate_budget,
                exact_cowatch=args.exact_cowatch,
                hybrid_retrieval=args.hybrid_retrieval,
                hybrid_epochs=args.hybrid_epochs,
                hybrid_embedding_dim=args.hybrid_embedding_dim,
                seed=args.seed,
                sequence_mode=args.sequence_mode,
                sequence_epochs=args.sequence_epochs,
                evaluation_split=args.evaluation_split,
                recency_candidates_per_rule=args.recency_candidates_per_rule,
                recency_reserved_per_rule=args.recency_reserved_per_rule,
                recency_max_age_days=args.recency_max_age_days,
                gbdt_backend=args.gbdt_backend,
                gbdt_device=args.gbdt_device,
                candidate_rules=args.candidate_rules,
                reserved_per_rule=args.reserved_per_rule,
                source_candidates_per_rule=args.source_candidates_per_rule,
                maximum_evaluation_profiles=args.max_eval_profiles,
                evaluation_profile_seed=args.eval_profile_seed,
                rule_affinity=args.rule_affinity,
                rule_affinity_half_life_days=args.rule_affinity_half_life_days,
                rule_ordering=args.rule_ordering,
                rule_scorer_fusion_weight=args.rule_scorer_fusion_weight,
                content_ordering=args.content_ordering,
                content_scorer_fusion_weight=args.content_scorer_fusion_weight,
                contents_per_rule_max=args.contents_per_rule_max,
                page_content_budget=args.page_content_budget,
                page_dedup=args.page_dedup,
                content_rank_scope=args.content_rank_scope,
                content_fcn_family=args.content_fcn_family,
                routed_fallback=args.routed_fallback,
                fcn_model_dir=args.fcn_model_dir,
            )
        payload = {"run_id": bundle.manifest.run_id, "evaluation": bundle.evaluation}
    else:
        bundle = popularity_live_test(
            root,
            domain,
            num_profiles=args.num_profiles,
            warm_repetitions=args.warm_repetitions,
            output_dir=args.output_dir,
            model_family=args.model_family,
            num_contents=args.num_contents,
            profile_chunk_size=args.profile_chunk_size,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            resume=args.resume,
            train_if_missing=args.train_if_missing,
            run_id=args.run_id,
            amp=args.amp,
            sample_gpu=not args.no_gpu_sampling,
            candidate_budget=args.candidate_budget,
            exact_cowatch=args.exact_cowatch,
            rule_ordering=args.rule_ordering,
            rule_scorer_fusion_weight=args.rule_scorer_fusion_weight,
            content_ordering=args.content_ordering,
            content_scorer_fusion_weight=args.content_scorer_fusion_weight,
            contents_per_rule_max=args.contents_per_rule_max,
            page_content_budget=args.page_content_budget,
            page_dedup=args.page_dedup,
            recency_candidates_per_rule=args.recency_candidates_per_rule,
            recency_reserved_per_rule=args.recency_reserved_per_rule,
            recency_max_age_days=args.recency_max_age_days,
            skip_offline_eval=args.skip_offline_eval,
            candidate_rules=args.candidate_rules,
            reserved_per_rule=args.reserved_per_rule,
            source_candidates_per_rule=args.source_candidates_per_rule,
            gbdt_backend=args.gbdt_backend,
            gbdt_device=args.gbdt_device,
            content_rank_scope=args.content_rank_scope,
            content_fcn_family=args.content_fcn_family,
            routed_fallback=args.routed_fallback,
            fcn_model_dir=args.fcn_model_dir,
        )
        payload = {
            "run_id": bundle.manifest.run_id,
            "inference": bundle.inference,
            "output": bundle.output,
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
