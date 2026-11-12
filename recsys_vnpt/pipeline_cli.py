from __future__ import annotations

import argparse
import json

from recsys_vnpt.engagement import (
    DEFAULT_SESSION_GAP_MINUTES,
    DEFAULT_TARGET_VERSION,
)
from recsys_vnpt.gbdt import GBDT_BACKENDS, GBDT_DEVICES
from recsys_vnpt.implicit_baselines import IMPLICIT_BASELINE_FAMILIES
from recsys_vnpt.orderings import add_ordering_arguments, apply_combination
from recsys_vnpt.rule_assembly import PAGE_DEDUP_POLICIES

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


def _add_scoring_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags deciding which page a run is scored on, and how it is ordered.

    Shared by both evaluation subcommands so they cannot drift: a challenger
    scored on a different page shape than its baseline is not comparable.
    """

    # Depth over breadth: fewer rules, each retrieved deeper, at the same budget.
    parser.add_argument("--candidate-rules", type=int, default=30)
    parser.add_argument("--reserved-per-rule", type=int, default=33)
    parser.add_argument("--source-candidates-per-rule", type=int, default=33)
    parser.add_argument("--rule-affinity", choices=("count", "gain"), default="count")
    parser.add_argument("--rule-affinity-half-life-days", type=float)
    add_ordering_arguments(parser)
    parser.add_argument("--contents-per-rule-max", type=int, default=30)
    parser.add_argument("--page-content-budget", type=int, default=600)
    parser.add_argument(
        "--page-dedup", choices=PAGE_DEDUP_POLICIES, default="balanced_regret"
    )
    parser.add_argument(
        "--evaluation-split", choices=("validation", "test"), default="test"
    )
    parser.add_argument("--max-eval-profiles", type=int)
    parser.add_argument("--eval-profile-seed", type=int, default=42)
    _add_gbdt_arguments(parser)


def _add_gbdt_arguments(parser: argparse.ArgumentParser) -> None:
    """Library and device of the learned scorers, wherever a run fits them."""

    parser.add_argument("--gbdt-backend", choices=GBDT_BACKENDS, default="xgboost")
    parser.add_argument("--gbdt-device", choices=GBDT_DEVICES, default="auto")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Shared RecSys-VNPT v2 pipeline.")
    parser.add_argument("--root", default=".")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--domain", choices=("all", "clip", "movie", "music"), default="all")
    prepare.add_argument("--max-interaction-rows", type=int)
    prepare.add_argument("--max-behavior-rows", type=int)
    prepare.add_argument("--skip-behavior", action="store_true")
    prepare.add_argument("--force", action="store_true")
    prepare.add_argument("--full-source-hash", action="store_true")
    prepare.add_argument("--target-version", default=DEFAULT_TARGET_VERSION)
    prepare.add_argument(
        "--session-gap-minutes", type=int, default=DEFAULT_SESSION_GAP_MINUTES
    )
    prepare.add_argument("--user-reference-date", default="2025-02-20")

    temporal = sub.add_parser("build-temporal")
    temporal.add_argument("--domain", choices=("all", "clip", "movie", "music"), default="all")
    temporal.add_argument("--max-profiles", type=int)
    temporal.add_argument("--max-contents", type=int)
    temporal.add_argument("--seed", type=int, default=42)
    temporal.add_argument(
        "--metadata-as-of", action=argparse.BooleanOptionalAction, default=True
    )
    temporal.add_argument("--force", action="store_true")
    temporal.add_argument("--target-version", default=DEFAULT_TARGET_VERSION)
    temporal.add_argument(
        "--sequence-mode", choices=("none", "historical"), default="historical"
    )
    temporal.add_argument(
        "--window-mode", choices=("pinned", "relative"), default="pinned"
    )
    temporal.add_argument("--impression-weight", type=float, default=1.0)

    train = sub.add_parser("train")
    train.add_argument("--domain", choices=("all", "clip", "movie", "music"), default="all")
    train.add_argument("--model-family", choices=MODEL_FAMILIES, default="fcn_official")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=4096)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--checkpoint-selection",
        choices=("hierarchical", "log_loss"),
        default="hierarchical",
    )
    train.add_argument("--rule-loss-weight", type=float, default=0.4)
    train.add_argument("--content-loss-weight", type=float, default=0.4)
    train.add_argument("--auxiliary-loss-weight", type=float, default=0.2)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--require-cuda", action="store_true")
    train.add_argument("--amp", action="store_true")
    train.add_argument("--hybrid-retrieval", action="store_true")
    train.add_argument("--hybrid-epochs", type=int, default=5)
    train.add_argument("--hybrid-embedding-dim", type=int, default=64)
    train.add_argument("--hybrid-holdout-days", type=int)
    train.add_argument(
        "--sequence-mode", choices=("none", "historical"), default="none"
    )
    train.add_argument("--sequence-epochs", type=int, default=3)

    baseline = sub.add_parser("evaluate-baseline")
    baseline.add_argument("--domain", choices=("all", "clip", "movie", "music"), default="all")
    baseline.add_argument("--candidate-budget", type=int, default=1000)
    baseline.add_argument("--exact-cowatch", action="store_true")
    baseline.add_argument(
        "--baseline-family",
        choices=IMPLICIT_BASELINE_FAMILIES,
        default="graded_per_rule_popularity",
    )
    baseline.add_argument("--epochs", type=int, default=10)
    baseline.add_argument("--embedding-dim", type=int, default=32)
    baseline.add_argument("--fusion-bpr-weight", type=float, default=1.0)
    baseline.add_argument("--batch-size", type=int, default=8192)
    baseline.add_argument("--seed", type=int, default=42)
    baseline.add_argument("--device")
    baseline.add_argument("--hybrid-retrieval", action="store_true")
    baseline.add_argument("--hybrid-epochs", type=int, default=5)
    baseline.add_argument("--hybrid-embedding-dim", type=int, default=64)
    baseline.add_argument("--hybrid-holdout-days", type=int)
    baseline.add_argument(
        "--hybrid-item-identity-policy",
        choices=("target_only", "context_and_target"),
        default="target_only",
    )
    baseline.add_argument(
        "--hybrid-scoring-catalog-policy",
        choices=("all_input", "eligible_only"),
        default="all_input",
    )
    baseline.add_argument(
        "--external-candidate-mode", choices=("rerank", "replace"), default="rerank"
    )
    baseline.add_argument("--external-candidate-slots", type=int, default=100)
    baseline.add_argument("--external-rrf-weight", type=float, default=0.25)
    baseline.add_argument(
        "--external-replacement-requires-history",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    baseline.add_argument(
        "--external-replacement-requires-witness",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    baseline.add_argument("--content-external-rrf-weight", type=float, default=0.0)
    baseline.add_argument("--content-rrf-constant", type=int, default=60)
    baseline.add_argument("--recency-candidates-per-rule", type=int, default=8)
    baseline.add_argument("--recency-reserved-per-rule", type=int, default=8)
    baseline.add_argument("--recency-max-age-days", type=int, default=30)
    baseline.add_argument("--cowatch-candidates-per-rule", type=int, default=0)
    baseline.add_argument("--tag-candidates-per-rule", type=int, default=0)
    _add_scoring_arguments(baseline)
    baseline.add_argument(
        "--sequence-mode", choices=("none", "historical"), default="none"
    )
    baseline.add_argument("--sequence-epochs", type=int, default=3)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--domain", choices=("all", "clip", "movie", "music"), default="all")
    evaluate.add_argument("--model-family", choices=MODEL_FAMILIES, default="fcn_official")
    evaluate.add_argument("--run-id")
    evaluate.add_argument("--batch-size", type=int, default=4096)
    evaluate.add_argument("--amp", action="store_true")
    evaluate.add_argument(
        "--eval-serving-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    evaluate.add_argument("--candidate-budget", type=int, default=1000)
    evaluate.add_argument("--exact-cowatch", action="store_true")
    evaluate.add_argument("--hybrid-retrieval", action="store_true")
    evaluate.add_argument("--hybrid-epochs", type=int, default=5)
    evaluate.add_argument("--hybrid-embedding-dim", type=int, default=64)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument(
        "--sequence-mode", choices=("none", "historical"), default="none"
    )
    evaluate.add_argument("--sequence-epochs", type=int, default=3)
    evaluate.add_argument("--recency-candidates-per-rule", type=int, default=8)
    evaluate.add_argument("--recency-reserved-per-rule", type=int, default=8)
    evaluate.add_argument("--recency-max-age-days", type=int, default=30)
    _add_scoring_arguments(evaluate)

    live = sub.add_parser("live-test")
    live.add_argument("--domain", choices=("all", "clip", "movie", "music"), default="all")
    live.add_argument(
        "--num-profiles", type=int, default=1000, help="0 serves every profile"
    )
    live.add_argument("--warm-repetitions", type=int, default=3)
    live.add_argument("--output-dir")
    live.add_argument("--model-family", choices=("per_rule_popularity", *MODEL_FAMILIES), default="per_rule_popularity")
    live.add_argument("--num-contents", type=int)
    live.add_argument("--profile-chunk-size", type=int, default=128)
    live.add_argument("--batch-size", type=int, default=4096)
    live.add_argument("--amp", action="store_true")
    live.add_argument("--dry-run", action="store_true")
    live.add_argument("--resume", action="store_true")
    live.add_argument("--train-if-missing", action="store_true")
    live.add_argument("--no-gpu-sampling", action="store_true")
    live.add_argument(
        "--skip-offline-eval",
        action="store_true",
        help="skip scoring the served profiles against the temporal test truth",
    )
    live.add_argument("--run-id")
    live.add_argument("--candidate-budget", type=int, default=1000)
    live.add_argument("--exact-cowatch", action="store_true")
    add_ordering_arguments(live)
    live.add_argument("--contents-per-rule-max", type=int, default=30)
    live.add_argument("--page-content-budget", type=int, default=600)
    live.add_argument(
        "--page-dedup",
        choices=PAGE_DEDUP_POLICIES,
        default="balanced_regret",
    )
    live.add_argument("--recency-candidates-per-rule", type=int, default=8)
    live.add_argument("--recency-reserved-per-rule", type=int, default=8)
    live.add_argument("--recency-max-age-days", type=int, default=30)
    live.add_argument("--candidate-rules", type=int, default=30)
    live.add_argument("--reserved-per-rule", type=int, default=33)
    live.add_argument("--source-candidates-per-rule", type=int, default=33)
    _add_gbdt_arguments(live)

    args = apply_combination(parser.parse_args(argv))
    if args.command == "prepare":
        paths, metrics = prepare_canonical_project(
            args.root,
            args.domain,
            args.max_interaction_rows,
            args.max_behavior_rows,
            include_behavior=not args.skip_behavior,
            force=args.force,
            full_source_hash=args.full_source_hash,
            target_version=args.target_version,
            session_gap_minutes=args.session_gap_minutes,
            user_reference_date=args.user_reference_date,
        )
        print(json.dumps({"paths": {key: str(value) for key, value in paths.items()}, "metrics": metrics}, indent=2))
    elif args.command == "build-temporal":
        paths = build_temporal_project(
            args.root,
            args.domain,
            max_profiles=args.max_profiles,
            max_contents=args.max_contents,
            force=args.force,
            seed=args.seed,
            metadata_as_of=args.metadata_as_of,
            target_version=args.target_version,
            sequence_mode=args.sequence_mode,
            window_mode=args.window_mode,
            impression_weight=args.impression_weight,
        )
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    elif args.command == "train":
        bundle = train_temporal_model(
            args.root,
            args.domain,
            args.model_family,
            args.resume,
            args.epochs,
            args.batch_size,
            args.require_cuda,
            args.amp,
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
        print(json.dumps({"run_id": bundle.manifest.run_id, "training": bundle.training}, indent=2))
    elif args.command == "evaluate-baseline":
        bundle = evaluate_popularity_baseline(
            args.root,
            args.domain,
            candidate_budget=args.candidate_budget,
            exact_cowatch=args.exact_cowatch,
            baseline_family=args.baseline_family,
            epochs=args.epochs,
            embedding_dim=args.embedding_dim,
            batch_size=args.batch_size,
            fusion_bpr_weight=args.fusion_bpr_weight,
            seed=args.seed,
            device=args.device,
            hybrid_retrieval=args.hybrid_retrieval,
            hybrid_epochs=args.hybrid_epochs,
            hybrid_embedding_dim=args.hybrid_embedding_dim,
            hybrid_holdout_days=args.hybrid_holdout_days,
            hybrid_item_identity_policy=args.hybrid_item_identity_policy,
            hybrid_scoring_catalog_policy=args.hybrid_scoring_catalog_policy,
            sequence_mode=args.sequence_mode,
            sequence_epochs=args.sequence_epochs,
            evaluation_split=args.evaluation_split,
            maximum_evaluation_profiles=args.max_eval_profiles,
            evaluation_profile_seed=args.eval_profile_seed,
            external_candidate_mode=args.external_candidate_mode,
            external_candidate_slots=args.external_candidate_slots,
            external_rrf_weight=args.external_rrf_weight,
            external_replacement_requires_history=(
                args.external_replacement_requires_history
            ),
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
            gbdt_backend=args.gbdt_backend,
            gbdt_device=args.gbdt_device,
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
            content_rank_scope=args.content_rank_scope,
            content_fcn_family=args.content_fcn_family,
            routed_fallback=args.routed_fallback,
            fcn_model_dir=args.fcn_model_dir,
        )
        print(
            json.dumps(
                {
                    "run_id": bundle.manifest.run_id,
                    "evaluation": {
                        key: bundle.evaluation.get(key)
                        for key in (
                            "candidate_recall@1000",
                            "rule_ndcg@20",
                            "macro_content_ndcg@10",
                            "overall_recall@200",
                            "exact_20_normal_rule_rate",
                            "exact_10_content_rate",
                            "completion_gain_rule_ndcg@20",
                            "completion_gain_macro_content_ndcg@10",
                            "watch_time_gain_rule_ndcg@20",
                            "watch_time_gain_macro_content_ndcg@10",
                        )
                    },
                    "inference": bundle.inference,
                },
                indent=2,
            )
        )
    elif args.command == "evaluate":
        bundle = evaluate_temporal_model(
            args.root,
            args.domain,
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
            maximum_evaluation_profiles=args.max_eval_profiles,
            evaluation_profile_seed=args.eval_profile_seed,
            recency_candidates_per_rule=args.recency_candidates_per_rule,
            recency_reserved_per_rule=args.recency_reserved_per_rule,
            recency_max_age_days=args.recency_max_age_days,
            gbdt_backend=args.gbdt_backend,
            gbdt_device=args.gbdt_device,
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
            content_rank_scope=args.content_rank_scope,
            content_fcn_family=args.content_fcn_family,
            routed_fallback=args.routed_fallback,
            fcn_model_dir=args.fcn_model_dir,
        )
        print(
            json.dumps(
                {
                    "run_id": bundle.manifest.run_id,
                    "evaluation": {
                        key: bundle.evaluation.get(key)
                        for key in (
                            "candidate_recall@1000",
                            "rule_ndcg@20",
                            "macro_content_ndcg@10",
                            "overall_recall@200",
                            "exact_20_normal_rule_rate",
                            "exact_10_content_rate",
                            "completion_gain_rule_ndcg@20",
                            "completion_gain_macro_content_ndcg@10",
                            "watch_time_gain_rule_ndcg@20",
                            "watch_time_gain_macro_content_ndcg@10",
                        )
                    },
                    "inference": bundle.inference,
                },
                indent=2,
            )
        )
    else:
        bundle = popularity_live_test(
            args.root,
            args.domain,
            args.num_profiles,
            args.warm_repetitions,
            args.output_dir,
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
        print(json.dumps({"run_id": bundle.manifest.run_id, "inference": bundle.inference, "output": bundle.output}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
