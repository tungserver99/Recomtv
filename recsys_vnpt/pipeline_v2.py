
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from recsys_vnpt.canonical import (
    cache_is_current,
    historical_behavior_features_from_parts,
    read_partitioned_events,
    select_content_metadata_as_of,
)
from recsys_vnpt.engagement import (
    DEFAULT_TARGET_VERSION,
    GAIN_COLUMNS,
    LabelContract,
    historical_sequence_features,
    label_engagement_window,
)
from recsys_vnpt.evaluation import evaluate_scored_candidates
from recsys_vnpt.hybrid_retrieval import HybridRetrievalConfig
from recsys_vnpt.metrics_schema import MetricBundle
from recsys_vnpt.metrics import hierarchical_relative_harmonic_score
from recsys_vnpt.metrics_store import (
    MetricsStore,
    create_manifest,
)
from recsys_vnpt.model_runner import (
    TrainingConfig,
    predict,
    prepare_inference_input,
    train_model,
)
from recsys_vnpt.paths import (
    data_artifact_path,
    model_artifact_path,
)
from recsys_vnpt.impressions import window_impression_negatives
from recsys_vnpt.rule_assembly import (
    SPECIAL_RULES,
    aggregate_model_rule_scores,
)
from recsys_vnpt.sequence_retrieval import SASRecLiteConfig
from recsys_vnpt.retrieval import historical_content_popularity
from recsys_vnpt.source_manifest import stable_hash
from recsys_vnpt.temporal import (
    DEFAULT_TEMPORAL_WINDOWS,
    build_pair_features,
    derive_relative_temporal_windows,
    evaluation_candidates,
    exclude_consumed_pairs,
    label_future_window,
    load_temporal_windows,
    mixed_negative_samples,
    persist_temporal_artifact,
    select_smoke_profile_cohort,
    split_aware_catalog_cap,
    temporal_window_fingerprint,
    validate_window_positives,
    write_temporal_windows,
)


from recsys_vnpt.prepare import (  # re-exported: callers still import these here
    _interaction_catalog_audit,
    _read_canonical_interactions,
    build_rule_membership_from_ids,
    prepare_canonical_project,
)

from recsys_vnpt.runtime import (
    _partition_serving_interaction_scopes,
    _rss_bytes,
)



def build_temporal_project(
    project_root: str | Path = ".",
    domain: str = "all",
    max_profiles: int | None = None,
    max_contents: int | None = None,
    force: bool = False,
    seed: int = 42,
    metadata_as_of: bool = True,
    target_version: str = DEFAULT_TARGET_VERSION,
    sequence_mode: str = "historical",
    window_mode: str = "pinned",
    time_train: int = 24,
    time_valid: int = 24,
    time_test: int = 24,
    impression_weight: float = 1.0,
    progress_callback: Callable[..., None] | None = None,
) -> dict[str, Path]:
    temporal_build_started = time.perf_counter()
    if progress_callback:
        progress_callback("temporal_stage", stage="load_canonical", status="started")
    root = Path(project_root).resolve()
    canonical = data_artifact_path("canonical", domain, project_root=root)
    if target_version != DEFAULT_TARGET_VERSION:
        raise ValueError(
            "legacy binary targets are audit-only; build promotion data with "
            f"--target-version {DEFAULT_TARGET_VERSION}"
        )
    if sequence_mode not in {"none", "historical"}:
        raise ValueError("sequence_mode must be none or historical")
    items = pd.read_parquet(canonical / "contents.parquet")
    users = pd.read_parquet(canonical / "users.parquet")
    mapping = pd.read_parquet(canonical / "profile_mapping.parquet")
    membership = pd.read_parquet(canonical / "content_rule_membership.parquet")
    interactions = _read_canonical_interactions(
        canonical / "interactions.parquet",
        content_ids=set(items["content_id"].astype(str)),
    )
    playback_events_path = canonical / "playback_events.parquet"
    label_contract_path = canonical / "label_contract_v4.json"
    if not playback_events_path.exists() or not label_contract_path.exists():
        raise FileNotFoundError(
            "Missing engagement v4 artifacts. Re-run pipeline prepare before "
            "building temporal data."
        )
    playback_events = pd.read_parquet(playback_events_path)
    label_contract = LabelContract.load(label_contract_path)
    if label_contract.domain != domain:
        raise ValueError(
            f"label contract domain mismatch: {label_contract.domain} != {domain}"
        )
    versions_path = canonical / "content_versions.parquet"
    if metadata_as_of and not versions_path.exists():
        raise FileNotFoundError(
            f"Missing {versions_path}. Re-run pipeline prepare to build temporal metadata versions."
        )
    content_versions = (
        pd.read_parquet(versions_path) if versions_path.exists() else pd.DataFrame()
    )
    if progress_callback:
        progress_callback(
            "temporal_stage", stage="load_canonical", status="completed",
            interactions=len(interactions), profiles=interactions["profile_id"].nunique(),
        )

    if window_mode == "relative":
        max_event = pd.to_datetime(
            interactions["create_date"], errors="coerce"
        ).max()
        if pd.isna(max_event):
            raise ValueError(
                "relative temporal windows require interactions with create_date"
            )
        temporal_windows = derive_relative_temporal_windows(
            max_event, time_train=time_train, time_valid=time_valid, time_test=time_test
        )
    elif window_mode == "pinned":
        temporal_windows = DEFAULT_TEMPORAL_WINDOWS
    else:
        raise ValueError("window_mode must be pinned or relative")

    # Labels are created before any smoke cap. Entity caps therefore cannot erase
    # the evidence that decides which profile/content rows enter the cohort.
    if progress_callback:
        progress_callback("temporal_stage", stage="label_windows", status="started", windows=len(temporal_windows))
    labeled_windows = {}
    legacy_deduplicated_windows = {}
    playback_partitions = canonical / "playback_events_by_date"
    for window in temporal_windows:
        if progress_callback:
            progress_callback("temporal_window", window=window.name, stage="labeling", status="started")
        window_events = (
            read_partitioned_events(
                playback_partitions, window.label_start, window.label_end
            )
            if playback_partitions.is_dir()
            else playback_events[
                pd.to_datetime(playback_events["event_end"], errors="coerce").between(
                    window.label_start, window.label_end
                )
            ]
        )
        labeled_windows[window.name] = label_engagement_window(
            window_events,
            items,
            label_contract,
            window.label_start,
            window.label_end,
        )
        deduplicated_legacy_input = window_events.rename(
            columns={"watch_seconds": "duration", "event_end": "create_date"}
        )
        legacy_deduplicated_windows[window.name] = label_future_window(
            deduplicated_legacy_input, items, window, domain
        )
        if progress_callback:
            progress_callback(
                "temporal_window", window=window.name, stage="labeling", status="completed",
                rows=len(labeled_windows[window.name]),
            )
    if progress_callback:
        progress_callback("temporal_stage", stage="label_windows", status="completed")
    if window_mode == "relative":
        validate_window_positives(labeled_windows)
    legacy_audit_windows = {
        window.name: label_future_window(interactions, items, window, domain)
        for window in temporal_windows
    }
    label_metrics_path = canonical / "label_metrics.json"
    label_metrics = (
        json.loads(label_metrics_path.read_text(encoding="utf-8"))
        if label_metrics_path.exists()
        else {}
    )
    label_sensitivity: dict[str, Any] = {}
    for window in temporal_windows:
        engagement = labeled_windows[window.name]
        legacy = legacy_audit_windows[window.name]
        engagement_keys = set(
            engagement[["profile_id", "content_id"]]
            .drop_duplicates()
            .astype(str)
            .itertuples(index=False, name=None)
        )
        legacy_positive_keys = set(
            legacy[
                pd.to_numeric(legacy["label"], errors="coerce")
                .fillna(0)
                .eq(1)
            ][["profile_id", "content_id"]]
            .drop_duplicates()
            .astype(str)
            .itertuples(index=False, name=None)
        )
        legacy_deduplicated = legacy_deduplicated_windows[window.name]
        legacy_deduplicated_positive_keys = set(
            legacy_deduplicated[
                pd.to_numeric(legacy_deduplicated["label"], errors="coerce")
                .fillna(0)
                .eq(1)
            ][["profile_id", "content_id"]]
            .drop_duplicates()
            .astype(str)
            .itertuples(index=False, name=None)
        )
        window_metrics: dict[str, Any] = {
            "engagement_observed_pairs": int(len(engagement_keys)),
            "legacy_observed_pairs": int(len(legacy)),
            "legacy_positive_pairs": int(len(legacy_positive_keys)),
            "legacy_positive_rate_on_observed": (
                len(legacy_positive_keys) / max(len(legacy), 1)
            ),
            "legacy_positive_overlap_with_engagement": int(
                len(legacy_positive_keys & engagement_keys)
            ),
            "legacy_duplicate_induced_positive_pairs": int(
                len(legacy_positive_keys - legacy_deduplicated_positive_keys)
            ),
            "engagement_pairs_without_legacy_positive": int(
                len(engagement_keys - legacy_positive_keys)
            ),
            "repeat_count_positive_rule_enabled": False,
            "duplicate_induced_positive_pairs": 0,
        }
        for gain_column in GAIN_COLUMNS:
            values = pd.to_numeric(
                engagement[gain_column], errors="coerce"
            ).fillna(0.0)
            window_metrics[gain_column] = {
                "sum": float(values.sum()),
                "mean": float(values.mean()) if len(values) else 0.0,
                "p50": float(values.quantile(0.5)) if len(values) else 0.0,
                "p95": float(values.quantile(0.95)) if len(values) else 0.0,
            }
        label_sensitivity[window.name] = window_metrics
    label_metrics.update(
        {
            "target_version": target_version,
            "label_contract_hash": label_contract.artifact_hash,
            "legacy_binary_promotion_allowed": False,
            "unobserved_samples_are_true_negatives": False,
            "label_sensitivity": label_sensitivity,
        }
    )
    label_metrics_path.write_text(
        json.dumps(label_metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    selected_profiles = select_smoke_profile_cohort(
        labeled_windows, max_profiles, seed=seed
    )
    if selected_profiles is not None:
        interactions = interactions[
            interactions["profile_id"].astype(str).isin(selected_profiles)
        ]
        playback_events = playback_events[
            playback_events["profile_id"].astype(str).isin(selected_profiles)
        ]
        mapping = mapping[mapping["profile_id"].astype(str).isin(selected_profiles)]
        labeled_windows = {
            name: frame[frame["profile_id"].astype(str).isin(selected_profiles)]
            for name, frame in labeled_windows.items()
        }
    full_popularity = historical_content_popularity(interactions)
    positive_frames = [
        frame[pd.to_numeric(frame["label"], errors="coerce").fillna(0).eq(1)]
        for frame in labeled_windows.values()
    ]
    items, membership = split_aware_catalog_cap(
        items,
        membership,
        positive_frames,
        full_popularity,
        max_contents,
    )
    selected_contents = set(items["content_id"].astype(str))
    interactions = interactions[
        interactions["content_id"].astype(str).isin(selected_contents)
    ]
    labeled_windows = {
        name: frame[frame["content_id"].astype(str).isin(selected_contents)]
        for name, frame in labeled_windows.items()
    }
    temporal = canonical / "temporal"
    temporal.mkdir(parents=True, exist_ok=True)
    time_config = {
        "mode": window_mode,
        "time_train": time_train if window_mode == "relative" else None,
        "time_valid": time_valid if window_mode == "relative" else None,
        "time_test": time_test if window_mode == "relative" else None,
    }
    write_temporal_windows(temporal, temporal_windows, time_config=time_config)
    expected_names = [window.name for window in temporal_windows] + [
        "validation_candidates",
        "test_candidates",
    ]
    canonical_metrics_path = canonical / "data_metrics.json"
    canonical_fingerprint_value = (
        json.loads(canonical_metrics_path.read_text(encoding="utf-8")).get(
            "canonical_fingerprint", "unknown"
        )
        if canonical_metrics_path.exists()
        else "unknown"
    )
    temporal_fingerprint = stable_hash(
        {
            "canonical": canonical_fingerprint_value,
            "max_profiles": max_profiles,
            "max_contents": max_contents,
            "window_fingerprint": temporal_window_fingerprint(
                temporal_windows, window_mode, time_train, time_valid, time_test
            ),
            "seed": seed,
            "train_negative_policy": "2_same_rule_1_popularity_1_uniform",
            "positive_policy": "novel_not_consumed",
            "evaluation_negatives": 1000,
            "metadata_as_of": metadata_as_of,
            "target_version": target_version,
            "label_contract_hash": label_contract.artifact_hash,
            "sequence_mode": sequence_mode,
        }
    )
    temporal_cache = temporal / "cache_metadata.json"
    expected_paths = {
        name: temporal / f"{name}.parquet" for name in expected_names
    }
    if (
        not force
        and cache_is_current(temporal_cache, temporal_fingerprint)
        and all(path.exists() for path in expected_paths.values())
    ):
        if progress_callback:
            progress_callback("temporal_cache", status="reused", fingerprint=temporal_fingerprint)
        return expected_paths
    behavior_paths = sorted((canonical / "behavior").glob("part_*.parquet"))
    outputs: dict[str, Path] = {}
    window_timings: dict[str, dict[str, float]] = {}
    positive_policy_counts: dict[str, dict[str, int]] = {}
    for window in temporal_windows:
        if progress_callback:
            progress_callback("temporal_window", window=window.name, stage="features_sampling_candidates", status="started")
        raw_positives = labeled_windows[window.name][
            labeled_windows[window.name]["label"].eq(1)
        ].drop_duplicates(["profile_id", "content_id"])
        history_pairs = interactions[
            pd.to_datetime(interactions["create_date"], errors="coerce")
            <= window.history_end
        ][["profile_id", "content_id"]].drop_duplicates()
        novel_positives = exclude_consumed_pairs(raw_positives, history_pairs)
        positive_policy_counts[window.name] = {
            "raw_positive_pairs": int(len(raw_positives)),
            "novel_positive_pairs": int(len(novel_positives)),
            "repeat_positive_pairs_excluded": int(
                len(raw_positives) - len(novel_positives)
            ),
        }
    build_config = {
        "max_profiles": max_profiles,
        "max_contents": max_contents,
        "profiles": int(interactions["profile_id"].nunique()),
        "contents": int(items["content_id"].nunique()),
        "interaction_rows": int(len(interactions)),
        "seed": seed,
        "metadata_as_of": metadata_as_of,
        "target_version": target_version,
        "target_semantics": label_contract.to_dict()["target_semantics"],
        "label_contract_hash": label_contract.artifact_hash,
        "sequence_mode": sequence_mode,
        "positive_pairs": {
            name: int(
                pd.to_numeric(frame["label"], errors="coerce").fillna(0).eq(1).sum()
            )
            for name, frame in labeled_windows.items()
        },
        "positive_policy_counts": positive_policy_counts,
    }
    (temporal / "build_config.json").write_text(
        json.dumps(build_config, indent=2, sort_keys=True), encoding="utf-8"
    )
    # Shown-not-watched exposure, when the prepare step produced it.
    impressions_path = canonical / "impressions.parquet"
    impressions = (
        pd.read_parquet(impressions_path) if impressions_path.exists() else None
    )
    impression_window_counts: dict[str, int] = {}
    for window in temporal_windows:
        if progress_callback:
            progress_callback("temporal_window", window=window.name, stage="features_sampling_candidates", status="started")
        window_started = time.perf_counter()
        sampling_seconds = 0.0
        feature_seconds = 0.0
        candidate_seconds = 0.0
        impression_negative_rows = 0
        behavior_features = (
            historical_behavior_features_from_parts(
                behavior_paths,
                window.history_end,
                set(interactions["profile_id"].astype(str)),
            )
            if behavior_paths
            else None
        )
        labeled = labeled_windows[window.name]
        target_columns = [
            "profile_id",
            "content_id",
            "label",
            *GAIN_COLUMNS,
            "sample_type",
            "sample_weight",
            "target_semantics",
            "episode_count",
            "total_watch_seconds",
            "max_episode_watch_seconds",
        ]
        raw_positives = labeled[labeled["label"].eq(1)][
            [column for column in target_columns if column in labeled]
        ].drop_duplicates(["profile_id", "content_id"])
        history = interactions[pd.to_datetime(interactions["create_date"], errors="coerce") <= window.history_end]
        consumed = history[["profile_id", "content_id"]].drop_duplicates()
        positives = exclude_consumed_pairs(raw_positives, consumed)
        popularity = history.groupby("content_id").size().rename("popularity")
        catalog = items[["content_id"]].drop_duplicates().merge(popularity, on="content_id", how="left").fillna({"popularity": 0})
        sampling_started = time.perf_counter()
        window_impressions = (
            window_impression_negatives(
                impressions,
                labeled[["profile_id", "content_id"]],
                window.label_start,
                window.label_end,
                consumed=consumed,
            )
            if impressions is not None and not impressions.empty
            else {}
        )
        negatives = mixed_negative_samples(
            positives,
            catalog,
            consumed,
            membership=membership,
            ratio=4,
            seed=seed,
            impression_negatives=window_impressions or None,
            impression_weight=impression_weight,
        )
        for gain_column in GAIN_COLUMNS:
            negatives[gain_column] = 0.0
        negatives["target_semantics"] = target_version
        # shown_not_watched is observed exposure, so it keeps its own name;
        # only sampled contrasts carry the unobserved_ prefix.
        sampled_mask = negatives["sample_type"].ne("shown_not_watched")
        negatives.loc[sampled_mask, "sample_type"] = (
            "unobserved_" + negatives.loc[sampled_mask, "sample_type"].astype(str)
        )
        impression_negative_rows = int((~sampled_mask).sum())
        sampling_seconds = time.perf_counter() - sampling_started
        positive_rows = positives.assign(
            label=1, sample_type="observed_engagement", sample_weight=1.0
        )
        pairs = pd.concat(
            [positive_rows, negatives], ignore_index=True
        ).drop_duplicates(["profile_id", "content_id"], keep="first")
        pairs = pairs.merge(mapping[["profile_id", "username"]], on="profile_id", how="left")
        window_items = (
            select_content_metadata_as_of(content_versions, window.history_end)
            if metadata_as_of
            else items
        )
        window_items = window_items[
            window_items["content_id"].astype(str).isin(selected_contents)
        ]
        feature_started = time.perf_counter()
        frame = build_pair_features(
            pairs, interactions, window_items, users, window.history_end
        )
        sequence_features = historical_sequence_features(
            playback_events,
            window.history_end,
        )
        if sequence_mode == "none":
            sequence_features = sequence_features.drop(
                columns=["historical_content_sequence"], errors="ignore"
            )
        frame = frame.merge(sequence_features, on="profile_id", how="left")
        for column in (
            "historical_session_count",
            "historical_playback_count",
            "historical_sequence_depth",
        ):
            if column in frame:
                frame[column] = pd.to_numeric(
                    frame[column], errors="coerce"
                ).fillna(0.0)
        if behavior_features is not None:
            frame = frame.merge(behavior_features, on="profile_id", how="left")
            for column in behavior_features.columns:
                if column != "profile_id":
                    frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        path = persist_temporal_artifact(frame, temporal / f"{window.name}.parquet")
        feature_seconds = time.perf_counter() - feature_started
        outputs[window.name] = path
        if window.split in {"validation", "test"}:
            candidate_started = time.perf_counter()
            candidates = evaluation_candidates(
                raw_positives,
                catalog,
                consumed=consumed,
                negatives_per_profile=1000,
                seed=seed,
            )
            candidates = candidates.merge(mapping[["profile_id", "username"]], on="profile_id", how="left")
            outputs[f"{window.name}_candidates"] = persist_temporal_artifact(
                candidates,
                temporal / f"{window.name}_candidates.parquet",
            )
            candidate_seconds = time.perf_counter() - candidate_started
        impression_window_counts[window.name] = impression_negative_rows
        window_timings[window.name] = {
            "sampling_seconds": sampling_seconds,
            "feature_and_training_persist_seconds": feature_seconds,
            "candidate_generation_and_persist_seconds": candidate_seconds,
            "total_seconds": time.perf_counter() - window_started,
        }
        if progress_callback:
            progress_callback(
                "temporal_window", window=window.name, stage="features_sampling_candidates", status="completed",
                rows=len(frame), candidates=len(candidates) if window.split in {"validation", "test"} else 0,
            )
    temporal_build_seconds = time.perf_counter() - temporal_build_started
    build_config["window_timings"] = window_timings
    build_config["impression_negative_rows"] = impression_window_counts
    build_config["impression_source"] = (
        str(impressions_path) if impressions is not None else None
    )
    build_config["temporal_build_time_seconds"] = temporal_build_seconds
    (temporal / "build_config.json").write_text(
        json.dumps(build_config, indent=2, sort_keys=True), encoding="utf-8"
    )
    metadata_metrics = {
        "schema_version": 4,
        "seed": seed,
        "metadata_as_of": metadata_as_of,
        "target_version": target_version,
        "target_semantics": label_contract.to_dict()["target_semantics"],
        "label_contract_hash": label_contract.artifact_hash,
        "classification_metrics_valid": False,
        "profiles": int(interactions["profile_id"].nunique()),
        "contents": int(items["content_id"].nunique()),
        "interaction_rows": int(len(interactions)),
        "training_rows": int(
            sum(
                len(pd.read_parquet(outputs[f"train_{index}"]))
                for index in (1, 2, 3)
            )
        ),
        "test_profiles": int(
            pd.read_parquet(outputs["test_candidates"])["profile_id"].nunique()
        ),
        "test_positives": int(
            pd.to_numeric(
                pd.read_parquet(outputs["test_candidates"])["label"],
                errors="coerce",
            ).fillna(0).eq(1).sum()
        ),
        "test_novel_positives": positive_policy_counts["test"][
            "novel_positive_pairs"
        ],
        "test_repeat_positive_pairs_excluded": positive_policy_counts["test"][
            "repeat_positive_pairs_excluded"
        ],
        "test_completion_gain_sum": float(
            pd.to_numeric(
                pd.read_parquet(outputs["test_candidates"])["completion_gain"],
                errors="coerce",
            ).fillna(0.0).sum()
        ),
        "test_watch_time_gain_sum": float(
            pd.to_numeric(
                pd.read_parquet(outputs["test_candidates"])["watch_time_gain"],
                errors="coerce",
            ).fillna(0.0).sum()
        ),
        "temporal_build_time_seconds": temporal_build_seconds,
        "window_timings": window_timings,
    }
    (temporal / "data_metrics.json").write_text(
        json.dumps(metadata_metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporal_cache.write_text(
        json.dumps({"fingerprint": temporal_fingerprint}, indent=2),
        encoding="utf-8",
    )
    return outputs


def train_temporal_model(
    project_root: str | Path,
    domain: str,
    model_family: str,
    resume: bool = False,
    epochs: int = 100,
    batch_size: int = 4096,
    require_cuda: bool = False,
    amp: bool = False,
    seed: int = 42,
    checkpoint_selection: str = "hierarchical",
    rule_loss_weight: float = 0.4,
    content_loss_weight: float = 0.4,
    auxiliary_loss_weight: float = 0.2,
    hybrid_retrieval: bool = False,
    hybrid_epochs: int = 5,
    hybrid_embedding_dim: int = 64,
    hybrid_holdout_days: int | None = None,
    sequence_mode: str = "none",
    sequence_epochs: int = 3,
) -> MetricBundle:
    root = Path(project_root).resolve()
    temporal = data_artifact_path("canonical", domain, "temporal", project_root=root)
    train_frames = []
    for index in (1, 2, 3):
        frame = pd.read_parquet(temporal / f"train_{index}.parquet")
        frame["temporal_window"] = f"train_{index}"
        train_frames.append(frame)
    train = pd.concat(train_frames, ignore_index=True)
    validation = pd.read_parquet(temporal / "validation.parquet")
    validation["temporal_window"] = "validation"
    membership = pd.read_parquet(
        data_artifact_path(
            "canonical",
            domain,
            "content_rule_membership.parquet",
            project_root=root,
        )
    )
    if model_family == "hierarchical_fcn":
        normalized_membership = membership[
            ["content_id", "rule_id"]
        ].drop_duplicates().astype(str)
        normalized_membership = normalized_membership[
            ~normalized_membership["rule_id"].isin(SPECIAL_RULES)
        ]
        train = train.drop(columns=["rule_id"], errors="ignore").merge(
            normalized_membership, on="content_id", how="inner"
        )
        validation = validation.drop(columns=["rule_id"], errors="ignore").merge(
            normalized_membership, on="content_id", how="inner"
        )
    config = TrainingConfig(
        model_family=model_family,
        epochs=epochs,
        batch_size=batch_size,
        amp=amp,
        seed=seed,
        checkpoint_selection=checkpoint_selection,
        rule_loss_weight=rule_loss_weight,
        content_loss_weight=content_loss_weight,
        auxiliary_loss_weight=auxiliary_loss_weight,
    )
    model_output = model_artifact_path(domain, "v2", model_family, project_root=root)
    selector = None
    if checkpoint_selection == "hierarchical":
        validation_truth = pd.read_parquet(
            temporal / "validation_candidates.parquet"
        )
        (
            validation_serving,
            retrieval_rule_scores,
            _,
            validation_evaluation_truth,
            validation_popularity,
        ) = (
            _serving_candidates_for_evaluation(
                root,
                domain,
                validation_truth,
                candidate_budget=1000,
                cutoff=str(
                    load_temporal_windows(
                        data_artifact_path(
                            "canonical", domain, "temporal", project_root=root
                        )
                    )[3].history_end
                ),
                hybrid_retrieval=hybrid_retrieval,
                hybrid_config=HybridRetrievalConfig(
                    epochs=hybrid_epochs,
                    embedding_dim=hybrid_embedding_dim,
                    temporal_holdout_days=(
                        hybrid_holdout_days
                        if hybrid_holdout_days is not None
                        else 11
                    ),
                    seed=seed,
                ),
                sequence_mode=sequence_mode,
                sequence_config=SASRecLiteConfig(
                    epochs=sequence_epochs,
                    seed=seed,
                ),
            )
        )
        selector_prepared: dict[str, Any] = {}

        def selector(model, contract):
            prepared = selector_prepared.get(contract.artifact_hash)
            if prepared is None:
                prepared = prepare_inference_input(
                    contract, validation_serving, model_family
                )
                selector_prepared[contract.artifact_hash] = prepared
            validation_scores = predict(
                model,
                contract,
                validation_serving,
                model_family,
                batch_size=batch_size,
                amp=amp,
                prepared=prepared,
            )
            scored_validation = validation_serving.copy()
            scored_validation["score"] = validation_scores
            active_rule_scores = retrieval_rule_scores
            if model_family == "hierarchical_fcn":
                scored_validation["model_rule_score"] = predict(
                    model,
                    contract,
                    validation_serving,
                    model_family,
                    batch_size=batch_size,
                    amp=amp,
                    output_key="rule_score",
                    prepared=prepared,
                )
                model_rule_scores = aggregate_model_rule_scores(
                    scored_validation,
                    membership,
                )
                active_rule_scores = pd.concat(
                    [retrieval_rule_scores, model_rule_scores], ignore_index=True
                ).drop_duplicates(["profile_id", "rule_id"], keep="last")
            quality, _ = evaluate_scored_candidates(
                scored_validation,
                membership,
                validation_popularity,
                ground_truth_frame=validation_evaluation_truth,
                profile_rule_scores=active_rule_scores,
            )
            graded_selection_keys = [
                f"{gain_column}_{metric}"
                for gain_column in GAIN_COLUMNS
                for metric in ("rule_ndcg@20", "macro_content_ndcg@10")
            ]
            if all(key in quality for key in graded_selection_keys):
                graded_values = [
                    max(float(quality[key]), 0.0)
                    for key in graded_selection_keys
                ]
                score = (
                    len(graded_values)
                    / sum(1.0 / value for value in graded_values)
                    if min(graded_values) > 0
                    else 0.0
                )
            else:
                score = hierarchical_relative_harmonic_score(
                    float(quality["rule_ndcg@20"]),
                    float(quality["macro_content_ndcg@10"]),
                    1.0,
                    1.0,
                )
            selection_metrics = {
                "rule_ndcg@20": quality["rule_ndcg@20"],
                "macro_content_ndcg@10": quality[
                    "macro_content_ndcg@10"
                ],
                "overall_recall@200": quality["overall_recall@200"],
                "hierarchical_harmonic_score": score,
            }
            selection_metrics.update(
                {
                    key: quality[key]
                    for key in graded_selection_keys
                    if key in quality
                }
            )
            return score, selection_metrics

    _, contract, training_metrics = train_model(
        train,
        validation,
        domain,
        model_output,
        config,
        resume=resume,
        require_cuda=require_cuda,
        validation_selector=selector,
    )
    (model_output / "training_metrics.json").write_text(
        json.dumps(training_metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = create_manifest(
        root,
        domain,
        (
            "sprint-2"
            if model_family in {"dense_repaired_baseline", "embedding_mlp_baseline"}
            else "sprint-3"
        ),
        model_family,
        seed=config.seed,
        split={window.name: {"history_end": str(window.history_end), "label_start": str(window.label_start), "label_end": str(window.label_end)} for window in load_temporal_windows(data_artifact_path("canonical", domain, "temporal", project_root=root))},
        hashes={
            "feature_contract": contract.artifact_hash,
            "training_data": training_metrics["training_data_hash"],
            "validation_data": training_metrics["validation_data_hash"],
        },
        config={
            **asdict(config),
            "hybrid_retrieval": hybrid_retrieval,
            "hybrid_epochs": hybrid_epochs,
            "hybrid_embedding_dim": hybrid_embedding_dim,
            "hybrid_holdout_days": (
                hybrid_holdout_days if hybrid_holdout_days is not None else 11
            ),
            "sequence_mode": sequence_mode,
            "sequence_epochs": sequence_epochs,
        },
    )
    bundle = MetricBundle(manifest=manifest, training=training_metrics)
    MetricsStore(root).write_bundle(bundle)
    return bundle


from recsys_vnpt.evaluation_runs import (  # re-exported for existing callers
    _evaluation_seen_entities,
    _evaluation_truth_selection,
    _serving_candidates_for_evaluation,
    evaluate_popularity_baseline,
    evaluate_temporal_model,
    scoring_manifest_config,
)



from recsys_vnpt.serving import (  # re-exported for existing callers
    popularity_live_test,
    serving_manifest_config,
)

