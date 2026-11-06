from __future__ import annotations

from typing import Any

import pandas as pd

from recsys_vnpt.metrics import (
    bootstrap_confidence_interval,
    graded_hierarchical_ranking_metrics,
    hierarchical_ranking_metrics,
    ranking_metrics,
)
from recsys_vnpt.model_runner import classification_metrics
from recsys_vnpt.rule_assembly import (
    RuleAssemblyConfig,
    SPECIAL_RULES,
    bayesian_rule_popularity,
    explode_content_rule_membership,
    rank_scored_rules_and_all_with_diagnostics,
)


def ground_truth_by_profile(frame: pd.DataFrame) -> dict[str, set[str]]:
    if not {"profile_id", "content_id", "label"}.issubset(frame):
        raise ValueError("evaluation frame requires profile_id, content_id, and label")
    positives = frame[pd.to_numeric(frame["label"], errors="coerce").fillna(0).eq(1)]
    return {
        str(profile_id): set(group["content_id"].astype(str))
        for profile_id, group in positives.groupby("profile_id")
    }


def content_rule_map(membership: pd.DataFrame) -> dict[str, set[str]]:
    if not {"content_id", "rule_id"}.issubset(membership):
        raise ValueError("membership requires content_id and rule_id")
    return {
        str(content_id): set(group["rule_id"].astype(str))
        for content_id, group in membership.groupby("content_id")
    }


def ground_truth_gains_by_profile(
    frame: pd.DataFrame,
    gain_column: str,
) -> dict[str, dict[str, float]]:
    required = {"profile_id", "content_id", gain_column}
    if not required.issubset(frame):
        return {}
    work = frame[["profile_id", "content_id", gain_column]].copy()
    work[gain_column] = pd.to_numeric(
        work[gain_column], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    work = work[work[gain_column].gt(0)]
    aggregated = work.groupby(
        [work["profile_id"].astype(str), work["content_id"].astype(str)],
        sort=True,
    )[gain_column].max()
    output: dict[str, dict[str, float]] = {}
    for (profile_id, content_id), gain in aggregated.items():
        output.setdefault(str(profile_id), {})[str(content_id)] = float(gain)
    return output


def _page_shape_metrics(
    ranked_rules: dict[str, list[dict[str, Any]]],
) -> dict[str, float]:
    """Slot count, distinct contents and repeat rate of the served page."""

    slots: list[int] = []
    unique: list[int] = []
    rule_sizes: list[int] = []
    for rows in ranked_rules.values():
        contents = [content for row in rows for content in row.get("content_ids", [])]
        if not contents:
            continue
        slots.append(len(contents))
        unique.append(len(set(contents)))
        rule_sizes.extend(len(row.get("content_ids", [])) for row in rows)
    if not slots:
        return {}
    total_slots = float(sum(slots))
    return {
        "page_slots": sum(slots) / len(slots),
        "page_unique_contents": sum(unique) / len(unique),
        "page_duplicate_slot_rate": (total_slots - sum(unique)) / total_slots,
        "page_min_contents_per_rule": float(min(rule_sizes)),
        "page_max_contents_per_rule": float(max(rule_sizes)),
        "page_mean_contents_per_rule": sum(rule_sizes) / len(rule_sizes),
    }


COHORT_COLUMNS = ("profile_oov", "content_oov", "activity_bucket", "content_family")

GAIN_COLUMNS = ("completion_gain", "watch_time_gain")


def evaluation_input_columns(
    scored: pd.DataFrame, ground_truth_frame: pd.DataFrame | None
) -> list[str]:
    """The columns the metrics actually read, in a stable order.

    The serving path passes the full candidate frame here. Copying all of it
    materialised 2.14 GiB of object dtype for a run that reads five columns plus
    the cohort keys, and ran a 16 GB machine out of memory on a 100-profile
    smoke test. Gains are only needed when this frame is also the truth.
    """

    wanted = ["profile_id", "content_id", "type_id", "label", "score"]
    wanted += [column for column in COHORT_COLUMNS if column in scored.columns]
    if ground_truth_frame is None:
        # This frame is then also the truth, so it has to keep what the truth
        # is read for: the graded gains and the label semantics.
        wanted += [column for column in GAIN_COLUMNS if column in scored.columns]
        wanted.append("target_semantics")
    return [
        column
        for column in dict.fromkeys(wanted)
        if column in scored.columns
    ]


def evaluate_scored_candidates(
    scored: pd.DataFrame,
    membership: pd.DataFrame,
    historical_popularity: pd.Series | dict[str, float] | None = None,
    ground_truth_frame: pd.DataFrame | None = None,
    profile_rule_scores: pd.DataFrame | None = None,
    score_semantics: str = "rank_score",
    assembly_config: "RuleAssemblyConfig | None" = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    required = {"profile_id", "content_id", "type_id", "label", "score"}
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"scored evaluation candidates are missing columns: {missing}")
    if score_semantics not in {"rank_score", "probability"}:
        raise ValueError("score_semantics must be rank_score or probability")
    normalized = scored.loc[
        :, evaluation_input_columns(scored, ground_truth_frame)
    ].copy()
    normalized["profile_id"] = normalized["profile_id"].astype("string")
    normalized["content_id"] = normalized["content_id"].astype("string")
    normalized["score"] = (
        pd.to_numeric(normalized["score"], errors="coerce")
        .astype("float32")
        .fillna(0.0)
    )
    all_rule_k = max(
        int(
            membership.loc[
                ~membership["rule_id"].astype(str).isin(SPECIAL_RULES),
                "rule_id",
            ].nunique()
        ),
        20,
    )
    truth = ground_truth_by_profile(
        normalized if ground_truth_frame is None else ground_truth_frame
    )
    overall_top = (
        normalized.sort_values(
            ["profile_id", "score", "content_id"],
            ascending=[True, False, True],
            kind="stable",
        )
        .drop_duplicates(["profile_id", "content_id"], keep="first")
        .groupby("profile_id", sort=False, observed=True)
        .head(200)
    )
    overall_recommendations = {
        str(profile_id): list(map(str, content_ids))
        for profile_id, content_ids in overall_top.groupby(
            "profile_id", sort=False, observed=True
        )["content_id"]
        .agg(list)
        .items()
    }
    expanded = explode_content_rule_membership(
        normalized[["profile_id", "content_id", "type_id", "score"]],
        membership,
    )
    if profile_rule_scores is not None and not profile_rule_scores.empty:
        required_rule_scores = {"profile_id", "rule_id", "rule_score"}
        if not required_rule_scores.issubset(profile_rule_scores):
            raise ValueError(
                "profile rule scores require profile_id, rule_id, and rule_score"
            )
        normalized_rule_scores = profile_rule_scores[
            ["profile_id", "rule_id", "rule_score"]
        ].copy()
        normalized_rule_scores["profile_id"] = normalized_rule_scores[
            "profile_id"
        ].astype("string")
        normalized_rule_scores["rule_id"] = normalized_rule_scores["rule_id"].astype(
            "string"
        )
        expanded = expanded.merge(
            normalized_rule_scores.drop_duplicates(
                ["profile_id", "rule_id"], keep="last"
            ),
            on=["profile_id", "rule_id"],
            how="left",
        )
        expanded["rule_score"] = pd.to_numeric(
            expanded["rule_score"], errors="coerce"
        )
    rule_popularity = bayesian_rule_popularity(
        historical_popularity or {}, membership
    ) if not isinstance(historical_popularity, pd.Series) else bayesian_rule_popularity(
        historical_popularity, membership
    )
    (
        ranked_rules,
        all_eligible_ranked_rules,
        rule_ranking_diagnostics,
    ) = rank_scored_rules_and_all_with_diagnostics(
        expanded,
        rule_popularity,
        rule_k=20,
        content_k=(
            assembly_config.max_contents if assembly_config is not None else 10
        ),
        config=assembly_config,
    )
    content_rules = content_rule_map(membership)
    hierarchical = hierarchical_ranking_metrics(
        ranked_rules,
        truth,
        content_rules,
        rule_k=20,
        content_k=10,
    )
    all_rule_hierarchical = hierarchical_ranking_metrics(
        all_eligible_ranked_rules,
        truth,
        content_rules,
        rule_k=all_rule_k,
        content_k=10,
    )
    hierarchical["all_rule_macro_content_ndcg@10"] = float(
        all_rule_hierarchical["macro_content_ndcg@10"]
    )
    hierarchical["all_rule_macro_content_recall@10"] = float(
        all_rule_hierarchical["macro_content_recall@10"]
    )
    truth_frame = normalized if ground_truth_frame is None else ground_truth_frame
    graded_metrics: dict[str, Any] = {}
    gain_truth: dict[str, dict[str, dict[str, float]]] = {}
    for gain_column in GAIN_COLUMNS:
        gains = ground_truth_gains_by_profile(truth_frame, gain_column)
        if not gains:
            continue
        gain_truth[gain_column] = gains
        graded_metrics.update(
            graded_hierarchical_ranking_metrics(
                ranked_rules,
                gains,
                content_rules,
                gain_name=gain_column,
                rule_k=20,
                content_k=10,
            )
        )
        all_rule_values = graded_hierarchical_ranking_metrics(
            all_eligible_ranked_rules,
            gains,
            content_rules,
            gain_name=gain_column,
            rule_k=all_rule_k,
            content_k=10,
        )
        graded_metrics[
            f"{gain_column}_all_rule_macro_content_ndcg@10"
        ] = float(all_rule_values[f"{gain_column}_macro_content_ndcg@10"])
        graded_metrics[
            f"{gain_column}_all_rule_macro_content_gain_recall@10"
        ] = float(
            all_rule_values[
                f"{gain_column}_macro_content_gain_recall@10"
            ]
        )
    overall = ranking_metrics(overall_recommendations, truth, k_values=(200,))
    classification: dict[str, float] | None = None
    if score_semantics == "probability":
        labels = pd.to_numeric(
            normalized["label"], errors="coerce"
        ).fillna(0).to_numpy()
        scores = normalized["score"].to_numpy()
        classification = (
            classification_metrics(labels, scores)
            if len(normalized)
            else {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "log_loss": 0.0,
                "auc": 0.0,
            }
        )
    truth_profiles = sorted(profile for profile, values in truth.items() if values)
    rule_counts = {
        profile: len(ranked_rules.get(profile, [])) for profile in truth_profiles
    }
    ranked_content_counts = [
        len(row.get("content_ids", []))
        for profile in truth_profiles
        for row in ranked_rules.get(profile, [])
    ]
    candidate_counts = normalized.groupby("profile_id")["content_id"].nunique()
    truth_pairs = truth_frame[
        pd.to_numeric(truth_frame["label"], errors="coerce").fillna(0).eq(1)
    ][["profile_id", "content_id"]].drop_duplicates()
    candidate_hits = len(
        truth_pairs.astype({"profile_id": "string", "content_id": "string"}).merge(
            normalized[["profile_id", "content_id"]].drop_duplicates(),
            on=["profile_id", "content_id"],
            how="inner",
        )
    )
    actual_candidate_recall = candidate_hits / max(len(truth_pairs), 1)

    profile_metrics: dict[str, dict[str, float]] = {}
    for profile in truth_profiles:
        values = hierarchical_ranking_metrics(
            {profile: ranked_rules.get(profile, [])},
            {profile: truth[profile]},
            content_rules,
            rule_k=20,
            content_k=10,
        )
        profile_metrics[profile] = {
            "rule_ndcg@20": float(values["rule_ndcg@20"]),
            "macro_content_ndcg@10": float(values["macro_content_ndcg@10"]),
            "overall_recall@200": float(values["overall_recall@200"]),
        }
        all_rule_values = hierarchical_ranking_metrics(
            {profile: all_eligible_ranked_rules.get(profile, [])},
            {profile: truth[profile]},
            content_rules,
            rule_k=all_rule_k,
            content_k=10,
        )
        profile_metrics[profile].update(
            {
                "all_rule_macro_content_ndcg@10": float(
                    all_rule_values["macro_content_ndcg@10"]
                ),
                "all_rule_macro_content_recall@10": float(
                    all_rule_values["macro_content_recall@10"]
                ),
            }
        )
        for gain_column, gains_by_profile in gain_truth.items():
            profile_gains = gains_by_profile.get(profile, {})
            if not profile_gains:
                # A binary positive without positive gain must not contribute a
                # fabricated 0.0 to gain CIs; headline gain metrics skip it too.
                continue
            gain_values = graded_hierarchical_ranking_metrics(
                {profile: ranked_rules.get(profile, [])},
                {profile: profile_gains},
                content_rules,
                gain_name=gain_column,
                rule_k=20,
                content_k=10,
            )
            profile_metrics[profile].update(
                {
                    f"{gain_column}_rule_ndcg@20": float(
                        gain_values[f"{gain_column}_rule_ndcg@20"]
                    ),
                    f"{gain_column}_macro_content_ndcg@10": float(
                        gain_values[
                            f"{gain_column}_macro_content_ndcg@10"
                        ]
                    ),
                    f"{gain_column}_overall_ndcg@200": float(
                        gain_values[f"{gain_column}_overall_ndcg@200"]
                    ),
                }
            )
            all_rule_gain_values = graded_hierarchical_ranking_metrics(
                {profile: all_eligible_ranked_rules.get(profile, [])},
                {profile: profile_gains},
                content_rules,
                gain_name=gain_column,
                rule_k=all_rule_k,
                content_k=10,
            )
            profile_metrics[profile].update(
                {
                    f"{gain_column}_all_rule_macro_content_ndcg@10": float(
                        all_rule_gain_values[
                            f"{gain_column}_macro_content_ndcg@10"
                        ]
                    ),
                    f"{gain_column}_all_rule_macro_content_gain_recall@10": float(
                        all_rule_gain_values[
                            f"{gain_column}_macro_content_gain_recall@10"
                        ]
                    ),
                }
            )
    confidence_intervals = {
        metric: bootstrap_confidence_interval(
            [values[metric] for values in profile_metrics.values()]
        )
        for metric in (
            "rule_ndcg@20",
            "macro_content_ndcg@10",
            "overall_recall@200",
        )
    }
    for gain_column in gain_truth:
        for suffix in (
            "rule_ndcg@20",
            "macro_content_ndcg@10",
            "overall_ndcg@200",
        ):
            metric = f"{gain_column}_{suffix}"
            confidence_intervals[metric] = bootstrap_confidence_interval(
                [
                    values[metric]
                    for values in profile_metrics.values()
                    if metric in values
                ]
            )
        for suffix in (
            "all_rule_macro_content_ndcg@10",
            "all_rule_macro_content_gain_recall@10",
        ):
            metric = f"{gain_column}_{suffix}"
            confidence_intervals[metric] = bootstrap_confidence_interval(
                [
                    values[metric]
                    for values in profile_metrics.values()
                    if metric in values
                ]
            )

    cohort_metrics: dict[str, Any] = {}
    positive_mask = pd.to_numeric(
        normalized["label"], errors="coerce"
    ).fillna(0).eq(1)
    for column in COHORT_COLUMNS:
        if column not in normalized:
            continue
        cohort_frame = normalized[[column, "score"]].copy()
        cohort_frame["_positive"] = positive_mask.to_numpy(dtype=bool)
        cohort_frame["_positive_score"] = cohort_frame["score"].where(
            cohort_frame["_positive"], 0.0
        )
        summaries = cohort_frame.groupby(
            column,
            dropna=False,
            observed=True,
            sort=True,
        ).agg(
            candidate_pairs=("score", "size"),
            positive_pairs=("_positive", "sum"),
            mean_score=("score", "mean"),
            positive_score_sum=("_positive_score", "sum"),
        )
        for value, row in summaries.iterrows():
            positive_pairs = int(row["positive_pairs"])
            cohort_metrics[f"{column}={value}"] = {
                "candidate_pairs": int(row["candidate_pairs"]),
                "positive_pairs": positive_pairs,
                "mean_score": float(row["mean_score"]),
                "positive_mean_score": float(row["positive_score_sum"])
                / positive_pairs
                if positive_pairs
                else 0.0,
            }
    fallback_rows = (
        int(normalized["fallback"].fillna(False).astype(bool).sum())
        if "fallback" in normalized
        else 0
    )
    # Page shape of the list actually served: how many slots, how many of them
    # carry a content the page already showed under another rule.
    page_shape = _page_shape_metrics(ranked_rules)
    variable_metrics: dict[str, Any] = {}
    if assembly_config is not None and assembly_config.variable_page:
        served_rules = ranked_rules
        page_budget = assembly_config.page_content_budget or (
            assembly_config.normal_rule_count * assembly_config.min_contents
        )
        variable_metrics.update(
            hierarchical_ranking_metrics(
                served_rules,
                truth,
                content_rules,
                rule_k=20,
                content_k=assembly_config.max_contents,
                page_k=page_budget,
                content_label="var",
                page_label="page",
            )
        )
        for gain_column, gains_by_profile in gain_truth.items():
            variable_metrics.update(
                graded_hierarchical_ranking_metrics(
                    served_rules,
                    gains_by_profile,
                    content_rules,
                    gain_name=gain_column,
                    rule_k=20,
                    content_k=assembly_config.max_contents,
                    page_k=page_budget,
                    content_label="var",
                    page_label="page",
                )
            )

    metrics: dict[str, Any] = {
        **hierarchical,
        **graded_metrics,
        "target_semantics": (
            str(truth_frame["target_semantics"].dropna().iloc[0])
            if "target_semantics" in truth_frame
            and len(truth_frame["target_semantics"].dropna())
            else "legacy_binary"
        ),
        "score_semantics": score_semantics,
        "classification_metrics_valid": score_semantics == "probability",
        "candidate_profiles": int(normalized["profile_id"].nunique()),
        "candidate_pairs": int(len(normalized)),
        "expanded_candidate_rule_rows": int(len(expanded)),
        "mean_rule_memberships_per_candidate": len(expanded)
        / max(len(normalized), 1),
        "candidate_mean_actual_k": float(candidate_counts.mean())
        if not candidate_counts.empty
        else 0.0,
        "candidate_max_actual_k": int(candidate_counts.max())
        if not candidate_counts.empty
        else 0,
        "candidate_recall@actual_k": float(actual_candidate_recall),
        "candidate_recall@1000": float(actual_candidate_recall),
        "flat_candidate_recall@200": overall["recall@200"],
        "flat_candidate_ndcg@200": overall["ndcg@200"],
        "exact_20_normal_rule_rate": (
            sum(count == 20 for count in rule_counts.values())
            / max(len(rule_counts), 1)
        ),
        # Same contract expressed two ways: exactly ten for a fixed page, or
        # within [min, max] for a variable one.
        "content_count_within_range_rate": (
            sum(
                (
                    assembly_config.min_contents
                    if assembly_config is not None
                    else 10
                )
                <= count
                <= (
                    assembly_config.max_contents
                    if assembly_config is not None
                    else 10
                )
                for count in ranked_content_counts
            )
            / max(len(ranked_content_counts), 1)
        ),
        "exact_10_content_rate": (
            sum(count == 10 for count in ranked_content_counts)
            / max(len(ranked_content_counts), 1)
        ),
        "underfilled_rule_groups_excluded": int(
            rule_ranking_diagnostics["underfilled_rule_groups_excluded"]
        ),
        "candidate_rule_groups": int(rule_ranking_diagnostics["rule_groups"]),
        "eligible_candidate_rule_groups": int(
            rule_ranking_diagnostics["eligible_rule_groups"]
        ),
        "eligible_candidate_rule_group_rate": float(
            rule_ranking_diagnostics["eligible_rule_groups"]
            / max(rule_ranking_diagnostics["rule_groups"], 1)
        ),
        "mean_eligible_candidate_rules_per_profile": float(
            rule_ranking_diagnostics["eligible_rule_groups"]
            / max(normalized["profile_id"].nunique(), 1)
        ),
        "score_variance": float(normalized["score"].var(ddof=0))
        if len(normalized)
        else 0.0,
        "mean_profile_score_variance": float(
            normalized.groupby("profile_id")["score"].var(ddof=0).fillna(0).mean()
        )
        if len(normalized)
        else 0.0,
        "fallback_candidate_rows": fallback_rows,
        "fallback_candidate_rate": fallback_rows / max(len(normalized), 1),
        **page_shape,
        **variable_metrics,
        "confidence_intervals": confidence_intervals,
        "cohorts": cohort_metrics,
        "per_profile": profile_metrics,
    }
    for gain_column in gain_truth:
        total_gain = sum(
            sum(profile_gains.values())
            for profile_gains in gain_truth[gain_column].values()
        )
        candidate_keys = set(
            normalized[["profile_id", "content_id"]]
            .drop_duplicates()
            .astype(str)
            .itertuples(index=False, name=None)
        )
        hit_gain = sum(
            gain
            for profile_id, profile_gains in gain_truth[gain_column].items()
            for content_id, gain in profile_gains.items()
            if (str(profile_id), str(content_id)) in candidate_keys
        )
        metrics[f"candidate_{gain_column}_recall@actual_k"] = (
            float(hit_gain / total_gain) if total_gain > 0 else 0.0
        )
    if classification is not None:
        metrics["classification"] = classification
    return metrics, ranked_rules
