from __future__ import annotations

from math import log2
from statistics import mean
from typing import Any, Iterable

import numpy as np

def _dcg(hits: list[int]) -> float:
    return sum(hit / log2(idx + 2) for idx, hit in enumerate(hits))


def _average_precision(hits: list[int], relevant_count: int) -> float:
    if relevant_count == 0:
        return 0.0
    precision_sum = 0.0
    hit_count = 0
    for idx, hit in enumerate(hits, start=1):
        if hit:
            hit_count += 1
            precision_sum += hit_count / idx
    return precision_sum / min(relevant_count, len(hits))


def ranking_metrics(
    recommendations: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    k_values: Iterable[int] = (10, 50, 100, 200),
) -> dict[str, float]:
    users = [user for user in recommendations if ground_truth.get(user)]
    metrics: dict[str, float] = {"users_evaluated": len(users)}
    if not users:
        for k in k_values:
            metrics[f"recall@{k}"] = 0.0
            metrics[f"hit_rate@{k}"] = 0.0
            metrics[f"map@{k}"] = 0.0
            metrics[f"ndcg@{k}"] = 0.0
        return metrics

    for k in k_values:
        recalls: list[float] = []
        hit_rates: list[float] = []
        maps: list[float] = []
        ndcgs: list[float] = []
        for user in users:
            truth = set(ground_truth[user])
            top_k = recommendations.get(user, [])[:k]
            hits = [1 if item in truth else 0 for item in top_k]
            hit_count = sum(hits)
            recalls.append(hit_count / len(truth))
            hit_rates.append(1.0 if hit_count else 0.0)
            maps.append(_average_precision(hits, len(truth)))
            ideal_hits = [1] * min(len(truth), k)
            ideal_dcg = _dcg(ideal_hits)
            ndcgs.append(_dcg(hits) / ideal_dcg if ideal_dcg else 0.0)
        metrics[f"recall@{k}"] = mean(recalls)
        metrics[f"hit_rate@{k}"] = mean(hit_rates)
        metrics[f"map@{k}"] = mean(maps)
        metrics[f"ndcg@{k}"] = mean(ndcgs)
    return metrics


def output_health_metrics(
    result_with_rule: dict[str, Any],
    candidate_content_count: int,
    top_n: int | None = None,
) -> dict[str, float | int]:
    profiles_scored = len(result_with_rule)
    recommendations_returned = 0
    unmatched_rule_count = 0
    distinct_content_ids: set[str] = set()

    for profile_result in result_with_rule.values():
        suggested = profile_result.get("suggested_content", {})
        recommendations_returned += len(suggested)
        for content_id, content in suggested.items():
            distinct_content_ids.add(str(content_id))
            if str(content.get("rule_id")) == "-100":
                unmatched_rule_count += 1

    metrics: dict[str, float | int] = {
        "profiles_scored": profiles_scored,
        "recommendations_returned": recommendations_returned,
        "candidate_contents": int(candidate_content_count),
        "distinct_recommended_contents": len(distinct_content_ids),
        "catalog_coverage": (
            len(distinct_content_ids) / candidate_content_count
            if candidate_content_count > 0
            else 0.0
        ),
        "average_recommendations_per_profile": (
            recommendations_returned / profiles_scored
            if profiles_scored > 0
            else 0.0
        ),
        "unmatched_rule_count": unmatched_rule_count,
        "unmatched_rule_rate": (
            unmatched_rule_count / recommendations_returned
            if recommendations_returned > 0
            else 0.0
        ),
    }
    if top_n is not None:
        metrics["top_n"] = int(top_n)
    return metrics


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    if not relevant:
        return 0.0
    hits = [1 if str(item) in relevant else 0 for item in recommended[:k]]
    ideal = [1] * min(len(relevant), k)
    ideal_dcg = _dcg(ideal)
    return _dcg(hits) / ideal_dcg if ideal_dcg else 0.0


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(map(str, recommended[:k])) & set(map(str, relevant))) / len(relevant)


def graded_ndcg_at_k(
    recommended: list[str],
    gains: dict[str, float],
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    normalized = {
        str(item): max(float(gain), 0.0)
        for item, gain in gains.items()
        if np.isfinite(float(gain)) and float(gain) > 0
    }
    if not normalized:
        return 0.0
    observed = [normalized.get(str(item), 0.0) for item in recommended[:k]]
    ideal = sorted(normalized.values(), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    return _dcg(observed) / ideal_dcg if ideal_dcg else 0.0


def gain_recall_at_k(
    recommended: list[str],
    gains: dict[str, float],
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    normalized = {
        str(item): max(float(gain), 0.0)
        for item, gain in gains.items()
        if np.isfinite(float(gain)) and float(gain) > 0
    }
    denominator = sum(normalized.values())
    if denominator <= 0:
        return 0.0
    unique = list(dict.fromkeys(map(str, recommended[:k])))
    return sum(normalized.get(item, 0.0) for item in unique) / denominator


def hierarchical_ranking_metrics(
    recommendations: dict[str, list[dict[str, Any]]],
    ground_truth: dict[str, set[str]],
    content_rules: dict[str, set[str]],
    rule_k: int = 20,
    content_k: int = 10,
    page_k: int | None = None,
    content_label: str | None = None,
    page_label: str | None = None,
) -> dict[str, float | int]:
    """Evaluate ordered rules and content within each rule on future positives.

    ``page_k`` is the number of content slots the page actually serves; it
    defaults to ``rule_k * content_k`` so a fixed-size page is unchanged. The
    label overrides let a variable-length page publish its metrics under stable
    names (e.g. ``macro_content_ndcg@var``) instead of names that drift with k.

    Comparability note: the recall family normalises by the truth set, so it is
    comparable across page shapes; the NDCG family normalises by an ideal that
    depends on k, so NDCG at different k values is not directly comparable.
    """

    page_budget = page_k if page_k is not None else rule_k * content_k
    content_name = content_label if content_label is not None else str(content_k)
    page_name = page_label if page_label is not None else str(page_budget)

    rule_ndcgs: list[float] = []
    rule_recalls: list[float] = []
    rule_hits: list[float] = []
    content_ndcgs: list[float] = []
    content_recalls: list[float] = []
    content_hits: list[float] = []
    overall_ndcgs: list[float] = []
    overall_recalls: list[float] = []
    profiles_without_truth = 0

    for profile in sorted(map(str, ground_truth)):
        truth = set(map(str, ground_truth.get(profile, set())))
        if not truth:
            profiles_without_truth += 1
            continue
        rows = recommendations.get(profile, [])
        truth_rules = {
            str(rule_id)
            for content_id in truth
            for rule_id in content_rules.get(str(content_id), set())
        }
        recommended_rules = [str(row["rule_id"]) for row in rows[:rule_k]]
        rule_ndcgs.append(ndcg_at_k(recommended_rules, truth_rules, rule_k))
        rule_recalls.append(recall_at_k(recommended_rules, truth_rules, rule_k))
        rule_hits.append(float(bool(set(recommended_rules) & truth_rules)))

        recommended_by_rule: dict[str, list[str]] = {}
        flattened: list[str] = []
        seen_contents: set[str] = set()
        for row in rows[:rule_k]:
            rule_id = str(row["rule_id"])
            contents = [str(item) for item in row.get("content_ids", [])]
            recommended_by_rule[rule_id] = contents[:content_k]
            for content_id in contents[:content_k]:
                if content_id not in seen_contents:
                    flattened.append(content_id)
                    seen_contents.add(content_id)

        profile_content_ndcgs: list[float] = []
        profile_content_recalls: list[float] = []
        profile_content_hits: list[float] = []
        for rule_id in sorted(truth_rules):
            contents = recommended_by_rule.get(rule_id, [])
            relevant_in_rule = {
                content_id
                for content_id in truth
                if rule_id in set(map(str, content_rules.get(content_id, set())))
            }
            profile_content_ndcgs.append(
                ndcg_at_k(contents, relevant_in_rule, content_k)
            )
            profile_content_recalls.append(
                recall_at_k(contents, relevant_in_rule, content_k)
            )
            profile_content_hits.append(
                float(bool(set(contents[:content_k]) & relevant_in_rule))
            )
        content_ndcgs.append(
            mean(profile_content_ndcgs) if profile_content_ndcgs else 0.0
        )
        content_recalls.append(
            mean(profile_content_recalls) if profile_content_recalls else 0.0
        )
        content_hits.append(
            mean(profile_content_hits) if profile_content_hits else 0.0
        )
        overall_ndcgs.append(ndcg_at_k(flattened, truth, page_budget))
        overall_recalls.append(recall_at_k(flattened, truth, page_budget))

    def average(values: list[float]) -> float:
        return mean(values) if values else 0.0

    return {
        "profiles_evaluated": len(rule_ndcgs),
        "profiles_without_truth": profiles_without_truth,
        f"rule_ndcg@{rule_k}": average(rule_ndcgs),
        f"rule_recall@{rule_k}": average(rule_recalls),
        f"rule_hit_rate@{rule_k}": average(rule_hits),
        f"macro_content_ndcg@{content_name}": average(content_ndcgs),
        f"macro_content_recall@{content_name}": average(content_recalls),
        f"macro_content_hit_rate@{content_name}": average(content_hits),
        f"overall_ndcg@{page_name}": average(overall_ndcgs),
        f"overall_recall@{page_name}": average(overall_recalls),
    }


def graded_hierarchical_ranking_metrics(
    recommendations: dict[str, list[dict[str, Any]]],
    ground_truth_gains: dict[str, dict[str, float]],
    content_rules: dict[str, set[str]],
    gain_name: str,
    rule_k: int = 20,
    content_k: int = 10,
    page_k: int | None = None,
    content_label: str | None = None,
    page_label: str | None = None,
) -> dict[str, float | int]:
    """Evaluate the same serving hierarchy with continuous engagement gain."""

    page_budget = page_k if page_k is not None else rule_k * content_k
    content_name = content_label if content_label is not None else str(content_k)
    page_name = page_label if page_label is not None else str(page_budget)

    rule_ndcgs: list[float] = []
    rule_gain_recalls: list[float] = []
    content_ndcgs: list[float] = []
    content_gain_recalls: list[float] = []
    overall_ndcgs: list[float] = []
    overall_gain_recalls: list[float] = []
    profiles_without_gain = 0

    for profile in sorted(map(str, ground_truth_gains)):
        gains = {
            str(content_id): max(float(value), 0.0)
            for content_id, value in ground_truth_gains.get(profile, {}).items()
            if np.isfinite(float(value)) and float(value) > 0
        }
        if not gains:
            profiles_without_gain += 1
            continue
        rows = recommendations.get(profile, [])[:rule_k]
        rule_gains: dict[str, float] = {}
        for content_id, gain in gains.items():
            for rule_id in content_rules.get(content_id, set()):
                normalized_rule = str(rule_id)
                rule_gains[normalized_rule] = (
                    rule_gains.get(normalized_rule, 0.0) + gain
                )
        recommended_rules = [str(row["rule_id"]) for row in rows]
        rule_ndcgs.append(
            graded_ndcg_at_k(recommended_rules, rule_gains, rule_k)
        )
        rule_gain_recalls.append(
            gain_recall_at_k(recommended_rules, rule_gains, rule_k)
        )

        recommended_by_rule: dict[str, list[str]] = {}
        flattened: list[str] = []
        seen_contents: set[str] = set()
        for row in rows:
            rule_id = str(row["rule_id"])
            contents = [str(item) for item in row.get("content_ids", [])][
                :content_k
            ]
            recommended_by_rule[rule_id] = contents
            for content_id in contents:
                if content_id not in seen_contents:
                    flattened.append(content_id)
                    seen_contents.add(content_id)

        per_rule_ndcg: list[float] = []
        per_rule_recall: list[float] = []
        for rule_id in sorted(rule_gains):
            gains_in_rule = {
                content_id: gain
                for content_id, gain in gains.items()
                if rule_id in set(map(str, content_rules.get(content_id, set())))
            }
            per_rule_ndcg.append(
                graded_ndcg_at_k(
                    recommended_by_rule.get(rule_id, []),
                    gains_in_rule,
                    content_k,
                )
            )
            per_rule_recall.append(
                gain_recall_at_k(
                    recommended_by_rule.get(rule_id, []),
                    gains_in_rule,
                    content_k,
                )
            )
        content_ndcgs.append(mean(per_rule_ndcg) if per_rule_ndcg else 0.0)
        content_gain_recalls.append(
            mean(per_rule_recall) if per_rule_recall else 0.0
        )
        overall_ndcgs.append(
            graded_ndcg_at_k(flattened, gains, page_budget)
        )
        overall_gain_recalls.append(
            gain_recall_at_k(flattened, gains, page_budget)
        )

    def average(values: list[float]) -> float:
        return mean(values) if values else 0.0

    prefix = str(gain_name)
    return {
        f"{prefix}_profiles_evaluated": len(rule_ndcgs),
        f"{prefix}_profiles_without_gain": profiles_without_gain,
        f"{prefix}_rule_ndcg@{rule_k}": average(rule_ndcgs),
        f"{prefix}_rule_gain_recall@{rule_k}": average(rule_gain_recalls),
        f"{prefix}_macro_content_ndcg@{content_name}": average(content_ndcgs),
        f"{prefix}_macro_content_gain_recall@{content_name}": average(
            content_gain_recalls
        ),
        f"{prefix}_overall_ndcg@{page_name}": average(overall_ndcgs),
        f"{prefix}_overall_gain_recall@{page_name}": average(
            overall_gain_recalls
        ),
    }


def bootstrap_confidence_interval(
    values: Iterable[float],
    confidence: float = 0.95,
    iterations: int = 2000,
    seed: int = 42,
) -> dict[str, float | int]:
    samples = np.asarray(list(values), dtype=float)
    samples = samples[np.isfinite(samples)]
    if not 0 < confidence < 1 or iterations < 1:
        raise ValueError("confidence must be in (0, 1) and iterations positive")
    if samples.size == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "samples": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=float)
    for index in range(iterations):
        means[index] = rng.choice(samples, size=samples.size, replace=True).mean()
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(samples.mean()),
        "lower": float(np.quantile(means, alpha)),
        "upper": float(np.quantile(means, 1.0 - alpha)),
        "samples": int(samples.size),
    }


def paired_bootstrap_confidence_interval(
    baseline: dict[str, float],
    candidate: dict[str, float],
    confidence: float = 0.95,
    iterations: int = 2000,
    seed: int = 42,
) -> dict[str, float | int]:
    shared = sorted(set(map(str, baseline)) & set(map(str, candidate)))
    differences = [
        float(candidate[profile]) - float(baseline[profile]) for profile in shared
    ]
    result = bootstrap_confidence_interval(
        differences, confidence=confidence, iterations=iterations, seed=seed
    )
    result["paired_profiles"] = len(shared)
    return result


def hierarchical_relative_harmonic_score(
    candidate_rule_ndcg: float,
    candidate_content_ndcg: float,
    baseline_rule_ndcg: float,
    baseline_content_ndcg: float,
) -> float:
    ratios = []
    for candidate_value, baseline_value in (
        (candidate_rule_ndcg, baseline_rule_ndcg),
        (candidate_content_ndcg, baseline_content_ndcg),
    ):
        if baseline_value <= 0:
            ratios.append(1.0 if candidate_value > 0 else 0.0)
        else:
            ratios.append(candidate_value / baseline_value)
    return (
        float(2.0 / ((1.0 / ratios[0]) + (1.0 / ratios[1])))
        if min(ratios) > 0
        else 0.0
    )
