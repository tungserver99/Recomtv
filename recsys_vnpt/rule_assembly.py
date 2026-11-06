from __future__ import annotations

import heapq

from dataclasses import dataclass
from typing import Any

import pandas as pd


SPECIAL_RULES = ("112", "109", "108", "110")

# Whole-page de-duplication policies. Both CLIs build their `--page-dedup`
# choices from this tuple and `test_cli_config_parity` compares against it, so
# a new policy reaches the command line in the same commit that adds it.
PAGE_DEDUP_POLICIES = (
    "none",
    "greedy_rule_rank",
    "relative_fit",
    "tail_only",
    "balanced_head",
    "balanced_regret",
)


class InsufficientRuleCoverage(ValueError):
    pass


def _rule_sort_value(rule_id: str) -> tuple[int, str]:
    try:
        return int(rule_id), rule_id
    except ValueError:
        return 2**31 - 1, rule_id


@dataclass(frozen=True)
class RuleAssemblyConfig:
    normal_rule_count: int = 20
    contents_per_rule: int = 10
    special_rules: tuple[str, ...] = SPECIAL_RULES
    # Variable-length rules and whole-page dedup. Defaults reproduce the fixed
    # 20x10 page exactly; both features are opt-in.
    contents_per_rule_max: int | None = None
    page_content_budget: int | None = None
    page_dedup: str = "none"

    @property
    def min_contents(self) -> int:
        return self.contents_per_rule

    @property
    def max_contents(self) -> int:
        return self.contents_per_rule_max or self.contents_per_rule

    @property
    def variable_page(self) -> bool:
        return self.max_contents > self.min_contents or self.page_dedup != "none"

    def __post_init__(self) -> None:
        if self.max_contents < self.min_contents:
            raise ValueError("contents_per_rule_max cannot be below contents_per_rule")
        if self.page_dedup not in PAGE_DEDUP_POLICIES:
            raise ValueError(f"unsupported page_dedup policy: {self.page_dedup}")
        budget = self.page_content_budget
        if budget is not None and budget < self.normal_rule_count * self.min_contents:
            raise ValueError(
                "page_content_budget cannot be below normal_rule_count * contents_per_rule"
            )


def _normalize_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"profile_id", "rule_id", "content_id", "type_id", "score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"scored candidates are missing columns: {missing}")
    output = frame.copy()
    for column in ("profile_id", "rule_id", "content_id", "type_id"):
        output[column] = output[column].astype("string")
    output["score"] = pd.to_numeric(output["score"], errors="coerce").astype("float32")
    output = output.dropna(subset=["profile_id", "rule_id", "content_id", "type_id", "score"])
    output = output[~output["rule_id"].isin(SPECIAL_RULES)]
    if "fallback" not in output.columns:
        output["fallback"] = False
    else:
        output["fallback"] = output["fallback"].fillna(False).astype(bool)
    return output


def assemble_profile_rules(
    scored_candidates: pd.DataFrame,
    rule_popularity: dict[str, float] | None = None,
    fallback_candidates: pd.DataFrame | None = None,
    config: RuleAssemblyConfig = RuleAssemblyConfig(),
) -> dict[str, Any]:
    candidates = _normalize_candidates(scored_candidates)
    profiles = candidates["profile_id"].drop_duplicates().tolist()
    if len(profiles) != 1:
        raise ValueError("assemble_profile_rules requires exactly one profile")
    return assemble_recommendations(
        candidates,
        rule_popularity,
        fallback_candidates,
        config,
    )[str(profiles[0])]


def allocate_rule_contents(
    pool: pd.DataFrame,
    ranked_keys: pd.DataFrame,
    config: RuleAssemblyConfig,
) -> pd.DataFrame:
    """Pick the contents each selected rule serves.

    ``pool`` holds up to ``max_contents`` candidates per (profile, rule) already
    ordered best-first; ``ranked_keys`` lists the selected rules per profile in
    serving order. With the default fixed-size, no-dedup config this is just the
    first ``contents_per_rule`` rows of each group, so the served page is
    unchanged. Otherwise every rule first secures ``min_contents`` (skipping
    contents already placed elsewhere on the page when de-duplicating), then the
    remaining page budget is handed out one slot at a time to whichever rule has
    the highest-scoring next candidate — so the cut-off emerges from comparing
    rules against each other instead of a fixed threshold.
    """

    selected = pool.merge(ranked_keys[["profile_id", "rule_id"]], on=["profile_id", "rule_id"], how="inner")
    if not config.variable_page:
        return selected.groupby(["profile_id", "rule_id"], sort=False).head(
            config.min_contents
        )

    order = {
        (str(row.profile_id), str(row.rule_id)): index
        for index, row in enumerate(ranked_keys.itertuples(index=False))
    }
    dedup = config.page_dedup != "none"
    # tail_only keeps each rule's leading contents exactly as they would be
    # without de-duplication - those are the items a viewer sees first in a
    # shelf, and a strong title legitimately belongs in several shelves - and
    # removes repeats only from the depth added beyond the minimum.
    dedup_head = dedup and config.page_dedup != "tail_only"
    budget = config.page_content_budget or (
        config.normal_rule_count * config.min_contents
    )
    keep: list[tuple[str, str, str]] = []
    for profile_id, profile_pool in selected.groupby("profile_id", sort=False):
        rules: dict[str, list[tuple[int, str, float]]] = {}
        for row in profile_pool.itertuples(index=False):
            rules.setdefault(str(row.rule_id), []).append(
                (row.Index if hasattr(row, "Index") else 0, str(row.content_id), float(row.score))
            )
        rule_order = sorted(
            rules, key=lambda rule_id: order.get((str(profile_id), rule_id), 1 << 30)
        )
        if config.page_dedup == "relative_fit":
            rule_order = _relative_fit_order(rules, rule_order)
        used: set[str] = set()
        rank_of_order = {rule_id: index for index, rule_id in enumerate(rule_order)}
        taken: dict[str, list[str]] = {rule_id: [] for rule_id in rule_order}
        cursor: dict[str, int] = {rule_id: 0 for rule_id in rule_order}

        def _next(rule_id: str) -> tuple[str, float] | None:
            items = rules[rule_id]
            index = cursor[rule_id]
            skip_used = (
                dedup_head if len(taken[rule_id]) < config.min_contents else dedup
            )
            while index < len(items):
                _, content_id, score = items[index]
                if (skip_used and content_id in used) or content_id in taken[rule_id]:
                    index += 1
                    continue
                cursor[rule_id] = index
                return content_id, score
            cursor[rule_id] = index
            return None

        def _take(rule_id: str) -> bool:
            nxt = _next(rule_id)
            if nxt is None:
                return False
            taken[rule_id].append(nxt[0])
            used.add(nxt[0])
            cursor[rule_id] += 1
            return True

        if config.page_dedup in ("balanced_head", "balanced_regret"):
            # Depth-major: every rule takes its best available before any rule
            # takes its second. Filling rule by rule instead hands the first
            # rule its whole head from an untouched pool and leaves the last
            # rules drawing theirs from one already stripped of ~190 contents,
            # so the entire de-duplication cost lands on the rules a viewer
            # scrolls down to. Spreading it charges each rule about one slot.
            def _regret(rule_id: str) -> float:
                # How much this rule loses if another takes its next candidate:
                # the score gap to the one after it. Scores share the rule term
                # inside a rule, so the gap is comparable across rules even
                # though the levels are not.
                items = rules[rule_id]
                index = cursor[rule_id]
                head: list[float] = []
                while index < len(items) and len(head) < 2:
                    _, content_id, score = items[index]
                    if content_id not in used and content_id not in taken[rule_id]:
                        head.append(score)
                    index += 1
                if not head:
                    return -1.0
                return head[0] - head[1] if len(head) > 1 else head[0]

            for _ in range(config.min_contents):
                pending = [
                    rule_id
                    for rule_id in rule_order
                    if len(taken[rule_id]) < config.min_contents
                ]
                if config.page_dedup == "balanced_regret":
                    pending.sort(
                        key=lambda rule_id: (-_regret(rule_id), rank_of_order[rule_id])
                    )
                for rule_id in pending:
                    if len(taken[rule_id]) < config.min_contents:
                        _take(rule_id)
        else:
            for rule_id in rule_order:
                while len(taken[rule_id]) < config.min_contents and _take(rule_id):
                    pass
        for rule_id in rule_order:
            if len(taken[rule_id]) < config.min_contents:
                # The contract is hard and de-duplication is best-effort: a rule
                # whose fresh candidates ran out re-uses contents already placed
                # elsewhere rather than serving fewer than the minimum. Every
                # candidate still available here is one another rule already
                # placed, so `used` cannot grow and no other rule's head moves.
                for _, content_id, _ in rules[rule_id]:
                    if len(taken[rule_id]) >= config.min_contents:
                        break
                    if content_id not in taken[rule_id]:
                        taken[rule_id].append(content_id)
                        used.add(content_id)
        remaining = budget - sum(len(v) for v in taken.values())
        # A heap over each rule's next candidate keeps the marginal allocation
        # O(budget log rules) instead of rescanning every rule for every slot;
        # entries can go stale when another rule takes the content first, so a
        # popped entry is re-checked against that rule's current next candidate.
        rank_of = {rule_id: index for index, rule_id in enumerate(rule_order)}
        frontier: list[tuple[float, int, str, str]] = []
        for rule_id in rule_order:
            if len(taken[rule_id]) >= config.max_contents:
                continue
            nxt = _next(rule_id)
            if nxt is not None:
                heapq.heappush(
                    frontier, (-nxt[1], rank_of[rule_id], rule_id, nxt[0])
                )
        while remaining > 0 and frontier:
            _, _, rule_id, content_id = heapq.heappop(frontier)
            if len(taken[rule_id]) >= config.max_contents:
                continue
            current = _next(rule_id)
            if current is None:
                continue
            if current[0] != content_id:
                heapq.heappush(
                    frontier, (-current[1], rank_of[rule_id], rule_id, current[0])
                )
                continue
            taken[rule_id].append(content_id)
            used.add(content_id)
            cursor[rule_id] += 1
            remaining -= 1
            if len(taken[rule_id]) < config.max_contents:
                following = _next(rule_id)
                if following is not None:
                    heapq.heappush(
                        frontier,
                        (-following[1], rank_of[rule_id], rule_id, following[0]),
                    )
        keep.extend(
            (str(profile_id), rule_id, content_id)
            for rule_id, contents in taken.items()
            for content_id in contents
        )
    if not keep:
        return selected.iloc[0:0]
    # One vectorised join beats scanning the (large, untrimmed) pool per profile.
    chosen = pd.DataFrame(keep, columns=["profile_id", "rule_id", "content_id"])
    for column in chosen.columns:
        chosen[column] = chosen[column].astype(selected[column].dtype)
    return selected.merge(chosen, on=list(chosen.columns), how="inner")


def _relative_fit_order(
    rules: dict[str, list[tuple[int, str, float]]], rule_order: list[str]
) -> list[str]:
    """Rule visit order that lets each content land where it stands out most.

    Rules whose best candidates are least available elsewhere go first, so a
    content that is rank 1 in a narrow rule is not consumed by a broad rule that
    merely happens to outrank it.
    """

    spread: dict[str, float] = {}
    for rule_id in rule_order:
        contents = [content_id for _, content_id, _ in rules[rule_id]]
        elsewhere = sum(
            1
            for content_id in contents
            if any(
                content_id in {c for _, c, _ in rules[other]}
                for other in rule_order
                if other != rule_id
            )
        )
        spread[rule_id] = elsewhere / max(len(contents), 1)
    return sorted(rule_order, key=lambda rule_id: (spread[rule_id], rule_order.index(rule_id)))


def assemble_recommendations(
    scored_candidates: pd.DataFrame,
    rule_popularity: dict[str, float] | None = None,
    fallback_candidates: pd.DataFrame | None = None,
    config: RuleAssemblyConfig = RuleAssemblyConfig(),
) -> dict[str, dict[str, Any]]:
    candidates = _normalize_candidates(scored_candidates)
    if candidates.empty:
        return {}
    profile_ids = set(candidates["profile_id"].astype(str))
    frames = [candidates]
    if fallback_candidates is not None and not fallback_candidates.empty:
        fallback = _normalize_candidates(fallback_candidates)
        fallback = fallback[fallback["profile_id"].astype(str).isin(profile_ids)].copy()
        fallback["fallback"] = True
        frames.append(fallback)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        ["profile_id", "rule_id", "score", "fallback", "content_id"],
        ascending=[True, True, False, True, True],
        kind="stable",
    ).drop_duplicates(["profile_id", "rule_id", "content_id"], keep="first")
    # De-duplication needs spare depth: a rule whose best contents were already
    # placed on the page must be able to reach further down its own list to
    # still meet the minimum, so the pool is only trimmed in fixed-size mode.
    pool = (
        combined
        if config.variable_page
        else combined.groupby(["profile_id", "rule_id"], sort=False).head(
            config.max_contents
        )
    )
    # Rank rules on a fixed window of their best contents so a rule that later
    # receives 30 slots is not out-ranked by a 10-slot rule purely because its
    # mean score is averaged over a longer tail.
    top = pool.groupby(["profile_id", "rule_id"], sort=False).head(
        config.min_contents
    )
    counts = top.groupby(["profile_id", "rule_id"])["content_id"].transform("size")
    top = top[counts.eq(config.min_contents)].copy()
    pool = pool.merge(
        top[["profile_id", "rule_id"]].drop_duplicates(),
        on=["profile_id", "rule_id"],
        how="inner",
    )
    stats = (
        top.groupby(["profile_id", "rule_id"], sort=False)
        .agg(
            mean_score=("score", "mean"),
            floor_score=("score", "min"),
            fallback_count=("fallback", "sum"),
            **(
                {"direct_rule_score": ("rule_score", "mean")}
                if "rule_score" in top
                else {}
            ),
        )
        .reset_index()
    )
    if "direct_rule_score" not in stats:
        stats["direct_rule_score"] = stats["mean_score"]
    else:
        stats["direct_rule_score"] = stats["direct_rule_score"].fillna(
            stats["mean_score"]
        )
    popularity = {str(key): float(value) for key, value in (rule_popularity or {}).items()}
    stats["popularity"] = stats["rule_id"].astype(str).map(popularity).fillna(0.0)
    stats["_numeric_rule"] = pd.to_numeric(stats["rule_id"], errors="coerce").fillna(2**31 - 1)
    stats = stats.sort_values(
        [
            "profile_id",
            "direct_rule_score",
            "mean_score",
            "floor_score",
            "popularity",
            "_numeric_rule",
            "rule_id",
        ],
        ascending=[True, False, False, False, False, True, True],
        kind="stable",
    )
    selected = stats.groupby("profile_id", sort=False).head(config.normal_rule_count).copy()
    selected_counts = selected.groupby("profile_id").size().to_dict()
    for profile_id in sorted(profile_ids):
        eligible_count = int(selected_counts.get(profile_id, 0))
        if eligible_count != config.normal_rule_count:
            raise InsufficientRuleCoverage(
                f"Profile {profile_id} has {eligible_count} eligible rules; "
                f"requires {config.normal_rule_count} rules with {config.min_contents} contents"
            )

    selected_top = allocate_rule_contents(pool, selected, config)
    selected_top = selected_top.sort_values(
        ["profile_id", "rule_id", "score", "fallback", "content_id"],
        ascending=[True, True, False, True, True],
        kind="stable",
    )
    content_map = {
        (str(profile_id), str(rule_id)): [
            {
                "content_id": str(row.content_id),
                "type_id": str(row.type_id),
                "score": float(row.score),
                "fallback": bool(row.fallback),
            }
            for row in group.itertuples(index=False)
        ]
        for (profile_id, rule_id), group in selected_top.groupby(
            ["profile_id", "rule_id"], sort=False
        )
    }
    output: dict[str, dict[str, Any]] = {}
    for profile_id, rows in selected.groupby("profile_id", sort=True):
        normal_rules = [
            {
                "rule_id": str(row.rule_id),
                "contents": content_map[(str(profile_id), str(row.rule_id))],
                "mean_score": float(row.mean_score),
                "floor_score": float(row.floor_score),
                "rule_score": float(row.direct_rule_score),
                "popularity": float(row.popularity),
                "fallback_count": int(row.fallback_count),
            }
            for row in rows.itertuples(index=False)
        ]
        output[str(profile_id)] = {
            "profile_id": str(profile_id),
            "normal_rules": normal_rules,
            "special_rules": list(config.special_rules),
        }
    return output


def bayesian_rule_popularity(
    content_scores: pd.Series | dict[str, float],
    membership: pd.DataFrame,
    prior_strength: float = 10.0,
) -> dict[str, float]:
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive")
    if not {"content_id", "rule_id"}.issubset(membership):
        raise ValueError("rule popularity requires content_id/rule_id membership")
    scores = pd.Series(content_scores, dtype=float)
    scores.index = scores.index.astype("string")
    rows = membership[["content_id", "rule_id"]].drop_duplicates().copy()
    rows["content_id"] = rows["content_id"].astype("string")
    rows["rule_id"] = rows["rule_id"].astype("string")
    rows = rows[~rows["rule_id"].isin(SPECIAL_RULES)]
    rows["score"] = rows["content_id"].map(scores).fillna(0.0)
    prior_mean = float(rows["score"].mean()) if not rows.empty else 0.0
    grouped = rows.groupby("rule_id")["score"].agg(["sum", "count"])
    smoothed = (grouped["sum"] + prior_strength * prior_mean) / (
        grouped["count"] + prior_strength
    )
    return {str(rule_id): float(value) for rule_id, value in smoothed.items()}


def rank_scored_rules(
    scored_candidates: pd.DataFrame,
    rule_popularity: dict[str, float] | None = None,
    rule_k: int = 20,
    content_k: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    output, _ = rank_scored_rules_with_diagnostics(
        scored_candidates,
        rule_popularity=rule_popularity,
        rule_k=rule_k,
        content_k=content_k,
    )
    return output


def rank_scored_rules_with_diagnostics(
    scored_candidates: pd.DataFrame,
    rule_popularity: dict[str, float] | None = None,
    rule_k: int = 20,
    content_k: int = 10,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    selected, _, diagnostics = rank_scored_rules_and_all_with_diagnostics(
        scored_candidates,
        rule_popularity=rule_popularity,
        rule_k=rule_k,
        content_k=content_k,
        include_all_eligible=False,
    )
    return selected, diagnostics


def _pool_depth(config: "RuleAssemblyConfig | None", content_k: int) -> int:
    """How deep to read each rule's candidate list before ranking."""

    if config is None or not config.variable_page:
        return content_k
    # De-duplication may skip many of a rule's best contents, so keep the whole
    # list available; the allocator trims it back to the served length.
    return 1 << 30


def rank_scored_rules_and_all_with_diagnostics(
    scored_candidates: pd.DataFrame,
    rule_popularity: dict[str, float] | None = None,
    rule_k: int = 20,
    content_k: int = 10,
    include_all_eligible: bool = True,
    config: RuleAssemblyConfig | None = None,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, int],
]:
    """Rank production rules and retain every eligible rule for diagnostics."""
    candidates = _normalize_candidates(scored_candidates)
    popularity = {str(key): float(value) for key, value in (rule_popularity or {}).items()}
    diagnostics = {
        "rule_groups": 0,
        "underfilled_rule_groups_excluded": 0,
        "eligible_rule_groups": 0,
    }
    if candidates.empty:
        return {}, {}, diagnostics

    group_columns = ["profile_id", "rule_id"]
    direct_scores = None
    if "rule_score" in candidates:
        candidates["rule_score"] = pd.to_numeric(
            candidates["rule_score"], errors="coerce"
        )
        direct_scores = (
            candidates.groupby(group_columns, sort=False, observed=True)[
                "rule_score"
            ]
            .mean()
            .rename("direct_rule_score")
            .reset_index()
        )

    ranked_contents = (
        candidates.sort_values(
            ["profile_id", "rule_id", "score", "content_id"],
            ascending=[True, True, False, True],
            kind="stable",
        )
        .drop_duplicates(
            ["profile_id", "rule_id", "content_id"], keep="first"
        )
        .groupby(group_columns, sort=False, observed=True)
        .head(_pool_depth(config, content_k))
        .copy()
    )
    minimum_contents = config.min_contents if config is not None else content_k
    group_counts = ranked_contents.groupby(
        group_columns, sort=False, observed=True
    ).size()
    diagnostics = {
        "rule_groups": int(len(group_counts)),
        "underfilled_rule_groups_excluded": int(
            ((group_counts > 0) & (group_counts < minimum_contents)).sum()
        ),
        "eligible_rule_groups": int(group_counts.ge(minimum_contents).sum()),
    }
    eligible_keys = (
        group_counts[group_counts.ge(minimum_contents)]
        .index.to_frame(index=False)
    )
    ranked_contents = ranked_contents.merge(
        eligible_keys,
        on=group_columns,
        how="inner",
        sort=False,
    )
    # Rank rules on a fixed window so a longer rule is not penalised for the
    # lower-scoring tail it is allowed to serve (mirrors the serving assembler).
    ranking_contents = ranked_contents.groupby(
        group_columns, sort=False, observed=True
    ).head(minimum_contents)
    if ranked_contents.empty:
        profiles = candidates["profile_id"].astype(str).drop_duplicates()
        empty = {profile: [] for profile in profiles}
        return empty, empty.copy(), diagnostics

    ranking_contents = ranking_contents.copy()
    ranking_contents["_score64"] = ranking_contents["score"].astype("float64")
    stats = (
        ranking_contents.groupby(group_columns, sort=False, observed=True)
        .agg(
            mean_score=("_score64", "mean"),
            floor_score=("_score64", "min"),
        )
        .reset_index()
    )
    if direct_scores is not None:
        stats = stats.merge(
            direct_scores,
            on=group_columns,
            how="left",
            sort=False,
        )
        stats["direct_rule_score"] = stats["direct_rule_score"].fillna(
            stats["mean_score"]
        )
    else:
        stats["direct_rule_score"] = stats["mean_score"]
    stats["popularity"] = stats["rule_id"].astype(str).map(popularity).fillna(0.0)
    stats["_numeric_rule"] = pd.to_numeric(
        stats["rule_id"], errors="coerce"
    ).fillna(2**31 - 1)
    ordered = stats.sort_values(
        [
            "profile_id",
            "direct_rule_score",
            "mean_score",
            "floor_score",
            "popularity",
            "_numeric_rule",
            "rule_id",
        ],
        ascending=[True, False, False, False, False, True, True],
        kind="stable",
    )
    selected = (
        ordered
        .groupby("profile_id", sort=False, observed=True)
        .head(rule_k)
    )
    def _content_map(frame: pd.DataFrame) -> dict[tuple[str, str], list[str]]:
        return {
            (str(profile_id), str(rule_id)): list(map(str, content_ids))
            for (profile_id, rule_id), content_ids in frame.groupby(
                group_columns, sort=False, observed=True
            )["content_id"]
            .agg(list)
            .items()
        }

    # The all-rule diagnostic keeps the fixed window across every eligible rule;
    # only the served page follows the variable allocation.
    content_map = _content_map(ranking_contents)
    served_map = content_map
    if config is not None and config.variable_page:
        served_map = _content_map(
            allocate_rule_contents(ranked_contents, selected, config)
        )

    def format_rows(
        rows: pd.DataFrame,
        content_map: dict[tuple[str, str], list[str]] = content_map,
    ) -> dict[str, list[dict[str, Any]]]:
        output: dict[str, list[dict[str, Any]]] = {}
        for profile_id, profile_rows in rows.groupby("profile_id", sort=True):
            output[str(profile_id)] = [
                {
                    "rule_id": str(row.rule_id),
                    "content_ids": list(
                        content_map.get((str(profile_id), str(row.rule_id)), [])
                    ),
                    "rule_score": float(row.direct_rule_score),
                    "mean_score": float(row.mean_score),
                    "floor_score": float(row.floor_score),
                    "popularity": float(row.popularity),
                }
                for row in profile_rows.itertuples(index=False)
            ]
        for profile_id in candidates["profile_id"].astype(str).drop_duplicates():
            output.setdefault(profile_id, [])
        return output

    all_eligible = format_rows(ordered) if include_all_eligible else {}
    return format_rows(selected, served_map), all_eligible, diagnostics


def explode_content_rule_membership(
    scored_contents: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    required_scores = {"profile_id", "content_id", "type_id", "score"}
    required_membership = {"content_id", "rule_id"}
    if not required_scores.issubset(scored_contents) or not required_membership.issubset(membership):
        raise ValueError("membership expansion requires scored profile/content rows and content/rule rows")
    scores = scored_contents.copy()
    rules = membership[["content_id", "rule_id"]].drop_duplicates().copy()
    scores["content_id"] = scores["content_id"].astype("string")
    rules["content_id"] = rules["content_id"].astype("string")
    return scores.merge(rules, on="content_id", how="inner")


def aggregate_model_rule_scores(
    scored_contents: pd.DataFrame,
    membership: pd.DataFrame,
    contribution_column: str = "model_rule_score",
    content_k: int = 10,
) -> pd.DataFrame:
    """Aggregate pair-level rule contributions over actual memberships."""
    required_scores = {"profile_id", "content_id", contribution_column}
    required_membership = {"content_id", "rule_id"}
    if not required_scores.issubset(scored_contents):
        raise ValueError(
            f"model rule scoring requires {sorted(required_scores)}"
        )
    if not required_membership.issubset(membership):
        raise ValueError("model rule scoring requires content-rule membership")
    contributions = scored_contents[
        ["profile_id", "content_id", contribution_column]
    ].copy()
    contributions["profile_id"] = contributions["profile_id"].astype("string")
    contributions["content_id"] = contributions["content_id"].astype("string")
    contributions[contribution_column] = pd.to_numeric(
        contributions[contribution_column], errors="coerce"
    )
    rules = membership[["content_id", "rule_id"]].drop_duplicates().copy()
    rules = rules.astype({"content_id": "string", "rule_id": "string"})
    rules = rules[~rules["rule_id"].isin(SPECIAL_RULES)]
    expanded = contributions.merge(rules, on="content_id", how="inner")
    expanded = expanded.dropna(subset=[contribution_column])
    top = (
        expanded.sort_values(
            ["profile_id", "rule_id", contribution_column, "content_id"],
            ascending=[True, True, False, True],
            kind="stable",
        )
        .drop_duplicates(["profile_id", "rule_id", "content_id"])
        .groupby(["profile_id", "rule_id"], sort=False, observed=True)
        .head(content_k)
    )
    return (
        top.groupby(
            ["profile_id", "rule_id"], as_index=False, sort=False, observed=True
        )[contribution_column]
        .mean()
        .rename(columns={contribution_column: "rule_score"})
    )
