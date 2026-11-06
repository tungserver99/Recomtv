import numpy as np
import pandas as pd

import pytest

from recsys_vnpt.export import serialize_assembled_output
from recsys_vnpt.output_validation import validate_client_output
from recsys_vnpt.retrieval import (
    RetrievalConfig,
    balanced_catalog_cap,
    build_candidate_pool,
    profile_rule_affinity_scores,
)
from recsys_vnpt.rule_assembly import SPECIAL_RULES, assemble_recommendations


def _catalog_fixture(rule_count=22, contents_per_rule=12):
    content_rows = []
    membership_rows = []
    scored_rows = []
    for rule in range(1, rule_count + 1):
        for rank in range(contents_per_rule):
            content_id = f"c{rule:02d}_{rank:02d}"
            content_rows.append(
                {
                    "content_id": content_id,
                    "type_id": "2",
                    "content_status": 1,
                    "tag_names": "tag",
                }
            )
            membership_rows.append({"content_id": content_id, "rule_id": str(rule)})
            scored_rows.append(
                {
                    "profile_id": "p1",
                    "content_id": content_id,
                    "type_id": "2",
                    "rule_id": str(rule),
                    "score": 1000 - rule * 10 - rank,
                }
            )
    return pd.DataFrame(content_rows), pd.DataFrame(membership_rows), pd.DataFrame(scored_rows)


def test_cowatch_candidates_per_rule_config_validation():
    assert RetrievalConfig().cowatch_candidates_per_rule == 0
    assert RetrievalConfig(cowatch_candidates_per_rule=80).cowatch_candidates_per_rule == 80
    with pytest.raises(ValueError, match="cowatch"):
        RetrievalConfig(cowatch_candidates_per_rule=-1)


def test_gain_rule_affinity_weighs_watch_seconds_and_recency():
    profiles = pd.DataFrame({"profile_id": ["p1"]})
    membership = pd.DataFrame(
        {"content_id": ["a", "b"], "rule_id": ["R1", "R2"]}
    )
    cutoff = pd.Timestamp("2025-02-01")
    interactions = pd.DataFrame(
        {
            "profile_id": ["p1", "p1", "p1", "p1", "p1"],
            "content_id": ["a", "b", "b", "b", "b"],
            # One long watch on R1 90 days ago; three short recent on R2;
            # one post-cutoff row that must be ignored.
            "create_date": [
                "2024-11-03",
                "2025-01-31",
                "2025-01-31",
                "2025-01-31",
                "2025-02-02",
            ],
            "duration": [100.0, 10.0, 10.0, 10.0, 100000.0],
        }
    )

    def scores(**kwargs):
        frame = profile_rule_affinity_scores(
            profiles, membership, None, interactions, cutoff, **kwargs
        )
        return dict(zip(frame["rule_id"], frame["rule_score"]))

    count = scores()
    gain = scores(affinity="gain")
    decayed = scores(affinity="gain", half_life_days=30.0)

    assert count["R2"] > count["R1"]  # 3 events beat 1 event
    assert gain["R1"] > gain["R2"]  # 100s beats 3x10s
    assert decayed["R2"] > decayed["R1"]  # 90d-old 100s decays to 12.5

    with pytest.raises(ValueError, match="count or gain"):
        scores(affinity="typo")
    with pytest.raises(ValueError, match="requires a cutoff"):
        profile_rule_affinity_scores(
            profiles, membership, None, interactions, None, half_life_days=30.0
        )
    with pytest.raises(ValueError, match="duration"):
        profile_rule_affinity_scores(
            profiles,
            membership,
            None,
            interactions.drop(columns="duration"),
            cutoff,
            affinity="gain",
        )


def test_exact_client_contract_has_20_by_10_and_special_suffix():
    contents, membership, scored = _catalog_fixture()
    recommendations = assemble_recommendations(scored)
    homepage, text = serialize_assembled_output(recommendations)
    catalog = {
        row.content_id: {
            "type_id": str(row.type_id),
            "content_status": "1",
            "recommendable": True,
        }
        for row in contents.itertuples(index=False)
    }
    valid_membership = set(membership.itertuples(index=False, name=None))
    result = validate_client_output(
        homepage,
        text,
        expected_profiles={"p1"},
        catalog=catalog,
        valid_membership=valid_membership,
    )
    assert result.passed, result.errors
    assert homepage[0]["d"].split(",")[-4:] == list(SPECIAL_RULES)
    assert result.metrics["exact_20_normal_rule_rate"] == 1.0
    assert result.metrics["exact_10_content_rate"] == 1.0
    assert result.metrics["special_no_content_rate"] == 1.0


def test_rule_assembly_accepts_amp_float16_scores():
    _, _, scored = _catalog_fixture(rule_count=20, contents_per_rule=10)
    scored["score"] = scored["score"].astype(np.float16)
    recommendations = assemble_recommendations(scored)
    assert len(recommendations["p1"]["normal_rules"]) == 20


def test_balanced_cap_and_retrieval_pool_preserve_output_eligibility():
    contents, membership, _ = _catalog_fixture()
    popularity = {row.content_id: index for index, row in enumerate(contents.itertuples())}
    capped_contents, capped_membership = balanced_catalog_cap(
        contents,
        membership,
        popularity,
        maximum_contents=220,
    )
    counts = capped_membership.groupby("rule_id")["content_id"].nunique()
    assert (counts >= 10).sum() >= 20
    assert capped_contents["content_id"].nunique() <= 220

    pool, metrics = build_candidate_pool(
        capped_contents,
        capped_membership,
        popularity,
        RetrievalConfig(global_candidates=30, safety_candidates_per_rule=10),
    )
    assert metrics["eligible_normal_rules"] >= 20
    assert pool["content_id"].nunique() <= len(capped_contents)


def test_output_validator_fails_invalid_membership():
    contents, _, scored = _catalog_fixture(rule_count=20, contents_per_rule=10)
    recommendations = assemble_recommendations(scored)
    homepage, text = serialize_assembled_output(recommendations)
    catalog = {
        row.content_id: {
            "type_id": str(row.type_id),
            "content_status": "1",
            "recommendable": True,
        }
        for row in contents.itertuples(index=False)
    }
    result = validate_client_output(
        homepage,
        text,
        expected_profiles={"p1"},
        catalog=catalog,
        valid_membership=set(),
    )
    assert not result.passed
    assert result.metrics["unmatched_rule_rate"] == 1.0


def _variable_page_fixture():
    rows = []
    for rule in range(1, 26):
        for rank in range(40):
            rows.append({
                "profile_id": "p1", "rule_id": str(rule),
                "content_id": f"c{rule:02d}_{rank:02d}", "type_id": "2",
                "score": 1000 - rule * 10 - rank,
            })
    # rule 2 shares rule 1's best contents, so de-dup must reach deeper
    for rank in range(20):
        rows.append({
            "profile_id": "p1", "rule_id": "2",
            "content_id": f"c01_{rank:02d}", "type_id": "2", "score": 990 - rank,
        })
    return pd.DataFrame(rows)


def _page(config):
    from recsys_vnpt.rule_assembly import RuleAssemblyConfig  # noqa: F401
    rules = assemble_recommendations(_variable_page_fixture(), config=config)["p1"]
    normal = rules["normal_rules"]
    sizes = [len(rule["contents"]) for rule in normal]
    contents = [c["content_id"] for rule in normal for c in rule["contents"]]
    return sizes, contents


def test_default_page_is_unchanged_fixed_20x10():
    from recsys_vnpt.rule_assembly import RuleAssemblyConfig

    sizes, contents = _page(RuleAssemblyConfig())

    assert len(sizes) == 20
    assert set(sizes) == {10}
    assert len(contents) == 200
    # duplicates across rules are still allowed in the default page
    assert len(set(contents)) < len(contents)


def test_page_dedup_keeps_every_rule_at_minimum_without_repeats():
    from recsys_vnpt.rule_assembly import RuleAssemblyConfig

    sizes, contents = _page(RuleAssemblyConfig(page_dedup="greedy_rule_rank"))

    assert set(sizes) == {10}
    assert len(contents) == 200
    assert len(set(contents)) == 200  # no content repeats anywhere on the page


def test_marginal_allocation_varies_rule_length_within_bounds_and_budget():
    from recsys_vnpt.rule_assembly import RuleAssemblyConfig

    sizes, contents = _page(
        RuleAssemblyConfig(
            contents_per_rule_max=30,
            page_content_budget=400,
            page_dedup="greedy_rule_rank",
        )
    )

    assert len(sizes) == 20
    assert min(sizes) >= 10 and max(sizes) <= 30
    assert min(sizes) < max(sizes)  # genuinely variable, not a flat split
    assert sum(sizes) == 400 == len(set(contents))


def test_rule_ranking_does_not_shift_when_rule_length_grows():
    from recsys_vnpt.rule_assembly import RuleAssemblyConfig

    frame = _variable_page_fixture()
    fixed = assemble_recommendations(frame, config=RuleAssemblyConfig())["p1"]
    variable = assemble_recommendations(
        frame,
        config=RuleAssemblyConfig(contents_per_rule_max=30, page_content_budget=400),
    )["p1"]

    order_fixed = [rule["rule_id"] for rule in fixed["normal_rules"]]
    order_variable = [rule["rule_id"] for rule in variable["normal_rules"]]
    assert order_fixed == order_variable


def test_assembly_config_rejects_impossible_settings():
    from recsys_vnpt.rule_assembly import RuleAssemblyConfig

    with pytest.raises(ValueError, match="contents_per_rule_max"):
        RuleAssemblyConfig(contents_per_rule_max=5)
    with pytest.raises(ValueError, match="page_dedup"):
        RuleAssemblyConfig(page_dedup="magic")
    with pytest.raises(ValueError, match="page_content_budget"):
        RuleAssemblyConfig(page_content_budget=100)


def test_dedup_never_serves_a_rule_below_the_minimum():
    from recsys_vnpt.rule_assembly import RuleAssemblyConfig

    # Every rule draws from a small shared pool, so strict de-duplication would
    # starve the later rules; the contract must win over de-duplication.
    rows = []
    for rule in range(1, 22):
        for index in range(12):
            content_id = f"shared_{index:02d}" if index < 6 else f"c{rule:02d}_{index:02d}"
            rows.append({
                "profile_id": "p1", "rule_id": str(rule), "content_id": content_id,
                "type_id": "2", "score": 1000 - rule * 10 - index,
            })

    served = assemble_recommendations(
        pd.DataFrame(rows), config=RuleAssemblyConfig(page_dedup="greedy_rule_rank")
    )["p1"]["normal_rules"]

    assert len(served) == 20
    assert all(len(rule["contents"]) == 10 for rule in served)
