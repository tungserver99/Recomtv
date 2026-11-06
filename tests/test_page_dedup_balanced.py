"""The floor phase decides where de-duplication damage lands.

`greedy_rule_rank` fills rule by rule, so rule 1 draws its whole head from an
untouched pool and the last rules draw theirs from one already stripped of
~190 contents. Every slot the page loses is therefore taken from the rules a
viewer scrolls down to. `balanced_head` fills depth-major instead: every rule
takes its best available before any rule takes its second.
"""

from __future__ import annotations

import pandas as pd

from recsys_vnpt.rule_assembly import RuleAssemblyConfig, allocate_rule_contents


def _pool(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["rule_id", "content_id", "score"])
    frame.insert(0, "profile_id", "p1")
    frame["type_id"] = "2"
    frame["fallback"] = False
    for column in ("profile_id", "rule_id", "content_id", "type_id"):
        frame[column] = frame[column].astype("string")
    return frame


def _served(frame: pd.DataFrame, rule_id: str) -> list[str]:
    rule = frame[frame["rule_id"] == rule_id]
    return rule["content_id"].astype(str).tolist()


CONTESTED = _pool(
    [
        ("A", "c1", 0.99),
        ("A", "c2", 0.98),
        ("A", "a3", 0.50),
        ("A", "a4", 0.40),
        ("B", "c1", 0.97),
        ("B", "c2", 0.96),
        ("B", "b3", 0.30),
        ("B", "b4", 0.20),
    ]
)
KEYS = pd.DataFrame({"profile_id": ["p1", "p1"], "rule_id": ["A", "B"]}).astype("string")


def test_row_major_dedup_strips_the_whole_head_of_the_later_rule() -> None:
    config = RuleAssemblyConfig(
        normal_rule_count=2,
        contents_per_rule=2,
        contents_per_rule_max=2,
        page_content_budget=4,
        page_dedup="greedy_rule_rank",
    )
    served = allocate_rule_contents(CONTESTED, KEYS, config)
    # Rule A takes both contested contents; rule B is left with its 3rd and 4th.
    assert _served(served, "A") == ["c1", "c2"]
    assert _served(served, "B") == ["b3", "b4"]


def test_balanced_head_gives_every_rule_its_best_available_first() -> None:
    config = RuleAssemblyConfig(
        normal_rule_count=2,
        contents_per_rule=2,
        contents_per_rule_max=2,
        page_content_budget=4,
        page_dedup="balanced_head",
    )
    served = allocate_rule_contents(CONTESTED, KEYS, config)
    # Depth-major: A takes c1, then B takes c2 rather than dropping to b3.
    assert _served(served, "A")[0] == "c1"
    assert _served(served, "B")[0] == "c2"
    # Still no content appears twice on the page.
    every = _served(served, "A") + _served(served, "B")
    assert len(every) == len(set(every))


def test_balanced_head_still_never_serves_a_rule_below_the_minimum() -> None:
    # Both rules can only be filled from the same two contents, so the hard
    # contract has to win over de-duplication.
    scarce = _pool(
        [
            ("A", "c1", 0.9),
            ("A", "c2", 0.8),
            ("B", "c1", 0.7),
            ("B", "c2", 0.6),
        ]
    )
    config = RuleAssemblyConfig(
        normal_rule_count=2,
        contents_per_rule=2,
        contents_per_rule_max=2,
        page_content_budget=4,
        page_dedup="balanced_head",
    )
    served = allocate_rule_contents(scarce, KEYS, config)
    assert len(_served(served, "A")) == 2
    assert len(_served(served, "B")) == 2
