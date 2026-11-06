"""The content scorer's sampler and rank helpers.

Both sampling strategies are pinned here because the intuitive one lost
matching the positive's popularity band makes a harder training
set but ranks below plain popularity, since the model never learns to
reject the never-watched tail that fills the real pool. The head of the rule
is what retrieval actually surfaces, so that is what the default draws from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys_vnpt import content_scorer


def _scope(contents_by_rule: dict[str, list[str]]) -> content_scorer.ContentScope:
    content_ids = sorted({c for values in contents_by_rule.values() for c in values})
    frame = pd.DataFrame(
        {
            "content_id": content_ids,
            "content_duration_log": np.zeros(len(content_ids), dtype=np.float32),
            "content_publish_year": np.zeros(len(content_ids), dtype=np.float32),
            "total_episode_log": np.zeros(len(content_ids), dtype=np.float32),
            "content_age_days": np.zeros(len(content_ids), dtype=np.float32),
            "type_index": np.zeros(len(content_ids), dtype=np.float32),
            "family_index": np.zeros(len(content_ids), dtype=np.float32),
            "category_index": np.zeros(len(content_ids), dtype=np.float32),
            "country_index": np.zeros(len(content_ids), dtype=np.float32),
        }
    )
    index = {value: position for position, value in enumerate(content_ids)}
    rules_by_content: dict[str, list[str]] = {}
    for rule_id, values in contents_by_rule.items():
        for content_id in values:
            rules_by_content.setdefault(content_id, []).append(rule_id)
    from scipy import sparse

    return content_scorer.ContentScope(
        contents=frame,
        index=index,
        contents_by_rule={
            rule_id: np.asarray(sorted(values), dtype=object)
            for rule_id, values in contents_by_rule.items()
        },
        rules_by_content={
            content_id: tuple(sorted(rules))
            for content_id, rules in rules_by_content.items()
        },
        tag_matrix=sparse.csr_matrix((len(content_ids), 0), dtype=np.float32),
        tag_idf=np.zeros(0, dtype=np.float32),
        tag_norm=np.zeros(len(content_ids), dtype=np.float32),
    )


def test_popularity_bands_cut_on_value_not_rank():
    # Nine contents nobody watched and one blockbuster: rank quartiles would
    # split the zeros into three separate "matched" buckets.
    popularity = np.array([0.0] * 9 + [500.0])
    bands = content_scorer.popularity_bands(popularity)
    assert set(bands[:9]) == {0}
    assert bands[9] > 0


def test_popularity_bands_separate_watched_levels():
    popularity = np.concatenate([np.zeros(50), np.arange(1, 51, dtype=float)])
    bands = content_scorer.popularity_bands(popularity, bands=4)
    assert set(bands[:50]) == {0}
    watched = bands[50:]
    assert watched.min() >= 1
    assert watched.max() > watched.min()
    # Monotone in popularity: a more popular content is never in a lower band.
    assert np.all(np.diff(watched) >= 0)


def test_sampler_matches_rule_and_band():
    scope = _scope(
        {
            "1": [f"a{i}" for i in range(40)],
            "2": [f"b{i}" for i in range(40)],
        }
    )
    popularity = np.zeros(scope.size, dtype=np.float32)
    for position, content_id in enumerate(scope.contents["content_id"]):
        # Every second content is popular, in both rules.
        popularity[position] = 100.0 if position % 2 == 0 else 0.0
    bands = content_scorer.popularity_bands(popularity)
    positives = pd.DataFrame(
        {
            "profile_id": ["p1"],
            "content_id": [scope.contents["content_id"].iloc[0]],
            "completion_gain": [1.0],
            "watch_time_gain": [1.0],
        }
    )
    pairs, counters = content_scorer.sample_training_pairs(
        positives, scope, bands, consumed={}, ratio=4, seed=7,
        strategy="band_matched",
    )
    negatives = pairs[pairs["sample_type"] != "observed_engagement"]
    assert len(negatives) == 4
    assert counters["same_rule_same_band"] == 4
    assert counters["any_servable"] == 0

    positive_id = str(positives["content_id"].iloc[0])
    rule = scope.rules_by_content[positive_id][0]
    same_rule = set(map(str, scope.contents_by_rule[rule]))
    positive_band = bands[scope.index[positive_id]]
    for content_id in negatives["content_id"]:
        assert content_id in same_rule
        assert bands[scope.index[content_id]] == positive_band


def test_sampler_never_returns_a_positive_or_consumed_content():
    scope = _scope({"1": [f"a{i}" for i in range(20)]})
    bands = np.zeros(scope.size, dtype=np.int16)
    positives = pd.DataFrame(
        {
            "profile_id": ["p1", "p1"],
            "content_id": ["a0", "a1"],
            "completion_gain": [1.0, 0.5],
            "watch_time_gain": [1.0, 0.5],
        }
    )
    pairs, _ = content_scorer.sample_training_pairs(
        positives, scope, bands, consumed={"p1": {"a2", "a3"}}, ratio=3, seed=11,
        strategy="band_matched",
    )
    negatives = set(pairs[pairs["sample_type"] != "observed_engagement"]["content_id"])
    assert not negatives & {"a0", "a1", "a2", "a3"}


def test_sampler_is_deterministic_for_a_seed():
    scope = _scope({"1": [f"a{i}" for i in range(30)]})
    bands = np.zeros(scope.size, dtype=np.int16)
    positives = pd.DataFrame(
        {
            "profile_id": ["p1"],
            "content_id": ["a0"],
            "completion_gain": [1.0],
            "watch_time_gain": [1.0],
        }
    )
    first, _ = content_scorer.sample_training_pairs(
        positives, scope, bands, consumed={}, ratio=4, seed=3,
        strategy="band_matched",
    )
    second, _ = content_scorer.sample_training_pairs(
        positives, scope, bands, consumed={}, ratio=4, seed=3,
        strategy="band_matched",
    )
    pd.testing.assert_frame_equal(first, second)


def test_sampler_rejects_a_ratio_below_one():
    scope = _scope({"1": ["a0", "a1"]})
    positives = pd.DataFrame(
        {
            "profile_id": ["p1"],
            "content_id": ["a0"],
            "completion_gain": [1.0],
            "watch_time_gain": [1.0],
        }
    )
    with pytest.raises(ValueError, match="ratio"):
        content_scorer.sample_training_pairs(
            positives, scope, np.zeros(2, dtype=np.int16), consumed={}, ratio=0
        )


def test_rule_head_sampler_draws_only_from_the_popular_head():
    scope = _scope({"1": [f"a{i:02d}" for i in range(60)]})
    popularity = np.zeros(scope.size, dtype=np.float32)
    # Only the first ten contents are ever watched, so only they can appear in
    # a retrieval pool -- and therefore only they are useful negatives.
    for position in range(10):
        popularity[position] = 100.0 - position
    positives = pd.DataFrame(
        {
            "profile_id": ["p1"],
            "content_id": ["a55"],
            "completion_gain": [1.0],
            "watch_time_gain": [1.0],
        }
    )
    pairs, counters = content_scorer.sample_training_pairs(
        positives,
        scope,
        content_scorer.popularity_bands(popularity),
        consumed={},
        ratio=4,
        seed=5,
        strategy="rule_head",
        popularity=popularity,
        head_size=10,
    )
    negatives = pairs[pairs["sample_type"] != "observed_engagement"]
    head = set(scope.contents["content_id"].iloc[:10])
    assert counters["same_rule_head"] == 4
    assert set(negatives["content_id"]) <= head


def test_rule_head_sampler_requires_popularity():
    scope = _scope({"1": ["a0", "a1"]})
    positives = pd.DataFrame(
        {
            "profile_id": ["p1"],
            "content_id": ["a0"],
            "completion_gain": [1.0],
            "watch_time_gain": [1.0],
        }
    )
    with pytest.raises(ValueError, match="popularity"):
        content_scorer.sample_training_pairs(
            positives, scope, np.zeros(2, dtype=np.int16), consumed={}, ratio=2
        )


def test_rank_within_rule_restarts_at_every_rule():
    frame = pd.DataFrame(
        {
            "profile_id": ["p1"] * 4,
            "candidate_rule_id": ["1", "1", "2", "2"],
            "score": [10.0, 5.0, 3.0, 1.0],
        }
    )
    ranks = content_scorer.rank_within_rule(frame, "score")
    # Ranking across the profile would give 1..4; within the rule the third
    # row is a rule head again and must match the first row's score.
    assert ranks[0] == pytest.approx(ranks[2])
    assert ranks[1] == pytest.approx(ranks[3])
    assert ranks[0] > ranks[1]
