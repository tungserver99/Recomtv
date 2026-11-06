import pandas as pd

from experiments.retrieval_attribution import _add_content_diagnostic_attributes


def test_content_diagnostic_attributes_use_observed_history_and_cutoff() -> None:
    attributes = pd.DataFrame(
        {
            "profile_id": ["target"] * 4,
            "content_id": ["zero", "low", "medium", "high"],
        }
    )
    observed_history = pd.DataFrame(
        [
            {"profile_id": f"low-{index}", "content_id": "low"}
            for index in range(2)
        ]
        + [
            {"profile_id": f"medium-{index}", "content_id": "medium"}
            for index in range(21)
        ]
        + [
            {"profile_id": f"high-{index}", "content_id": "high"}
            for index in range(101)
        ]
    )
    contents = pd.DataFrame(
        {
            "content_id": ["zero", "low", "medium", "high"],
            "publish_date": [
                pd.NaT,
                pd.Timestamp("2025-02-01"),
                pd.Timestamp("2025-01-26"),
                pd.Timestamp("2025-01-11"),
            ],
        }
    )

    result = _add_content_diagnostic_attributes(
        attributes,
        observed_history,
        contents,
        cutoff="2025-01-31",
    ).set_index("content_id")

    assert result["historical_popularity_bucket"].to_dict() == {
        "zero": "0",
        "low": "1-20",
        "medium": "21-100",
        "high": "101+",
    }
    assert result["latest_publish_age_bucket"].to_dict() == {
        "zero": "missing",
        "low": "latest_after_cutoff",
        "medium": "0-7d",
        "high": "8-30d",
    }
