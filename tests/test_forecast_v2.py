import pytest

from recsys_vnpt.forecast import RuntimeObservation, fit_runtime_forecast


def test_runtime_forecast_uses_holdout_and_refuses_a100_inference():
    observations = [
        RuntimeObservation(profiles=100, scored_pairs=50_000, seconds=2.0, hardware="cpu"),
        RuntimeObservation(profiles=200, scored_pairs=100_000, seconds=3.5, hardware="cpu"),
        RuntimeObservation(profiles=400, scored_pairs=200_000, seconds=6.5, hardware="cpu"),
        RuntimeObservation(profiles=800, scored_pairs=400_000, seconds=12.5, hardware="cpu"),
    ]
    result = fit_runtime_forecast(observations, target_profiles=(1_000_000,))
    assert result["model_type"] == "constant_candidates_per_profile"
    assert result["forecasts"]["1000000"]["median_seconds"] > 0
    assert result["a100_projection_status"] == "requires_a100_shadow_observations"
    assert result["holdout"]["forecast_error_rate"] < 0.10

    with pytest.raises(ValueError, match="at least three"):
        fit_runtime_forecast(observations[:2])
