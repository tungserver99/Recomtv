from __future__ import annotations

import pytest

from recsys_vnpt.hardware import NvidiaSmiSampler, parse_nvidia_smi_sample


def test_parse_nvidia_smi_sample() -> None:
    assert parse_nvidia_smi_sample("42, 1234, 55.5") == (42.0, 1234.0, 55.5)
    assert parse_nvidia_smi_sample("4, 1637, [N/A]") == (4.0, 1637.0, None)


def test_parse_nvidia_smi_sample_rejects_short_rows() -> None:
    with pytest.raises(ValueError, match="utilization and memory"):
        parse_nvidia_smi_sample("42")


def test_disabled_gpu_sampler_has_explicit_empty_metrics() -> None:
    sampler = NvidiaSmiSampler(enabled=False)
    sampler.start()

    assert sampler.stop() == {
        "gpu_sampler_available": False,
        "gpu_utilization_samples": 0,
        "gpu_utilization_mean_percent": 0.0,
        "gpu_utilization_p95_percent": 0.0,
        "gpu_utilization_max_percent": 0.0,
        "gpu_device_memory_peak_mib": 0.0,
        "gpu_power_available": False,
        "gpu_power_samples": 0,
        "gpu_power_mean_watts": 0.0,
    }
