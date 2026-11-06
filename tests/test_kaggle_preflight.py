from types import SimpleNamespace
import sys

import pytest

from experiments.kaggle_preflight import cuda_probe, preflight


class FakeTensor:
    def __init__(self, value, error=None):
        self.value, self.error = value, error

    def __mul__(self, multiplier):
        if self.error:
            raise self.error
        return FakeTensor(self.value * multiplier)


class FakeTorch:
    __version__ = "test"
    version = SimpleNamespace(cuda="test")

    def __init__(self, available, multiply_error=None):
        self.multiply_error = multiply_error
        self.cuda = SimpleNamespace(
            is_available=lambda: available,
            get_device_name=lambda _index: "fake GPU",
            get_device_capability=lambda _index: (7, 5),
            synchronize=lambda: None,
        )

    def tensor(self, values, device):
        assert device == "cuda"
        return FakeTensor(values[0], self.multiply_error)

    def allclose(self, left, right):
        return left.value == right.value


def test_cuda_probe_rejects_broken_kernel():
    fake = FakeTorch(available=True, multiply_error=RuntimeError("no kernel image"))
    with pytest.raises(RuntimeError, match="CUDA operation failed"):
        cuda_probe(fake)


def test_cuda_probe_rejects_cpu_only_torch():
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        cuda_probe(FakeTorch(available=False))


def test_cuda_probe_accepts_successful_kernel():
    result = cuda_probe(FakeTorch(available=True))

    assert result["gpu_verified"] is True
    assert result["capability"] == [7, 5]


def test_preflight_rejects_unmet_space_budget(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=1024)),
    )
    monkeypatch.setenv("RECSYS_KAGGLE_MIN_FREE_BYTES", str(10**30))

    with pytest.raises(RuntimeError, match="insufficient Kaggle working space"):
        preflight(tmp_path, require_gpu=False)
