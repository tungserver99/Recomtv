"""Kaggle runtime, resource, and real CUDA-kernel preflight checks."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def cuda_probe(torch_module: Any) -> dict[str, Any]:
    """Require a real, synchronized tensor operation on the CUDA device."""
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    try:
        left = torch_module.tensor([2.0], device="cuda")
        right = left * 3.0
        expected = torch_module.tensor([6.0], device="cuda")
        if not bool(torch_module.allclose(right, expected)):
            raise RuntimeError("CUDA result mismatch")
        torch_module.cuda.synchronize()
    except Exception as error:
        raise RuntimeError(f"CUDA operation failed: {error}") from error
    return {
        "gpu_verified": True,
        "gpu_name": torch_module.cuda.get_device_name(0),
        "capability": list(torch_module.cuda.get_device_capability(0)),
        "torch_version": str(torch_module.__version__),
        "cuda_runtime": str(torch_module.version.cuda),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia_smi_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,compute_cap,memory.total,driver_version",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "error": str(error)}
    return {
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _cuda_failure_diagnostic(
    error: BaseException, packages: dict[str, str | None], nvidia_smi: dict[str, Any]
) -> str:
    return (
        f"{error}; torch={packages.get('torch')!r}, "
        f"nvidia-smi={nvidia_smi.get('stdout') or nvidia_smi.get('error')!r}. "
        "Select a Kaggle accelerator and PyTorch/CUDA build supported by that GPU, "
        "then rerun; CPU fallback is not allowed for this gate."
    )


def preflight(working_root: Path, require_gpu: bool) -> dict[str, Any]:
    """Capture software/resources and fail closed when the GPU gate is required."""
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError("psutil is required for Kaggle resource preflight") from error

    disk = shutil.disk_usage(working_root)
    versions = {
        name: _package_version(name)
        for name in ("torch", "numpy", "pandas", "scikit-learn", "pytest")
    }
    nvidia_smi = _nvidia_smi_snapshot()
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "disk_free_bytes": disk.free,
        "memory_available_bytes": psutil.virtual_memory().available,
        "packages": versions,
        "nvidia_smi": nvidia_smi,
    }

    if require_gpu:
        try:
            import torch
        except ImportError as error:
            diagnostic = _cuda_failure_diagnostic(
                RuntimeError("PyTorch is unavailable for required CUDA preflight"),
                versions,
                nvidia_smi,
            )
            print(diagnostic, file=sys.stderr)
            raise RuntimeError(diagnostic) from error
        versions["torch"] = str(torch.__version__)
        report["packages"] = versions
        try:
            report.update(cuda_probe(torch))
        except Exception as error:
            diagnostic = _cuda_failure_diagnostic(error, versions, nvidia_smi)
            print(diagnostic, file=sys.stderr)
            raise RuntimeError(diagnostic) from error

    minimum = int(os.environ.get("RECSYS_KAGGLE_MIN_FREE_BYTES", str(2 * 1024**3)))
    report["minimum_free_bytes"] = minimum
    if disk.free < minimum:
        raise RuntimeError(
            f"insufficient Kaggle working space: {disk.free} < {minimum}"
        )
    return report
