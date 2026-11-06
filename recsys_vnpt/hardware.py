from __future__ import annotations

import shutil
import statistics
import subprocess
import threading
from dataclasses import dataclass, field


def parse_nvidia_smi_sample(text: str) -> tuple[float, float, float | None]:
    values = [value.strip() for value in text.strip().split(",")]
    if len(values) < 2:
        raise ValueError("nvidia-smi sample requires utilization and memory")
    utilization = float(values[0])
    memory_mib = float(values[1])
    power_watts = None
    if len(values) >= 3 and values[2] not in {"", "[N/A]", "N/A"}:
        power_watts = float(values[2])
    return utilization, memory_mib, power_watts


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


@dataclass
class NvidiaSmiSampler:
    interval_seconds: float = 1.0
    enabled: bool = True
    utilization: list[float] = field(default_factory=list)
    memory_mib: list[float] = field(default_factory=list)
    power_watts: list[float] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    def start(self) -> None:
        if not self.enabled or shutil.which("nvidia-smi") is None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float | int | bool]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 2))
        return self.summary()

    def _sample_loop(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
            "--id=0",
        ]
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=3,
                    creationflags=creation_flags,
                )
                utilization, memory, power = parse_nvidia_smi_sample(
                    result.stdout.splitlines()[0]
                )
                self.utilization.append(utilization)
                self.memory_mib.append(memory)
                if power is not None:
                    self.power_watts.append(power)
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self._stop.wait(self.interval_seconds)

    def summary(self) -> dict[str, float | int | bool]:
        return {
            "gpu_sampler_available": bool(self.utilization),
            "gpu_utilization_samples": len(self.utilization),
            "gpu_utilization_mean_percent": (
                float(statistics.fmean(self.utilization)) if self.utilization else 0.0
            ),
            "gpu_utilization_p95_percent": _percentile(self.utilization, 0.95),
            "gpu_utilization_max_percent": max(self.utilization, default=0.0),
            "gpu_device_memory_peak_mib": max(self.memory_mib, default=0.0),
            "gpu_power_available": bool(self.power_watts),
            "gpu_power_samples": len(self.power_watts),
            "gpu_power_mean_watts": (
                float(statistics.fmean(self.power_watts)) if self.power_watts else 0.0
            ),
        }
