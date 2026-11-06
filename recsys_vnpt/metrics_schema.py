from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


METRICS_SCHEMA_VERSION = 2


@dataclass
class RunManifest:
    run_id: str
    domain: str
    phase: str
    model_family: str
    created_at_utc: str
    git_commit: str = "unknown"
    dirty_worktree: bool = False
    seed: int = 42
    split: dict[str, Any] = field(default_factory=dict)
    hashes: dict[str, str] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    software: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    schema_version: int = METRICS_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunManifest":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in allowed})


@dataclass
class MetricBundle:
    manifest: RunManifest
    data: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    inference: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.manifest.schema_version != METRICS_SCHEMA_VERSION:
            raise ValueError(f"unsupported metrics schema: {self.manifest.schema_version}")
        if not self.manifest.run_id or not self.manifest.domain:
            raise ValueError("run_id and domain are required")
        for section_name in ("data", "training", "evaluation", "inference", "output"):
            section = getattr(self, section_name)
            if not isinstance(section, dict):
                raise TypeError(f"{section_name} metrics must be a mapping")


@dataclass(frozen=True)
class GateCheck:
    metric: str
    passed: bool
    actual: float | int | str | None
    expected: str


@dataclass
class GateResult:
    phase: str
    passed: bool
    checks: list[GateCheck]

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "passed": self.passed,
            "checks": [asdict(check) for check in self.checks],
        }
