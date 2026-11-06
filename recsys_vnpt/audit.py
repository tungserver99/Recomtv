from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from recsys_vnpt.config import DOMAIN_CONFIGS
from recsys_vnpt.data_sources import classify_artifacts
from recsys_vnpt.paths import rule_info_path, source_artifact_path, source_root


@dataclass(frozen=True)
class ContentAuditStatus:
    status: str
    reason: str
    pattern_count: int = 0


@dataclass(frozen=True)
class AuditReport:
    root: Path
    artifact_root: Path
    missing_artifacts: list[str]
    rule_info_rows: int
    rule_id_count: int
    content_status: dict[str, ContentAuditStatus]
    artifact_classification: dict[str, dict[str, str]]

    def to_markdown(self) -> str:
        lines = [
            "# Current State Audit",
            "",
            "## Artifact Root",
            "",
            f"- `{self.artifact_root}`",
            "",
            "## Missing Artifacts",
        ]
        if self.missing_artifacts:
            lines.extend(f"- `{item}`" for item in self.missing_artifacts)
        else:
            lines.append("- None detected")
        lines.extend(
            [
                "",
                "## Rule Info",
                f"- Rows: {self.rule_info_rows}",
                f"- Distinct rule IDs: {self.rule_id_count}",
                "",
                "## Content Coverage",
            ]
        )
        for domain, status in self.content_status.items():
            lines.append(f"- `{domain}`: {status.status} ({status.reason})")
        if self.artifact_classification:
            lines.extend(["", "## Artifact Classification"])
            for relative, info in sorted(self.artifact_classification.items()):
                lines.append(f"- `{relative}`: {info['kind']}")
        return "\n".join(lines) + "\n"


def _count_content_patterns(root: Path, patterns: tuple[str, ...]) -> int:
    content_root = source_artifact_path("mytv_vmp_content", project_root=root)
    if not content_root.exists():
        return 0
    count = 0
    for pattern in patterns:
        count += len(list(content_root.glob(f"**/{pattern}")))
    return count


def audit_project(root: str | Path) -> AuditReport:
    root = Path(root)
    artifacts_root = source_root(root)
    expected = [
        "duration",
        "mytv_vmp_content",
        "tags",
        "month_mytv_info.parquet",
        "rule_info.parquet",
    ]
    missing = [artifact for artifact in expected if not (artifacts_root / artifact).exists()]
    if "rule_info.parquet" in missing and (root / "rule_info.parquet").exists():
        missing.remove("rule_info.parquet")
    if "month_mytv_info.parquet" in missing and (root / "month_mytv_info.parquet").exists():
        missing.remove("month_mytv_info.parquet")

    rules_path = rule_info_path(root)
    if rules_path.exists():
        rule_info = pd.read_parquet(rules_path)
        rule_info_rows = len(rule_info)
        rule_id_count = int(rule_info["RULE_ID"].nunique()) if "RULE_ID" in rule_info.columns else 0
    else:
        rule_info_rows = 0
        rule_id_count = 0

    content_status: dict[str, ContentAuditStatus] = {}
    for domain, config in DOMAIN_CONFIGS.items():
        pattern_count = _count_content_patterns(root, config.content_patterns)
        if pattern_count == 0:
            content_status[domain] = ContentAuditStatus(
                status="excluded",
                reason="missing content source",
                pattern_count=0,
            )
        elif "duration" in missing:
            content_status[domain] = ContentAuditStatus(
                status="output-compatible-but-untrained",
                reason="missing interaction duration logs",
                pattern_count=pattern_count,
            )
        else:
            content_status[domain] = ContentAuditStatus(
                status="trainable",
                reason="content and interaction artifacts present",
                pattern_count=pattern_count,
            )

    return AuditReport(
        root=root,
        artifact_root=artifacts_root,
        missing_artifacts=missing,
        rule_info_rows=rule_info_rows,
        rule_id_count=rule_id_count,
        content_status=content_status,
        artifact_classification=classify_artifacts(root),
    )
