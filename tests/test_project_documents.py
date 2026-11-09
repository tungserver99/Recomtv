"""The root documents carry the client contract and the operating procedure, so
the facts they must state, and the files they point at, are tested."""

import inspect
import re
from dataclasses import fields
from pathlib import Path

from recsys_vnpt import content_scorer, rule_scorer
from recsys_vnpt.batch_contracts import EXIT_CODES, BatchConfig
from recsys_vnpt.engagement import DEFAULT_SESSION_GAP_MINUTES
from recsys_vnpt.retrieval import RetrievalConfig
from recsys_vnpt.rule_assembly import RuleAssemblyConfig
from recsys_vnpt.serving import serving_retrieval_config


REQUIRED_DOCS = [
    "README.md",
    "PRD.md",
    "OPERATIONS.md",
    "ARCHITECTURE.md",
    "SPEC.md",
    "METRICS.md",
    "TODO.md",
    "docs/KAGGLE_TESTING.md",
]


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _text(name: str) -> str:
    return (_root() / name).read_text(encoding="utf-8")


def test_the_document_set_exists():
    for name in REQUIRED_DOCS:
        assert _text(name).strip(), f"{name} is empty"


def test_documents_only_point_at_documents_that_exist():
    missing = {}
    for name in REQUIRED_DOCS:
        for reference in re.findall(r"`([\w./-]+\.md)`", _text(name)):
            if not (_root() / reference).exists() and not (_root() / "docs" / reference).exists():
                missing.setdefault(name, []).append(reference)
    assert not missing, f"dangling document references: {missing}"


def test_prd_carries_the_client_contract():
    text = _text("PRD.md")
    for token in (
        "MyTV",
        "non-sequential",
        "rule_process.py",
        "profile_id",
        "112,109,108,110",
        "trainable",
        "output-compatible-but-untrained",
        "excluded",
        "mytv_vmp_content",
        "duration",
        "_READY.json",
        "CURRENT.json",
    ):
        assert token in text, f"PRD.md does not state: {token}"


def test_architecture_describes_the_pipeline_not_the_product():
    text = _text("ARCHITECTURE.md")
    for token in (
        "non-sequential",
        "engagement_v1",
        "completion_gain",
        "watch_time_gain",
        "FeatureContract",
        "candidate_budget",
        "PAGE_DEDUP_POLICIES",
        "rule_ndcg@20",
        "macro_content_ndcg@10",
        "ServingContext",
    ):
        assert token in text, f"ARCHITECTURE.md does not describe: {token}"


def test_metrics_defines_the_primary_metrics():
    text = _text("METRICS.md")
    for token in ("rule_ndcg@20", "macro_content_ndcg@10", "overall_recall@page", "candidate_recall@1000"):
        assert token in text, f"METRICS.md does not define {token}"


def test_todo_is_the_integration_checklist_for_vnpt():
    text = _text("TODO.md")
    for token in ("_READY.json", "CURRENT.json", "qualify-hardware", "rename", "- [ ]"):
        assert token in text, f"TODO.md does not cover: {token}"


def test_documents_carry_no_measured_results():
    """Evaluation results live outside the repository, in .local/."""
    for name in REQUIRED_DOCS:
        text = _text(name)
        assert "Kết quả đo" not in text and "Thử nghiệm đã đóng" not in text, name


def test_operations_documents_every_failure_code():
    text = _text("OPERATIONS.md")
    for code, exit_code in EXIT_CODES.items():
        assert re.search(rf"\|\s*{exit_code}\s*\|\s*`{code}`", text), f"OPERATIONS.md lacks {exit_code} {code}"


def _section(text: str, keyword: str) -> str:
    """Body of the first level-2 section whose heading contains ``keyword``."""
    parts = re.split(r"^## ", text, flags=re.M)[1:]
    return next(part for part in parts if keyword in part.splitlines()[0])


def _cells(section: str, name: str) -> list[str]:
    """Cells of the table row whose first cell is ``name``, backticks removed."""
    for line in section.splitlines():
        cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
        if line.startswith("|") and cells[0] == name:
            return cells
    raise AssertionError(f"SPEC.md has no table row for {name}")


def test_spec_lists_every_scorer_feature_and_default():
    spec = _text("SPEC.md")
    batch = BatchConfig("incoming", "work", "delivery", "registry")
    for keyword, module, fit, fusion in (
        ("Rule scorer", rule_scorer, rule_scorer.fit_rule_scorer, "rule_scorer_fusion_weight"),
        ("Content scorer", content_scorer, content_scorer.fit_content_scorer, "content_scorer_fusion_weight"),
    ):
        section = _section(spec, keyword)
        for feature in module.FEATURE_COLUMNS:
            _cells(section, feature)
        for name, parameter in inspect.signature(fit).parameters.items():
            if parameter.default is not inspect.Parameter.empty:
                assert _cells(section, name)[1] == str(parameter.default), f"{keyword}: {name}"
        assert _cells(section, fusion)[1] == str(getattr(batch, fusion))
    content = _section(spec, "Content scorer")
    for feature in content_scorer.CATEGORICAL_FEATURES:
        assert "categorical" in _cells(content, feature)[1]
    assert _cells(_section(spec, "Label"), "session_gap_minutes")[1] == str(DEFAULT_SESSION_GAP_MINUTES)


def test_spec_retrieval_and_page_tables_match_library_and_serving_defaults():
    spec = _text("SPEC.md")
    batch = BatchConfig("incoming", "work", "delivery", "registry")
    served = serving_retrieval_config(
        candidate_budget=batch.candidate_budget,
        recency_candidates_per_rule=batch.recency_candidates_per_rule,
        recency_reserved_per_rule=batch.recency_reserved_per_rule,
        recency_max_age_days=batch.recency_max_age_days,
        candidate_rules=batch.candidate_rules,
        reserved_per_rule=batch.reserved_per_rule,
        source_candidates_per_rule=batch.source_candidates_per_rule,
    )
    retrieval = _section(spec, "Retrieval")
    for field in fields(RetrievalConfig):
        expected = [str(getattr(RetrievalConfig(), field.name)), str(getattr(served, field.name))]
        assert _cells(retrieval, field.name)[1:3] == expected, field.name
    page = _section(spec, "Dựng trang")
    for field in fields(RuleAssemblyConfig):
        if field.name == "special_rules":
            continue
        expected = [str(getattr(RuleAssemblyConfig(), field.name)), str(getattr(batch.assembly, field.name))]
        assert _cells(page, field.name)[1:3] == expected, field.name
