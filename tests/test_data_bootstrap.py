from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from recsys_vnpt.bootstrap import ensure_checkpoint
from recsys_vnpt.audit import audit_project
from recsys_vnpt.data_sources import (
    classify_artifacts,
    load_duration_sources,
    load_excel_schema_fields,
    load_user_metadata,
    resolve_user_metadata_path,
    write_merged_duration_parts,
)


def test_resolve_user_metadata_prefers_data_root_and_falls_back_to_project_root(tmp_path):
    root = tmp_path
    nested = root / "data" / "data"
    nested.mkdir(parents=True)
    fallback = root / "month_mytv_info.parquet"
    preferred = nested / "month_mytv_info.parquet"
    pd.DataFrame(
        [
            {
                "username": "u-root",
                "province": "HNI",
                "tentinh": "Ha Noi",
                "sex": 1,
                "birthday": "19900101",
                "package_code": "MYTV021",
            }
        ]
    ).to_parquet(fallback, index=False)

    assert resolve_user_metadata_path(root) == fallback

    pd.DataFrame(
        [
            {
                "username": "u-data",
                "province": "HNI",
                "tentinh": "Ha Noi",
                "sex": 1,
                "birthday": "19900101",
                "package_code": "MYTV021",
            }
        ]
    ).to_parquet(preferred, index=False)

    assert resolve_user_metadata_path(root) == preferred


def test_load_user_metadata_applies_aliases_and_reports_missing_columns(tmp_path):
    source = tmp_path / "month_mytv_info.parquet"
    pd.DataFrame(
        [
            {
                "user_name": "u1",
                "province_code": "HNI",
                "province_name": "Ha Noi",
                "gender": 1,
                "birth_date": "19900101",
                "product_code": "MYTV021",
            }
        ]
    ).to_parquet(source, index=False)

    loaded = load_user_metadata(path=source)

    assert loaded.iloc[0].to_dict() == {
        "username": "u1",
        "province": "HNI",
        "tentinh": "Ha Noi",
        "sex": 1,
        "birthday": "19900101",
        "package_code": "MYTV021",
    }

    bad = tmp_path / "bad.parquet"
    pd.DataFrame([{"username": "u1"}]).to_parquet(bad, index=False)
    with pytest.raises(ValueError, match="missing required user metadata columns"):
        load_user_metadata(path=bad)

    member_only = tmp_path / "member_only.parquet"
    pd.DataFrame(
        [
            {
                "member_id": "m1",
                "province": "HNI",
                "tentinh": "Ha Noi",
                "sex": 1,
                "birthday": "19900101",
                "package_code": "MYTV021",
            }
        ]
    ).to_parquet(member_only, index=False)
    with pytest.raises(ValueError, match="username"):
        load_user_metadata(path=member_only)


def test_artifact_classification_marks_source_generated_and_schema_docs(tmp_path):
    root = tmp_path
    for relative in [
        "data/data/duration",
        "data/data/mytv_vmp_content",
        "data/data/tags",
        "data/data/content_tags",
        "data/data/behavior",
        "clip/train_data",
        "clip/infer_data",
        "clip/merged_duration",
        "data/data/model",
        "clip/result",
    ]:
        (root / relative).mkdir(parents=True)
    workbook = root / "Dữ-liệu-phục-vụ-Recommend.xlsx"
    workbook.write_bytes(b"placeholder")

    report = classify_artifacts(root)

    assert report["data/data/duration"]["kind"] == "source"
    assert report["data/data/mytv_vmp_content"]["kind"] == "source"
    assert report["data/data/tags"]["kind"] == "source"
    assert report["data/data/content_tags"]["kind"] == "source"
    assert report["data/data/behavior"]["kind"] == "source"
    assert report["clip/train_data"]["kind"] == "generated"
    assert report["clip/infer_data"]["kind"] == "generated"
    assert report["clip/merged_duration"]["kind"] == "generated"
    assert report["data/data/model"]["kind"] == "generated"
    assert report["clip/result"]["kind"] == "generated"
    assert report["Dữ-liệu-phục-vụ-Recommend.xlsx"]["kind"] == "schema_documentation"


def test_artifact_classification_marks_resolved_user_metadata_as_source(tmp_path):
    fallback = tmp_path / "month_mytv_info.parquet"
    fallback.write_bytes(b"parquet placeholder")

    report = classify_artifacts(tmp_path)

    assert report["month_mytv_info.parquet"]["kind"] == "source"

    preferred = tmp_path / "data" / "data" / "month_mytv_info.parquet"
    preferred.parent.mkdir(parents=True)
    preferred.write_bytes(b"parquet placeholder")

    report = classify_artifacts(tmp_path)

    assert report["data/data/month_mytv_info.parquet"]["kind"] == "source"


def test_audit_report_includes_source_and_generated_artifact_summary(tmp_path):
    (tmp_path / "data" / "data" / "duration").mkdir(parents=True)
    (tmp_path / "clip" / "train_data").mkdir(parents=True)

    text = audit_project(tmp_path).to_markdown()

    assert "## Artifact Classification" in text
    assert "`data/data/duration`: source" in text
    assert "`clip/train_data`: generated" in text


def test_audit_accepts_root_user_metadata_fallback(tmp_path):
    (tmp_path / "data" / "data").mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "username": "u1",
                "province": "HNI",
                "tentinh": "Ha Noi",
                "sex": 1,
                "birthday": "19900101",
                "package_code": "MYTV021",
            }
        ]
    ).to_parquet(tmp_path / "month_mytv_info.parquet", index=False)

    report = audit_project(tmp_path)

    assert "month_mytv_info.parquet" not in report.missing_artifacts


def test_load_duration_sources_reads_csv_and_parquet_required_columns(tmp_path):
    source = tmp_path / "duration"
    source.mkdir()
    pd.DataFrame(
        [
            {"profile_id": "p1", "username": "u1", "content_id": "c1", "duration": 12, "extra": "x"},
            {"profile_id": "p2", "username": "u2", "content_id": "c2", "duration": 30, "extra": "y"},
        ]
    ).to_csv(source / "part.csv", index=False)
    pd.DataFrame(
        [{"profile_id": "p3", "username": "u3", "content_id": "c3", "duration": 40}]
    ).to_parquet(source / "part.parquet", index=False)

    loaded = load_duration_sources(source)

    assert loaded.to_dict("records") == [
        {"profile_id": "p1", "username": "u1", "content_id": "c1", "duration": 12},
        {"profile_id": "p2", "username": "u2", "content_id": "c2", "duration": 30},
        {"profile_id": "p3", "username": "u3", "content_id": "c3", "duration": 40},
    ]


def test_write_merged_duration_parts_is_deterministic(tmp_path):
    source = tmp_path / "duration"
    output = tmp_path / "merged"
    source.mkdir()
    pd.DataFrame(
        [
            {"profile_id": f"p{i}", "username": f"u{i}", "content_id": f"c{i}", "duration": i}
            for i in range(5)
        ]
    ).to_csv(source / "duration.csv", index=False)

    paths = write_merged_duration_parts(source, output, parts=2)

    assert [path.name for path in paths] == ["merged_part_1.parquet", "merged_part_2.parquet"]
    assert [len(pd.read_parquet(path)) for path in paths] == [3, 2]


def test_ensure_checkpoint_requires_explicit_train_if_missing(tmp_path):
    checkpoint = tmp_path / "best_model.pth"
    calls = []

    with pytest.raises(FileNotFoundError, match="--train-if-missing"):
        ensure_checkpoint("clip", checkpoint, train_if_missing=False, train_command=lambda: calls.append("train"))
    assert calls == []

    result = ensure_checkpoint(
        "clip",
        checkpoint,
        train_if_missing=True,
        train_command=lambda: (calls.append("train"), checkpoint.write_text("ok", encoding="utf-8")),
    )

    assert result == checkpoint
    assert calls == ["train"]


def test_load_excel_schema_fields_reads_dictionary_workbook(tmp_path):
    workbook = tmp_path / "dictionary.xlsx"
    sheet_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1"><c r="A1" t="inlineStr"><is><t>Tên bảng: </t></is></c><c r="B1" t="inlineStr"><is><t>it_customer_info</t></is></c></row>
    <row r="3"><c r="A3" t="inlineStr"><is><t>Key</t></is></c><c r="B3" t="inlineStr"><is><t>Tên trường</t></is></c><c r="C3" t="inlineStr"><is><t>Kiểu dữ liệu</t></is></c></row>
    <row r="4"><c r="B4" t="inlineStr"><is><t>username</t></is></c><c r="C4" t="inlineStr"><is><t>STRING</t></is></c></row>
    <row r="5"><c r="B5" t="inlineStr"><is><t>package_code</t></is></c><c r="C5" t="inlineStr"><is><t>STRING</t></is></c></row>
  </sheetData>
</worksheet>"""
    workbook_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="it_customer_info" sheetId="1" r:id="rId1"/></sheets>
</workbook>"""
    rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"""
    with ZipFile(workbook, "w") as zf:
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", rels_xml)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)

    fields = load_excel_schema_fields(workbook)

    assert fields["it_customer_info"]["username"] == "STRING"
    assert fields["it_customer_info"]["package_code"] == "STRING"


def test_domain_inference_scripts_expose_train_if_missing_flag():
    root = Path(__file__).resolve().parents[1]
    for domain in ("clip", "movie", "music", "all"):
        text = (root / domain / "infer.py").read_text(encoding="utf-8")
        assert "--train-if-missing" in text
        assert "ensure_checkpoint" in text
