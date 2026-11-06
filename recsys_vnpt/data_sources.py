from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import math
import xml.etree.ElementTree as ET

import pandas as pd

from recsys_vnpt.paths import source_artifact_path


USER_METADATA_NAME = "month_mytv_info.parquet"
REQUIRED_USER_COLUMNS = ("username", "province", "tentinh", "sex", "birthday", "package_code")
REQUIRED_DURATION_COLUMNS = ("profile_id", "username", "content_id", "duration")
USER_COLUMN_ALIASES = {
    "user_name": "username",
    "province_code": "province",
    "province_name": "tentinh",
    "gender": "sex",
    "birth_date": "birthday",
    "product_code": "package_code",
    "product_id": "package_code",
}
SOURCE_ARTIFACTS = {
    "data/data/duration",
    "data/data/mytv_vmp_content",
    "data/data/tags",
    "data/data/content_tags",
    "data/data/behavior",
}
GENERATED_ARTIFACT_NAMES = {"train_data", "infer_data", "merged_duration", "model", "result"}


def resolve_user_metadata_path(project_root: str | Path | None = None) -> Path:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    preferred = source_artifact_path(USER_METADATA_NAME, project_root=root)
    fallback = root / USER_METADATA_NAME
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Missing user metadata. Expected {preferred} or fallback {fallback}."
    )


def load_user_metadata(
    project_root: str | Path | None = None,
    path: str | Path | None = None,
    aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    source = Path(path) if path is not None else resolve_user_metadata_path(project_root)
    frame = pd.read_parquet(source)
    rename = aliases or USER_COLUMN_ALIASES
    rename_pairs = {
        old: new
        for old, new in rename.items()
        if old in frame.columns and new not in frame.columns
    }
    frame = frame.rename(columns=rename_pairs)
    missing = [column for column in REQUIRED_USER_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{source} is missing required user metadata columns: {missing}"
        )
    return frame


def load_duration_sources(source_dir: str | Path) -> pd.DataFrame:
    source = Path(source_dir)
    frames: list[pd.DataFrame] = []
    for path in sorted(source.glob("*.csv")):
        frames.append(_read_duration_csv(path))
    for path in sorted(source.glob("*.parquet")):
        frames.append(_select_duration_columns(pd.read_parquet(path), path))
    if not frames:
        raise FileNotFoundError(f"No CSV or parquet duration sources found under {source}")
    return pd.concat(frames, ignore_index=True)


def write_merged_duration_parts(
    source_dir: str | Path,
    output_dir: str | Path,
    parts: int = 5,
) -> list[Path]:
    if parts < 1:
        raise ValueError("parts must be at least 1")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = load_duration_sources(source_dir)
    if frame.empty:
        return []

    rows_per_part = math.ceil(len(frame) / parts)
    paths: list[Path] = []
    for index in range(parts):
        start = index * rows_per_part
        end = min(start + rows_per_part, len(frame))
        if start >= len(frame):
            break
        path = output / f"merged_part_{index + 1}.parquet"
        frame.iloc[start:end].to_parquet(path, index=False)
        paths.append(path)
    return paths


def classify_artifacts(project_root: str | Path) -> dict[str, dict[str, str]]:
    root = Path(project_root)
    report: dict[str, dict[str, str]] = {}

    for relative in sorted(SOURCE_ARTIFACTS):
        path = root / relative
        if path.exists():
            report[relative] = {"kind": "source", "path": str(path)}

    try:
        user_metadata_path = resolve_user_metadata_path(root)
    except FileNotFoundError:
        user_metadata_path = None
    if user_metadata_path is not None:
        try:
            relative = user_metadata_path.relative_to(root).as_posix()
        except ValueError:
            relative = user_metadata_path.as_posix()
        report.setdefault(relative, {"kind": "source", "path": str(user_metadata_path)})

    for path in sorted(root.rglob("*")):
        if not path.exists():
            continue
        relative = path.relative_to(root).as_posix()
        if path.name in GENERATED_ARTIFACT_NAMES or any(part in GENERATED_ARTIFACT_NAMES for part in path.parts):
            report.setdefault(relative, {"kind": "generated", "path": str(path)})
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            report.setdefault(relative, {"kind": "schema_documentation", "path": str(path)})

    return report


def load_excel_schema_fields(path: str | Path) -> dict[str, dict[str, str]]:
    workbook = Path(path)
    with ZipFile(workbook) as zf:
        shared = _read_shared_strings(zf)
        sheet_paths = _read_sheet_paths(zf)
        return {
            sheet_name: _read_sheet_fields(zf, sheet_path, shared)
            for sheet_name, sheet_path in sheet_paths.items()
        }


def _read_duration_csv(path: Path) -> pd.DataFrame:
    return _select_duration_columns(pd.read_csv(path, usecols=list(REQUIRED_DURATION_COLUMNS)), path)


def _select_duration_columns(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    missing = [column for column in REQUIRED_DURATION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing duration columns: {missing}")
    output = frame.loc[:, list(REQUIRED_DURATION_COLUMNS)].copy()
    output["duration"] = pd.to_numeric(output["duration"], errors="coerce").fillna(0)
    return output


def _read_shared_strings(zf: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root.findall("main:si", ns):
        strings.append("".join(text.text or "" for text in item.findall(".//main:t", ns)))
    return strings


def _read_sheet_paths(zf: ZipFile) -> dict[str, str]:
    ns = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    rel_key = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    relmap = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall("pkgrel:Relationship", ns)
    }
    paths: dict[str, str] = {}
    for sheet in workbook.findall(".//main:sheet", ns):
        target = relmap[sheet.attrib[rel_key]]
        paths[sheet.attrib["name"]] = "xl/" + target.lstrip("/") if not target.startswith("xl/") else target
    return paths


def _read_sheet_fields(zf: ZipFile, sheet_path: str, shared: list[str]) -> dict[str, str]:
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(zf.read(sheet_path))
    fields: dict[str, str] = {}
    for row in root.findall(".//main:sheetData/main:row", ns):
        cells = {
            _column_name(cell.attrib.get("r", "")): _cell_text(cell, shared)
            for cell in row.findall("main:c", ns)
        }
        field = cells.get("B", "").strip()
        dtype = cells.get("C", "").strip()
        if field and field != "Tên trường":
            fields[field] = dtype
    return fields


def _cell_text(cell: ET.Element, shared: list[str]) -> str:
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    value = cell.find("main:v", ns)
    if cell.attrib.get("t") == "s" and value is not None:
        return shared[int(value.text)]
    if cell.attrib.get("t") == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//main:t", ns))
    return value.text if value is not None else ""


def _column_name(reference: str) -> str:
    return "".join(char for char in reference if char.isalpha())
