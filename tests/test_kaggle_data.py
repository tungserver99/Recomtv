import sys
from pathlib import Path

import pytest

from experiments.kaggle_data import prepare_kaggle_data_root

# Kaggle is Linux; creating symlinks on Windows needs developer mode.
SYMLINKS = pytest.mark.skipif(sys.platform == "win32", reason="requires symlink creation")


def _make_raw_snapshot(root: Path) -> None:
    for name in ("behavior", "duration", "mytv_vmp_content", "tags", "content_tags"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "month_mytv_info.parquet").write_bytes(b"metadata")


@SYMLINKS
def test_kaggle_data_links_sources_without_copy(tmp_path):
    source = tmp_path / "input" / "data" / "data"
    _make_raw_snapshot(source)
    (source / "ignored-private-note.txt").write_text("do not mount", encoding="utf-8")
    target = tmp_path / "working" / "data"

    links = prepare_kaggle_data_root(source, target)

    assert (target / "behavior").is_symlink()
    assert (target / "month_mytv_info.parquet").is_symlink()
    assert (target / "behavior").resolve() == (source / "behavior").resolve()
    assert links["behavior"] == str((source / "behavior").resolve())
    assert (target / "canonical").is_dir()
    assert (target / "metrics").is_dir()
    assert (target / "model").is_dir()
    assert not (target / "ignored-private-note.txt").exists()


@SYMLINKS
def test_kaggle_data_links_prepared_snapshot_read_only(tmp_path):
    source = tmp_path / "input"
    (source / "canonical").mkdir(parents=True)
    (source / "canonical" / "source_manifest.json").write_text("{}", encoding="utf-8")
    target = tmp_path / "working"

    prepare_kaggle_data_root(source, target)

    assert (target / "canonical").is_symlink()
    assert (target / "canonical" / "source_manifest.json").exists()
    assert (target / "metrics").is_dir()
    assert (target / "model").is_dir()


def test_kaggle_data_rejects_missing_or_conflicting_sources(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    with pytest.raises(FileNotFoundError, match="raw source or canonical"):
        prepare_kaggle_data_root(source, tmp_path / "working")

    _make_raw_snapshot(source)
    target = tmp_path / "working"
    target.mkdir()
    (target / "content_tags").mkdir()
    with pytest.raises(FileExistsError, match="content_tags"):
        prepare_kaggle_data_root(source, target)
    assert not (target / "duration").exists()


@SYMLINKS
def test_kaggle_data_allows_reuse_of_matching_symlink(tmp_path):
    source = tmp_path / "input"
    _make_raw_snapshot(source)
    target = tmp_path / "working"
    target.mkdir()
    (target / "behavior").symlink_to(source / "behavior", target_is_directory=True)

    prepare_kaggle_data_root(source, target)

    assert (target / "behavior").resolve() == (source / "behavior").resolve()


@SYMLINKS
def test_kaggle_data_rejects_symlink_to_another_source(tmp_path):
    source = tmp_path / "input"
    other = tmp_path / "other" / "behavior"
    _make_raw_snapshot(source)
    other.mkdir(parents=True)
    target = tmp_path / "working"
    target.mkdir()
    (target / "behavior").symlink_to(other, target_is_directory=True)

    with pytest.raises(FileExistsError, match="behavior"):
        prepare_kaggle_data_root(source, target)


def test_kaggle_data_rejects_partial_raw_even_with_canonical(tmp_path):
    source = tmp_path / "input"
    (source / "behavior").mkdir(parents=True)
    (source / "canonical").mkdir()
    (source / "canonical" / "source_manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="partial Kaggle raw source"):
        prepare_kaggle_data_root(source, tmp_path / "working")
