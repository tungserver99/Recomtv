"""Mount Kaggle source snapshots read-only under a writable data root."""

from pathlib import Path


RAW_DIRS = ("behavior", "duration", "mytv_vmp_content", "tags", "content_tags")
RAW_FILES = ("month_mytv_info.parquet",)
PREPARED_DIRS = ("canonical", "model")
WRITABLE_DIRS = ("canonical", "metrics", "model")


def prepare_kaggle_data_root(
    input_data_root: Path, working_data_root: Path
) -> dict[str, str]:
    """Link an approved source snapshot into a writable Kaggle data root.

    A source must be either a complete raw snapshot or a prepared canonical
    snapshot carrying ``canonical/source_manifest.json``. Partial raw inputs
    are rejected even if a canonical directory is also present.
    """
    source = Path(input_data_root).resolve()
    target = Path(working_data_root)
    if not source.is_dir():
        raise FileNotFoundError(f"Kaggle input data root does not exist: {source}")

    raw_names = (*RAW_DIRS, *RAW_FILES)
    present = {name: (source / name).exists() for name in raw_names}
    has_any_raw = any(present.values())
    has_all_raw = all(present.values())
    has_prepared = (source / "canonical" / "source_manifest.json").is_file()

    if has_any_raw and not has_all_raw:
        raise ValueError("partial Kaggle raw source; provide every required raw input")
    if not has_all_raw and not has_prepared:
        raise FileNotFoundError(
            "Kaggle raw source or canonical/source_manifest.json required"
        )

    link_names = tuple(
        name
        for name in (*raw_names, *PREPARED_DIRS)
        if (source / name).exists()
    )

    # Validate the complete target layout before creating any links so a late
    # collision cannot leave behind a partially mounted source snapshot.
    for name in link_names:
        link = target / name
        source_path = (source / name).resolve()
        if link.is_symlink() and link.resolve() == source_path:
            continue
        if link.exists() or link.is_symlink():
            raise FileExistsError(f"conflicting Kaggle target: {name}")

    target.mkdir(parents=True, exist_ok=True)
    for name in link_names:
        link = target / name
        if not link.is_symlink():
            link.symlink_to(
                source / name,
                target_is_directory=name in (*RAW_DIRS, *PREPARED_DIRS),
            )

    for name in WRITABLE_DIRS:
        directory = target / name
        if directory.is_symlink():
            continue
        directory.mkdir(exist_ok=True)

    return {name: str((source / name).resolve()) for name in link_names}
