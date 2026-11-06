from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def ensure_checkpoint(
    domain: str,
    checkpoint_path: str | Path,
    train_if_missing: bool = False,
    train_command: Callable[[], object] | None = None,
) -> Path:
    """Return a checkpoint path, optionally creating it through a training callback."""

    path = Path(checkpoint_path)
    if path.exists():
        return path
    if not train_if_missing:
        raise FileNotFoundError(
            f"Missing {domain} checkpoint at {path}. Run training first or pass --train-if-missing."
        )
    if train_command is None:
        raise ValueError("train_command is required when train_if_missing=True")

    train_command()
    if not path.exists():
        raise FileNotFoundError(
            f"Training completed but {domain} checkpoint was not created at {path}."
        )
    return path
