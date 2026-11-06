"""Small helpers shared by more than one pipeline stage, kept here so the
stages can be separate modules without importing each other."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pandas as pd


def atomic_write_text(path: str | Path, text: str) -> None:
    """Replace ``path`` so a reader sees the old file or the new one, never a
    partial write. The stage file sits beside the target because a rename is
    only atomic within one filesystem."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_name(f".{target.name}.{uuid.uuid4().hex}.stage")
    with stage.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(stage, target)


def _rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            get_process = ctypes.windll.kernel32.GetCurrentProcess
            get_process.restype = wintypes.HANDLE
            handle = get_process()
            get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
            get_memory.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            get_memory.restype = wintypes.BOOL
            ok = get_memory(
                handle, ctypes.byref(counters), counters.cb
            )
            return int(counters.WorkingSetSize) if ok else 0
        except (AttributeError, OSError):
            return 0


def _partition_serving_interaction_scopes(
    loaded: pd.DataFrame,
    serving_content_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Keep truth-audit rows from influencing serving-time retrieval features."""
    if "content_id" not in loaded:
        raise ValueError("interaction scope partition requires content_id")
    serving_ids = set(map(str, serving_content_ids))
    normalized_ids = loaded["content_id"].astype(str)
    serving = loaded.loc[normalized_ids.isin(serving_ids)].copy().reset_index(
        drop=True
    )
    audit = loaded.reset_index(drop=True)
    audit_only_mask = ~normalized_ids.isin(serving_ids)
    metrics = {
        "loaded_interaction_rows": int(len(audit)),
        "serving_interaction_rows": int(len(serving)),
        "truth_audit_only_interaction_rows": int(audit_only_mask.sum()),
        "loaded_interaction_contents": int(normalized_ids.nunique()),
        "serving_interaction_contents": int(
            serving["content_id"].astype(str).nunique()
        ),
        "truth_audit_only_interaction_contents": int(
            normalized_ids.loc[audit_only_mask].nunique()
        ),
    }
    return serving, audit, metrics
