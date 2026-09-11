"""Resolve the explicitly allowed behavior-tree checkpoints."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from alignment_schema import (
    ALLOWED_CHECKPOINTS,
    FORBIDDEN_CHECKPOINT_MARKERS,
    REQUIRED_CHECKPOINT_FILES,
)


def resolve_checkpoint_path(raw: str) -> str:
    path = os.path.abspath(os.path.expanduser(str(raw or "").strip()))
    allowed = tuple(os.path.normpath(item) for item in ALLOWED_CHECKPOINTS)
    lowered = path.replace("\\", "/").lower()
    for marker in FORBIDDEN_CHECKPOINT_MARKERS:
        if marker.lower() in lowered:
            raise ValueError(
                f"Refusing checkpoint path {path!r}: contains forbidden marker {marker!r}. "
                f"Allowed paths are {list(ALLOWED_CHECKPOINTS)}"
            )
    if not path.endswith("/pretrained_model"):
        raise ValueError(f"Checkpoint must be a pretrained_model directory, got {path}")
    if os.path.normpath(path) not in allowed:
        raise ValueError(
            f"Checkpoint path is not approved: resolved={path}, "
            f"allowed={list(ALLOWED_CHECKPOINTS)}"
        )
    return path


def inspect_checkpoint_files(checkpoint_dir: str) -> Dict[str, Any]:
    missing: List[str] = []
    present: List[str] = []
    for name in REQUIRED_CHECKPOINT_FILES:
        full = os.path.join(checkpoint_dir, name)
        if os.path.isfile(full):
            present.append(os.path.abspath(full))
        else:
            missing.append(full)
    local_readable = not missing
    return {
        "checkpoint_dir": os.path.abspath(checkpoint_dir),
        "required_files": list(REQUIRED_CHECKPOINT_FILES),
        "present": present,
        "missing": missing,
        "local_readable": local_readable,
        "load_mode": "local" if local_readable else "remote_pinned",
    }


def format_checkpoint_report(info: Dict[str, Any]) -> str:
    lines = [
        f"resolved_checkpoint={info['checkpoint_dir']}",
        f"load_mode={info['load_mode']}",
        "required_files:",
    ]
    for name in info["required_files"]:
        full = os.path.join(info["checkpoint_dir"], name)
        status = "PRESENT" if full in info["present"] else "MISSING_LOCALLY"
        lines.append(f"  {status} {full}")
    return "\n".join(lines)
