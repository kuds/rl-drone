"""Utilities for capturing code version info with each training run."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _run_git_raw(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _run_git(args: list[str], cwd: Path) -> str | None:
    out = _run_git_raw(args, cwd)
    return None if out is None else out.strip()


def get_git_version_info(repo_path: str | Path | None = None) -> dict[str, Any]:
    """Return git state for the repo so training runs can be reproduced.

    The returned dict contains ``commit`` (full SHA), ``short_commit``,
    ``branch``, ``dirty`` (True if the working tree has uncommitted changes),
    ``run_id`` (short SHA with a ``-dirty`` suffix when applicable), and,
    when dirty, ``dirty_files`` listing modified paths.  When git is not
    available or ``repo_path`` is not inside a repository, returns a dict
    with ``available`` set to False.
    """
    cwd = Path(repo_path) if repo_path is not None else Path.cwd()

    commit = _run_git(["rev-parse", "HEAD"], cwd)
    if commit is None:
        return {"available": False}

    # Leading spaces in porcelain status are significant — don't strip.
    status_raw = _run_git_raw(["status", "--porcelain"], cwd) or ""
    status_lines = [line for line in status_raw.splitlines() if line]
    dirty = bool(status_lines)
    short_commit = commit[:8]
    run_id = f"{short_commit}-dirty" if dirty else short_commit

    info: dict[str, Any] = {
        "available": True,
        "commit": commit,
        "short_commit": short_commit,
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd),
        "dirty": dirty,
        "run_id": run_id,
    }
    if dirty:
        info["dirty_files"] = [line[3:] for line in status_lines]
    return info
