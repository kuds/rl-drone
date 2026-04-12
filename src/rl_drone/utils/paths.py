"""Filesystem layout for training artifacts.

Every training notebook writes artifacts into the same three-level tree so
results are easy to locate, compare, and sync:

    <base>/training jobs/<env_str>/<rl_type>/
    ├── tensorboard/              # shared across runs of this (env, algo)
    └── <timestamp>/              # one directory per training run
        ├── best_model.zip
        ├── final_model.zip
        ├── config.json
        ├── evaluations.npz
        ├── monitor/
        ├── checkpoints/
        ├── plots/
        └── videos/

Notebooks should call :func:`build_run_paths` in their setup cell and then
reference the returned :class:`RunPaths` fields throughout the notebook so
that path conventions stay consistent across environments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"


@dataclass(frozen=True)
class RunPaths:
    """Resolved filesystem paths for a single training run.

    Attributes:
        base_dir: ``<parent>/training jobs/<env_str>/<rl_type>/``. Contains
            ``tensorboard/`` and one subdirectory per run.
        run_dir: ``<base_dir>/<timestamp>/``. Root of a single run's artifacts.
        tensorboard_dir: ``<base_dir>/tensorboard/``. Shared across all runs
            of the same ``(env_str, rl_type)`` so TensorBoard can compare them.
        monitor_dir: ``<run_dir>/monitor/``. Stable-Baselines3 ``Monitor``
            CSV logs.
        checkpoints_dir: ``<run_dir>/checkpoints/``. Intermediate model
            snapshots from ``CheckpointCallback``.
        plots_dir: ``<run_dir>/plots/``. PNGs from
            ``TrainingPlotsCallback`` and post-training plots.
        videos_dir: ``<run_dir>/videos/``. MP4 rollouts and per-step CSV logs
            from ``VideoRecordCallback``.
    """

    base_dir: str
    run_dir: str
    tensorboard_dir: str
    monitor_dir: str
    checkpoints_dir: str
    plots_dir: str
    videos_dir: str

    def makedirs(self) -> None:
        """Create every directory in the layout if it does not already exist."""
        for path in (
            self.base_dir,
            self.run_dir,
            self.tensorboard_dir,
            self.monitor_dir,
            self.checkpoints_dir,
            self.plots_dir,
            self.videos_dir,
        ):
            os.makedirs(path, exist_ok=True)


def default_timestamp() -> str:
    """Return a filesystem-safe timestamp string (no ``:`` or spaces)."""
    return datetime.now().strftime(TIMESTAMP_FORMAT)


def build_run_paths(
    project_name: str,
    env_str: str,
    rl_type: str,
    parent_path: Optional[str] = None,
    use_google_drive: bool = True,
    timestamp: Optional[str] = None,
) -> RunPaths:
    """Build the artifact directory layout for a single training run.

    Args:
        project_name: Top-level project folder (e.g. ``"BitCrazy"``). Only
            used when ``use_google_drive`` is ``True``.
        env_str: Environment identifier, e.g. ``"DroneHover"``.
        rl_type: Algorithm identifier, e.g. ``"SAC"``.
        parent_path: Mount point for Google Drive (e.g. ``"/content/gdrive"``)
            when ``use_google_drive`` is ``True``. Ignored otherwise.
        use_google_drive: If ``True``, build paths under
            ``<parent_path>/MyDrive/Finding Theta/<project>/training jobs/``.
            If ``False``, build paths under ``/content/training jobs/``.
        timestamp: Optional override for the run directory name. Defaults to
            :func:`default_timestamp`.

    Returns:
        A :class:`RunPaths` instance. Directories are *not* created; call
        :meth:`RunPaths.makedirs` to materialize them.
    """
    if use_google_drive:
        if parent_path is None:
            raise ValueError(
                "parent_path must be provided when use_google_drive=True"
            )
        base_dir = os.path.join(
            parent_path,
            "MyDrive",
            "Finding Theta",
            project_name,
            "training jobs",
            env_str,
            rl_type,
        )
    else:
        base_dir = os.path.join("/content", "training jobs", env_str, rl_type)

    ts = timestamp or default_timestamp()
    run_dir = os.path.join(base_dir, ts)

    return RunPaths(
        base_dir=base_dir,
        run_dir=run_dir,
        tensorboard_dir=os.path.join(base_dir, "tensorboard"),
        monitor_dir=os.path.join(run_dir, "monitor"),
        checkpoints_dir=os.path.join(run_dir, "checkpoints"),
        plots_dir=os.path.join(run_dir, "plots"),
        videos_dir=os.path.join(run_dir, "videos"),
    )
