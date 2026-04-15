"""Plotting utilities for analysing training artifacts."""

from __future__ import annotations

import csv
import glob
import os
from typing import Sequence

import numpy as np


def plot_trajectory_3d(
    csv_path: str,
    *,
    track_points: np.ndarray | None = None,
    title: str = "Drone 3D Trajectory",
    save_path: str | None = None,
    show: bool = False,
):
    """Plot the drone's 3D flight path from a per-step CSV log.

    Args:
        csv_path: Path to a CSV file written by :class:`VideoRecordCallback`.
            Must contain columns ``drone_pos_x``, ``drone_pos_y``,
            ``drone_pos_z``.
        track_points: Optional *(N, 3)* array of track checkpoint positions to
            overlay (useful for the racer environment).
        title: Plot title.
        save_path: If given, save the figure to this path instead of showing.
        show: If *True*, call ``plt.show()`` (ignored when *save_path* is set).
    """
    import matplotlib
    if save_path and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, ys, zs = [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["drone_pos_x"]))
            ys.append(float(row["drone_pos_y"]))
            zs.append(float(row["drone_pos_z"]))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, zs, linewidth=0.8, label="Drone path")
    ax.scatter(xs[0], ys[0], zs[0], color="green", s=60, zorder=5, label="Start")
    ax.scatter(xs[-1], ys[-1], zs[-1], color="red", s=60, zorder=5, label="End")

    if track_points is not None:
        ax.scatter(
            track_points[:, 0],
            track_points[:, 1],
            track_points[:, 2],
            color="orange",
            s=40,
            marker="^",
            label="Checkpoints",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_reward_breakdown(
    csv_path: str,
    *,
    title: str = "Reward Component Breakdown",
    save_path: str | None = None,
    show: bool = False,
):
    """Plot per-step reward components from a CSV log.

    Any column whose name starts with ``reward_`` is treated as a reward
    component and plotted as a stacked area chart.

    Args:
        csv_path: Path to a CSV file written by :class:`VideoRecordCallback`.
        title: Plot title.
        save_path: If given, save the figure to this path.
        show: If *True*, call ``plt.show()``.
    """
    import matplotlib
    if save_path and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps: list[int] = []
    component_data: dict[str, list[float]] = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        reward_cols = [c for c in (reader.fieldnames or []) if c.startswith("reward_")]
        if not reward_cols:
            return None

        for col in reward_cols:
            component_data[col] = []

        for row in reader:
            steps.append(int(row["step"]))
            for col in reward_cols:
                component_data[col].append(float(row[col]))

    fig, ax = plt.subplots(figsize=(10, 5))
    for col in reward_cols:
        label = col.replace("reward_", "").replace("_", " ").title()
        ax.plot(steps, component_data[col], linewidth=0.9, label=label)

    ax.set_xlabel("Step")
    ax.set_ylabel("Reward Component")
    ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_training_reward_over_time(
    monitor_dir: str,
    *,
    smoothing_window: int = 50,
    x_axis: str = "timesteps",
    title: str = "Training Reward over Time",
    save_path: str | None = None,
    show: bool = False,
):
    """Plot per-episode training reward over time from SB3 Monitor CSVs.

    Reads every ``*.monitor.csv`` file written by Stable-Baselines3's
    ``Monitor`` wrapper in *monitor_dir* (one per vectorized worker),
    stitches the per-episode ``(reward, length, wall-time)`` triples back
    into chronological order, and plots the reward signal against the
    chosen x-axis. Compared with :func:`plot_learning_curves` — which only
    samples the evaluation env every ``eval_freq`` steps — this is the
    *dense* training-reward trace containing one point per training
    episode.

    A rolling mean is drawn on top of the raw episode-reward scatter to
    make the learning trend easy to read even when the per-episode signal
    is noisy.

    Args:
        monitor_dir: Directory containing SB3 ``*.monitor.csv`` files. This
            is typically ``paths.monitor_dir`` from :func:`build_run_paths`.
        smoothing_window: Rolling-mean window in episodes. Set to ``1`` to
            disable smoothing.
        x_axis: Either ``"timesteps"`` (cumulative training steps) or
            ``"walltime"`` (hours since run start).
        title: Figure title.
        save_path: If given, save the figure to this path.
        show: If *True*, call ``plt.show()`` (ignored when *save_path* is
            set and *show* is *False*).

    Returns:
        The matplotlib ``Figure``, or ``None`` if no monitor CSVs were
        found or they contained no completed episodes.
    """
    if x_axis not in ("timesteps", "walltime"):
        raise ValueError(
            f"x_axis must be 'timesteps' or 'walltime', got {x_axis!r}"
        )

    csv_files = sorted(glob.glob(os.path.join(monitor_dir, "*.monitor.csv")))
    if not csv_files:
        return None

    episodes: list[tuple[float, float, int]] = []
    for path in csv_files:
        with open(path) as f:
            first_line = f.readline()
            # SB3 Monitor writes a JSON metadata comment as line 1.
            if not first_line.startswith("#"):
                f.seek(0)
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    episodes.append(
                        (float(row["t"]), float(row["r"]), int(row["l"]))
                    )
                except (KeyError, ValueError):
                    # Skip malformed rows rather than crashing the plot.
                    continue

    if not episodes:
        return None

    # Sort episodes chronologically across all parallel workers so the
    # cumulative-step x-axis increases monotonically.
    episodes.sort(key=lambda e: e[0])
    walltimes = np.array([e[0] for e in episodes], dtype=float)
    rewards = np.array([e[1] for e in episodes], dtype=float)
    lengths = np.array([e[2] for e in episodes], dtype=int)
    cum_steps = np.cumsum(lengths)

    if x_axis == "timesteps":
        xs = cum_steps
        xlabel = "Training timesteps"
    else:
        xs = walltimes / 3600.0
        xlabel = "Wall-clock time (hours)"

    import matplotlib
    if save_path and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        xs,
        rewards,
        alpha=0.25,
        color="tab:blue",
        label="Episode reward",
    )

    window = max(1, int(smoothing_window))
    if window > 1 and rewards.size >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(rewards, kernel, mode="valid")
        ax.plot(
            xs[window - 1 :],
            smoothed,
            color="tab:blue",
            linewidth=2.0,
            label=f"Rolling mean (window={window})",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Episode reward")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
        if not show:
            plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_learning_curves(
    eval_file: str,
    *,
    save_dir: str | None = None,
    name_prefix: str = "eval",
    show: bool = False,
) -> list:
    """Generate reward and episode-length learning curves from an npz eval file.

    This is a convenience wrapper for producing the same plots that
    :class:`TrainingPlotsCallback` saves during training, but callable
    post-hoc from a notebook or script.

    Args:
        eval_file: Path to ``evaluations.npz``.
        save_dir: Directory to save PNGs into. If *None*, figures are returned
            without saving.
        name_prefix: Filename prefix.
        show: If *True*, call ``plt.show()``.

    Returns:
        List of matplotlib Figure objects.
    """
    import matplotlib
    if save_dir and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(eval_file) as data:
        timesteps = data["timesteps"]
        results = data["results"]
        ep_lengths = data["ep_lengths"]

    mean_r = np.mean(results, axis=1)
    std_r = np.std(results, axis=1)
    mean_ep = np.mean(ep_lengths, axis=1)
    std_ep = np.std(ep_lengths, axis=1)

    figs = []

    # Reward curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(timesteps, mean_r, label="Mean reward")
    ax.fill_between(timesteps, mean_r - std_r, mean_r + std_r, alpha=0.25)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward")
    ax.set_title("Evaluation Reward over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs.append(fig)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{name_prefix}_reward_curve.png"), dpi=150)

    # Episode length curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(timesteps, mean_ep, label="Mean episode length", color="tab:orange")
    ax.fill_between(
        timesteps, mean_ep - std_ep, mean_ep + std_ep, alpha=0.25, color="tab:orange"
    )
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Length")
    ax.set_title("Episode Length over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    figs.append(fig)

    if save_dir:
        fig.savefig(os.path.join(save_dir, f"{name_prefix}_episode_length.png"), dpi=150)

    if not show:
        for f in figs:
            plt.close(f)

    return figs
