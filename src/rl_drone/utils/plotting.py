"""Plotting utilities for analysing training artifacts."""

from __future__ import annotations

import csv
import glob
import os
from typing import Sequence

import numpy as np


def _find_tb_run_dir(tensorboard_dir: str) -> str | None:
    """Return the directory that actually holds TB event files.

    Stable-Baselines3 nests its event files under an auto-named subdirectory
    like ``SAC_1/`` inside the ``tensorboard_log`` path. This helper walks
    one level down and picks the most recently modified subdir containing
    ``events.out.tfevents.*`` files, or returns *tensorboard_dir* itself if
    it already contains events.
    """
    if not os.path.isdir(tensorboard_dir):
        return None

    direct = glob.glob(os.path.join(tensorboard_dir, "events.out.tfevents.*"))
    if direct:
        return tensorboard_dir

    subdirs = []
    for entry in os.listdir(tensorboard_dir):
        sub = os.path.join(tensorboard_dir, entry)
        if os.path.isdir(sub) and glob.glob(
            os.path.join(sub, "events.out.tfevents.*")
        ):
            subdirs.append(sub)
    if not subdirs:
        return None
    return max(subdirs, key=os.path.getmtime)


def _resolve_eval_files(runs: "Sequence[str] | str") -> list[tuple[str, str]]:
    """Normalize a ``runs`` argument to ``[(label, evaluations.npz), ...]``.

    Accepts either a parent directory (scanned for ``<run>/evaluations.npz``)
    or an explicit sequence of run directories / npz paths.
    """
    pairs: list[tuple[str, str]] = []
    if isinstance(runs, str):
        for path in sorted(glob.glob(os.path.join(runs, "*", "evaluations.npz"))):
            pairs.append((os.path.basename(os.path.dirname(path)), path))
        return pairs

    for entry in runs:
        if os.path.isdir(entry):
            npz = os.path.join(entry, "evaluations.npz")
            if os.path.exists(npz):
                pairs.append((os.path.basename(entry.rstrip(os.sep)), npz))
        elif os.path.exists(entry):
            label = os.path.basename(os.path.dirname(entry)) or entry
            pairs.append((label, entry))
    return pairs


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


def plot_training_losses(
    tensorboard_dir: str,
    *,
    scalars: Sequence[str] | None = None,
    smoothing_window: int = 10,
    title: str = "SAC Training Losses",
    save_path: str | None = None,
    show: bool = False,
):
    """Plot SB3 training-loss scalars from TensorBoard event files.

    Reads the TB event files written under *tensorboard_dir* by
    Stable-Baselines3 and renders one subplot per requested scalar. The
    default scalar set covers SAC's canonical diagnostics:

    * ``train/actor_loss`` — policy gradient loss.
    * ``train/critic_loss`` — Q-function MSE loss.
    * ``train/ent_coef`` — auto-tuned entropy coefficient (exploration
      pressure).

    Divergence, NaNs, or a collapsing entropy coefficient in these curves
    are the earliest signals of a misbehaving SAC run — usually long
    before the reward curve makes the problem obvious.

    Args:
        tensorboard_dir: TB log directory — typically
            ``paths.tensorboard_dir``. May point at the root dir or an
            SB3-nested subdir like ``.../tensorboard/SAC_1/``.
        scalars: TB scalar tag names to plot. Missing tags are silently
            skipped. Defaults to the SAC loss triple.
        smoothing_window: Rolling-mean window (in recorded update steps)
            for the overlaid smoothed curve. Set to ``1`` to disable.
        title: Figure title.
        save_path: If given, save the figure to this path.
        show: If *True*, call ``plt.show()``.

    Returns:
        The matplotlib ``Figure``, or ``None`` if no TB events were found
        or none of the requested scalars were present.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as exc:  # pragma: no cover - dependency is in [train]
        raise ImportError(
            "plot_training_losses requires the 'tensorboard' package. "
            "Install with: pip install 'rl-drone[train]'"
        ) from exc

    if scalars is None:
        scalars = ("train/actor_loss", "train/critic_loss", "train/ent_coef")

    run_dir = _find_tb_run_dir(tensorboard_dir)
    if run_dir is None:
        return None

    # size_guidance={"scalars": 0} loads every sample (no downsampling).
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))

    series: list[tuple[str, np.ndarray, np.ndarray]] = []
    for tag in scalars:
        if tag not in available:
            continue
        events = ea.Scalars(tag)
        if not events:
            continue
        steps = np.array([e.step for e in events], dtype=float)
        values = np.array([e.value for e in events], dtype=float)
        series.append((tag, steps, values))

    if not series:
        return None

    import matplotlib
    if save_path and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(series), 1, figsize=(10, 3.2 * len(series)), sharex=True
    )
    if len(series) == 1:
        axes = [axes]

    window = max(1, int(smoothing_window))
    for ax, (tag, steps, values) in zip(axes, series):
        ax.plot(steps, values, alpha=0.35, color="tab:blue", linewidth=0.9)
        if window > 1 and values.size >= window:
            kernel = np.ones(window) / window
            smoothed = np.convolve(values, kernel, mode="valid")
            ax.plot(
                steps[window - 1 :],
                smoothed,
                color="tab:blue",
                linewidth=2.0,
                label=f"Rolling mean (w={window})",
            )
            ax.legend(loc="best", fontsize="small")
        ax.set_ylabel(tag)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Training timesteps")
    fig.suptitle(title)
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


def plot_cross_run_comparison(
    runs: "Sequence[str] | str",
    *,
    labels: Sequence[str] | None = None,
    smoothing_window: int = 1,
    show_std: bool = False,
    title: str = "Cross-Run Reward Comparison",
    save_path: str | None = None,
    show: bool = False,
):
    """Overlay evaluation reward curves from multiple training runs.

    Useful for hyperparameter sweeps and regression detection — this is
    the chart you reach for as soon as you have more than one run to
    compare. Each run contributes one line on a shared timesteps x-axis
    so you can spot which run is winning and by how much.

    Args:
        runs: Either a parent directory (scanned for
            ``<subdir>/evaluations.npz``) or a sequence of run directories
            / explicit ``evaluations.npz`` paths. Passing a ``RunPaths``
            ``base_dir`` auto-discovers every run recorded for that
            environment + algorithm combo.
        labels: Optional labels in the same order as *runs*. Defaults to
            each run directory's basename (e.g. the timestamp).
        smoothing_window: Rolling mean window in eval checkpoints. ``1``
            disables smoothing.
        show_std: If *True*, overlay a translucent ±1 std band per run.
        title: Figure title.
        save_path: If given, save the figure to this path.
        show: If *True*, call ``plt.show()``.

    Returns:
        The matplotlib ``Figure``, or ``None`` if no runs were found.
    """
    pairs = _resolve_eval_files(runs)
    if not pairs:
        return None

    if labels is not None:
        labels = list(labels)
        if len(labels) != len(pairs):
            raise ValueError(
                f"labels has length {len(labels)} but found {len(pairs)} runs"
            )
        pairs = [(lbl, path) for lbl, (_, path) in zip(labels, pairs)]

    import matplotlib
    if save_path and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0
    for label, eval_file in pairs:
        try:
            with np.load(eval_file) as data:
                timesteps = np.asarray(data["timesteps"], dtype=float)
                results = np.asarray(data["results"], dtype=float)
        except (FileNotFoundError, KeyError, OSError):
            continue
        if timesteps.size == 0 or results.size == 0:
            continue

        mean_r = np.mean(results, axis=1)
        std_r = np.std(results, axis=1)

        window = max(1, int(smoothing_window))
        if window > 1 and mean_r.size >= window:
            kernel = np.ones(window) / window
            mean_r = np.convolve(mean_r, kernel, mode="valid")
            std_r = np.convolve(std_r, kernel, mode="valid")
            timesteps = timesteps[window - 1 :]

        line, = ax.plot(timesteps, mean_r, label=label, linewidth=1.8)
        if show_std:
            ax.fill_between(
                timesteps,
                mean_r - std_r,
                mean_r + std_r,
                alpha=0.15,
                color=line.get_color(),
            )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean evaluation reward")
    ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
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
