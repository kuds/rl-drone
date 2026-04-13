"""Post-training stage summary report writer.

Produces a human-readable ``stage_summary.txt`` next to other run artifacts
that captures the headline training outcome in a single glance: when the run
happened, how long it took, how many steps were trained, what the final and
best evaluation rewards were, and a fresh post-training rollout on the best
model.

Example output::

    BitCrazy: DroneHover Summary
    ==================================================

    Project:        BitCrazy
    Environment:    DroneHover
    Description:    SAC training run for drone hover
    Algorithm:      SAC
    Date:           2026-04-13 14:30:00
    Timesteps:      1,500,000
    Duration:       2h 15m 30s
    Final eval:     -17.07 +/- 4.94
    Avg ep length:  89.3 +/- 13.8 steps (0.89s sim time)
    Best eval:      -5.23 +/- 3.14 (at 1,200,000 steps)

    Best Model Evaluation (30 episodes)
    ----------------------------------------
      Reward:       -17.07 +/- 4.94
      Ep length:    89.3 +/- 13.8 steps (0.89s sim time)

The ``Avg fwd vel`` / ``Fwd vel`` lines are only included when the
environment populates ``info["drone_speed"]`` (currently the racer env).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Mapping, Optional

import numpy as np


_LABEL_WIDTH = 16
_HEADER_RULE = "=" * 50
_SUB_RULE = "-" * 40


def format_duration(delta: timedelta) -> str:
    """Format a ``timedelta`` as ``"Xh Ym Zs"`` (or ``"Ym Zs"``)."""
    total_seconds = max(0, int(delta.total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"


def _format_mean_std(mean: float, std: float, precision: int = 2) -> str:
    return f"{mean:.{precision}f} +/- {std:.{precision}f}"


def _label_line(label: str, value: str) -> str:
    return f"{(label + ':').ljust(_LABEL_WIDTH)}{value}"


def _step_dt_from_env(env: Any) -> Optional[float]:
    """Return the simulator wall-clock seconds elapsed per ``env.step``.

    Tries, in order: ``env.metadata['render_fps']`` (any env), raw MuJoCo
    attributes ``env.frame_skip * env.model.opt.timestep`` (MujocoEnv), and
    the same path through a VecEnv's ``get_attr``.
    """
    metadata = getattr(env, "metadata", None)
    if isinstance(metadata, Mapping):
        fps = metadata.get("render_fps")
        if fps:
            try:
                return 1.0 / float(fps)
            except (TypeError, ValueError, ZeroDivisionError):
                pass

    for attr_path in (("frame_skip", "model"), ):
        try:
            frame_skip = getattr(env, attr_path[0])
            mj_model = getattr(env, attr_path[1])
            return float(frame_skip) * float(mj_model.opt.timestep)
        except AttributeError:
            pass

    get_attr = getattr(env, "get_attr", None)
    if callable(get_attr):
        try:
            frame_skips = get_attr("frame_skip")
            mj_models = get_attr("model")
            if frame_skips and mj_models:
                return float(frame_skips[0]) * float(mj_models[0].opt.timestep)
        except Exception:
            pass

    return None


def _format_ep_length(mean: float, std: float, step_dt: Optional[float]) -> str:
    base = f"{mean:.1f} +/- {std:.1f} steps"
    if step_dt is None or step_dt <= 0:
        return base
    return f"{base} ({mean * step_dt:.2f}s sim time)"


def read_eval_history(eval_file: str) -> Optional[dict]:
    """Load ``evaluations.npz`` and return final / best eval summaries.

    Returns ``None`` if the file does not exist, is empty, or malformed.
    """
    if not eval_file or not os.path.exists(eval_file):
        return None
    try:
        with np.load(eval_file) as data:
            if not data.files:
                return None
            timesteps = np.asarray(data["timesteps"])
            results = np.asarray(data["results"])
            ep_lengths = np.asarray(data["ep_lengths"])
    except (OSError, KeyError, ValueError):
        return None

    if timesteps.size == 0 or results.size == 0:
        return None

    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)
    mean_lengths = ep_lengths.mean(axis=1)
    std_lengths = ep_lengths.std(axis=1)

    final_idx = int(timesteps.size - 1)
    best_idx = int(np.argmax(mean_rewards))

    return {
        "n_evals": int(timesteps.size),
        "final_step": int(timesteps[final_idx]),
        "final_reward_mean": float(mean_rewards[final_idx]),
        "final_reward_std": float(std_rewards[final_idx]),
        "final_length_mean": float(mean_lengths[final_idx]),
        "final_length_std": float(std_lengths[final_idx]),
        "best_step": int(timesteps[best_idx]),
        "best_reward_mean": float(mean_rewards[best_idx]),
        "best_reward_std": float(std_rewards[best_idx]),
    }


def run_best_model_evaluation(
    model: Any,
    env: Any,
    n_episodes: int = 30,
    deterministic: bool = True,
) -> dict:
    """Run ``n_episodes`` rollouts and return aggregate reward/length/speed.

    Works with a single-env SB3 VecEnv (``n_envs=1``) which is what the
    notebooks pass in after loading the best model. If the environment's
    ``info`` dict populates ``"drone_speed"``, the rollout also returns
    per-episode mean speeds; otherwise ``speeds`` is an empty array.
    """
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_mean_speeds: list[float] = []

    for _ in range(n_episodes):
        obs = env.reset()
        total_reward = 0.0
        step_count = 0
        step_speeds: list[float] = []
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            total_reward += float(np.asarray(reward).item())
            step_count += 1

            step_info = info[0] if isinstance(info, (list, tuple)) else info
            if isinstance(step_info, Mapping) and "drone_speed" in step_info:
                try:
                    step_speeds.append(float(step_info["drone_speed"]))
                except (TypeError, ValueError):
                    pass

            done_flag = np.asarray(done).item() if hasattr(done, "__len__") else bool(done)
            if done_flag:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        if step_speeds:
            episode_mean_speeds.append(float(np.mean(step_speeds)))

    return {
        "rewards": np.asarray(episode_rewards, dtype=np.float64),
        "lengths": np.asarray(episode_lengths, dtype=np.int64),
        "speeds": np.asarray(episode_mean_speeds, dtype=np.float64),
        "n_episodes": int(n_episodes),
    }


def format_stage_summary(
    *,
    project_name: str,
    env_str: str,
    algorithm: str,
    total_timesteps: int,
    training_start: datetime,
    training_end: datetime,
    rollout: dict,
    eval_history: Optional[dict] = None,
    description: str = "",
    step_dt: Optional[float] = None,
    title: Optional[str] = None,
) -> str:
    """Build the plain-text stage summary string."""
    title = title or f"{project_name}: {env_str} Summary"
    lines: list[str] = [title, _HEADER_RULE, ""]

    lines.append(_label_line("Project", project_name))
    lines.append(_label_line("Environment", env_str))
    if description:
        lines.append(_label_line("Description", description))
    lines.append(_label_line("Algorithm", algorithm))
    lines.append(_label_line("Date", training_end.strftime("%Y-%m-%d %H:%M:%S")))
    lines.append(_label_line("Timesteps", f"{int(total_timesteps):,}"))
    lines.append(_label_line("Duration", format_duration(training_end - training_start)))

    if eval_history is not None:
        lines.append(
            _label_line(
                "Final eval",
                _format_mean_std(
                    eval_history["final_reward_mean"],
                    eval_history["final_reward_std"],
                ),
            )
        )
        lines.append(
            _label_line(
                "Avg ep length",
                _format_ep_length(
                    eval_history["final_length_mean"],
                    eval_history["final_length_std"],
                    step_dt,
                ),
            )
        )

    rollout_speeds = np.asarray(rollout.get("speeds", np.asarray([])))
    if rollout_speeds.size > 0:
        lines.append(
            _label_line(
                "Avg fwd vel",
                f"{rollout_speeds.mean():.2f} +/- {rollout_speeds.std():.2f} m/s",
            )
        )

    if eval_history is not None:
        lines.append(
            _label_line(
                "Best eval",
                f"{_format_mean_std(eval_history['best_reward_mean'], eval_history['best_reward_std'])}"
                f" (at {eval_history['best_step']:,} steps)",
            )
        )

    rollout_rewards = np.asarray(rollout["rewards"])
    rollout_lengths = np.asarray(rollout["lengths"])
    n_episodes = int(rollout.get("n_episodes", rollout_rewards.size))

    lines.append("")
    lines.append(f"Best Model Evaluation ({n_episodes} episodes)")
    lines.append(_SUB_RULE)
    lines.append(
        "  "
        + _label_line(
            "Reward",
            _format_mean_std(float(rollout_rewards.mean()), float(rollout_rewards.std())),
        )
    )
    lines.append(
        "  "
        + _label_line(
            "Ep length",
            _format_ep_length(
                float(rollout_lengths.mean()),
                float(rollout_lengths.std()),
                step_dt,
            ),
        )
    )
    if rollout_speeds.size > 0:
        lines.append(
            "  "
            + _label_line(
                "Fwd vel",
                f"{rollout_speeds.mean():.3f} +/- {rollout_speeds.std():.3f} m/s",
            )
        )

    return "\n".join(lines) + "\n"


def write_stage_summary(
    *,
    save_dir: str,
    model: Any,
    eval_env: Any,
    project_name: str,
    env_str: str,
    algorithm: str,
    total_timesteps: int,
    training_start: datetime,
    training_end: datetime,
    description: str = "",
    eval_file: Optional[str] = None,
    n_eval_episodes: int = 30,
    file_name: str = "stage_summary.txt",
    title: Optional[str] = None,
    step_dt: Optional[float] = None,
    verbose: bool = True,
) -> str:
    """Run a fresh best-model rollout and write the stage summary to disk.

    Args:
        save_dir: Directory the summary file is written to. Typically
            ``paths.run_dir`` so it sits alongside ``best_model.zip`` and
            ``evaluations.npz``.
        model: A loaded SB3 model (usually the best model).
        eval_env: Vectorized evaluation environment (``n_envs=1``).
        project_name: Top-level project name (e.g. ``"BitCrazy"``).
        env_str: Environment identifier (e.g. ``"DroneHover"``).
        algorithm: Algorithm name (e.g. ``"SAC"``).
        total_timesteps: Total timesteps passed to ``model.learn``.
        training_start: Wall-clock time just before ``model.learn`` was called.
        training_end: Wall-clock time just after ``model.learn`` returned.
        description: Free-form run description written to the header.
        eval_file: Path to ``evaluations.npz``. When provided, the summary
            includes ``Final eval`` / ``Best eval`` / ``Avg ep length`` lines
            sourced from the eval-callback history.
        n_eval_episodes: Episodes to run for the ``Best Model Evaluation``
            section.
        file_name: Summary filename. Defaults to ``stage_summary.txt``.
        title: Optional override for the title line.
        step_dt: Optional override for per-step sim time, in seconds. When
            omitted, the value is inferred from ``eval_env``.
        verbose: Whether to print the summary to stdout after writing.

    Returns:
        Absolute path to the written summary file.
    """
    rollout = run_best_model_evaluation(model, eval_env, n_episodes=n_eval_episodes)
    eval_history = read_eval_history(eval_file) if eval_file else None
    effective_step_dt = step_dt if step_dt is not None else _step_dt_from_env(eval_env)

    text = format_stage_summary(
        project_name=project_name,
        env_str=env_str,
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        training_start=training_start,
        training_end=training_end,
        rollout=rollout,
        eval_history=eval_history,
        description=description,
        step_dt=effective_step_dt,
        title=title,
    )

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, file_name)
    with open(path, "w") as f:
        f.write(text)

    if verbose:
        print(f"Stage summary saved to {path}")
        print()
        print(text)

    return path
