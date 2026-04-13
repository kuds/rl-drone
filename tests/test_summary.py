"""Tests for the stage summary report writer."""

import importlib.util
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Load summary.py directly so the tests do not depend on optional MuJoCo /
# robot_descriptions imports pulled in by ``rl_drone.utils.__init__``.
_SUMMARY_PATH = Path(__file__).resolve().parents[1] / "src" / "rl_drone" / "utils" / "summary.py"
_spec = importlib.util.spec_from_file_location("rl_drone_summary_under_test", _SUMMARY_PATH)
summary = importlib.util.module_from_spec(_spec)
sys.modules["rl_drone_summary_under_test"] = summary
_spec.loader.exec_module(summary)

_step_dt_from_env = summary._step_dt_from_env
format_duration = summary.format_duration
format_stage_summary = summary.format_stage_summary
read_eval_history = summary.read_eval_history
run_best_model_evaluation = summary.run_best_model_evaluation
write_stage_summary = summary.write_stage_summary


class TestFormatDuration:
    def test_minutes_and_seconds(self):
        assert format_duration(timedelta(minutes=22, seconds=24)) == "22m 24s"

    def test_hours_minutes_seconds(self):
        assert format_duration(timedelta(hours=2, minutes=15, seconds=30)) == "2h 15m 30s"

    def test_zero(self):
        assert format_duration(timedelta(0)) == "0m 0s"

    def test_negative_clamped_to_zero(self):
        assert format_duration(timedelta(seconds=-5)) == "0m 0s"

    def test_exactly_one_hour(self):
        assert format_duration(timedelta(hours=1)) == "1h 0m 0s"


class TestReadEvalHistory:
    def test_missing_file(self, tmp_path):
        assert read_eval_history(str(tmp_path / "does_not_exist.npz")) is None

    def test_none_path(self):
        assert read_eval_history("") is None

    def test_happy_path(self, tmp_path):
        path = tmp_path / "evaluations.npz"
        np.savez(
            path,
            timesteps=np.array([100, 200, 300]),
            results=np.array([[-10.0, -12.0], [-5.0, -7.0], [-8.0, -6.0]]),
            ep_lengths=np.array([[50, 60], [80, 70], [90, 85]]),
        )
        out = read_eval_history(str(path))
        assert out is not None
        assert out["n_evals"] == 3
        assert out["final_step"] == 300
        assert out["final_reward_mean"] == pytest.approx(-7.0)
        assert out["best_step"] == 200
        assert out["best_reward_mean"] == pytest.approx(-6.0)
        assert out["final_length_mean"] == pytest.approx(87.5)


class _FakeModel:
    def predict(self, obs, deterministic=True):
        return np.zeros((1, 4), dtype=np.float32), None


class _FakeVecEnv:
    """A tiny SB3-VecEnv look-alike that yields predictable rollouts."""

    def __init__(self, ep_length=5, drone_speed=None, n_resets_to_done=None):
        self.ep_length = ep_length
        self.drone_speed = drone_speed
        self._step = 0

    def reset(self):
        self._step = 0
        return np.zeros((1, 4), dtype=np.float32)

    def step(self, action):
        self._step += 1
        done = self._step >= self.ep_length
        reward = np.array([1.0], dtype=np.float32)
        info = [{}]
        if self.drone_speed is not None:
            info[0]["drone_speed"] = self.drone_speed
        return (
            np.zeros((1, 4), dtype=np.float32),
            reward,
            np.array([done]),
            info,
        )


class _FakeVecNormalize:
    """VecEnv-wrapper look-alike that mimics ``VecNormalize(norm_reward=True)``.

    Each ``step`` emits a reward of ``1.0 / scale`` (the "normalized" value),
    while ``get_original_reward`` returns the un-normalized reward of ``1.0``.
    Used to exercise the VecNormalize-aware branch of
    ``run_best_model_evaluation``.
    """

    def __init__(self, ep_length=5, scale=32.0, norm_reward=True):
        self.ep_length = ep_length
        self.scale = float(scale)
        self.norm_reward = norm_reward
        self._step = 0
        self._last_raw = np.array([1.0], dtype=np.float32)

    def reset(self):
        self._step = 0
        return np.zeros((1, 4), dtype=np.float32)

    def step(self, action):
        self._step += 1
        done = self._step >= self.ep_length
        self._last_raw = np.array([1.0], dtype=np.float32)
        normalized = self._last_raw / self.scale
        return (
            np.zeros((1, 4), dtype=np.float32),
            normalized.astype(np.float32),
            np.array([done]),
            [{}],
        )

    def get_original_reward(self):
        return self._last_raw.copy()


class _FakeVecWrapper:
    """A passthrough VecEnv wrapper that delegates to ``self.venv``.

    Mirrors how ``VecVideoRecorder`` wraps a ``VecNormalize`` — the helper
    ``_find_vec_normalize`` should walk through this wrapper and still find
    the inner VecNormalize.
    """

    def __init__(self, venv):
        self.venv = venv

    def reset(self):
        return self.venv.reset()

    def step(self, action):
        return self.venv.step(action)


class TestRunBestModelEvaluation:
    def test_basic_rollout(self):
        model = _FakeModel()
        env = _FakeVecEnv(ep_length=5)
        result = run_best_model_evaluation(model, env, n_episodes=3)
        assert result["rewards"].shape == (3,)
        assert result["lengths"].shape == (3,)
        assert np.all(result["rewards"] == 5.0)
        assert np.all(result["lengths"] == 5)
        assert result["speeds"].size == 0
        assert result["n_episodes"] == 3

    def test_collects_drone_speed(self):
        model = _FakeModel()
        env = _FakeVecEnv(ep_length=4, drone_speed=0.7)
        result = run_best_model_evaluation(model, env, n_episodes=2)
        assert result["speeds"].shape == (2,)
        assert np.allclose(result["speeds"], 0.7)

    def test_uses_original_reward_when_vecnormalize(self):
        # Emulates a ``VecNormalize(norm_reward=True)`` eval env: step returns
        # ``1.0 / 32`` per step, but ``get_original_reward`` returns ``1.0``.
        # A 5-step episode should accumulate 5.0 (the un-normalized total),
        # not ``5 / 32`` — otherwise the reported reward in ``stage_summary``
        # is in different units from ``Best eval`` / ``evaluate_policy``.
        model = _FakeModel()
        env = _FakeVecNormalize(ep_length=5, scale=32.0)
        result = run_best_model_evaluation(model, env, n_episodes=3)
        assert np.allclose(result["rewards"], 5.0)
        assert np.all(result["lengths"] == 5)

    def test_uses_original_reward_through_wrapper_chain(self):
        # ``VecVideoRecorder``-style wrapper around a VecNormalize: the helper
        # should still unwrap to the inner VecNormalize.
        model = _FakeModel()
        inner = _FakeVecNormalize(ep_length=4, scale=10.0)
        wrapped = _FakeVecWrapper(inner)
        result = run_best_model_evaluation(model, wrapped, n_episodes=2)
        assert np.allclose(result["rewards"], 4.0)

    def test_step_reward_used_when_norm_reward_disabled(self):
        # When VecNormalize has ``norm_reward=False`` the per-step reward is
        # already raw, so the rollout should trust ``env.step``'s reward and
        # not second-guess it via ``get_original_reward``.
        model = _FakeModel()
        env = _FakeVecNormalize(ep_length=5, scale=32.0, norm_reward=False)
        result = run_best_model_evaluation(model, env, n_episodes=2)
        # 5 steps * (1.0 / 32) per step = 0.15625
        assert np.allclose(result["rewards"], 5.0 / 32.0)


class TestStepDtFromEnv:
    def test_from_metadata(self):
        env = SimpleNamespace(metadata={"render_fps": 100})
        assert _step_dt_from_env(env) == pytest.approx(0.01)

    def test_from_mujoco_attrs(self):
        model = SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
        env = SimpleNamespace(frame_skip=5, model=model, metadata={})
        assert _step_dt_from_env(env) == pytest.approx(0.01)

    def test_none_when_unavailable(self):
        env = SimpleNamespace(metadata={})
        assert _step_dt_from_env(env) is None


class TestFormatStageSummary:
    def _rollout(self, with_speed=False):
        return {
            "rewards": np.array([-17.0, -18.0, -16.0]),
            "lengths": np.array([89, 90, 88]),
            "speeds": np.array([0.45, 0.47, 0.46]) if with_speed else np.array([]),
            "n_episodes": 3,
        }

    def _eval_history(self):
        return {
            "n_evals": 4,
            "final_step": 655_360,
            "final_reward_mean": -17.07,
            "final_reward_std": 4.94,
            "final_length_mean": 89.3,
            "final_length_std": 13.8,
            "best_step": 524_288,
            "best_reward_mean": -5.23,
            "best_reward_std": 3.14,
        }

    def test_header_and_core_fields(self):
        text = format_stage_summary(
            project_name="BitCrazy",
            env_str="DroneHover",
            algorithm="SAC",
            total_timesteps=1_500_000,
            training_start=datetime(2026, 4, 13, 12, 0, 0),
            training_end=datetime(2026, 4, 13, 14, 15, 30),
            rollout=self._rollout(),
            eval_history=self._eval_history(),
            description="SAC training run for drone hover",
            step_dt=0.01,
        )
        assert "BitCrazy: DroneHover Summary" in text
        assert "=" * 50 in text
        assert "Project:        BitCrazy" in text
        assert "Environment:    DroneHover" in text
        assert "Description:    SAC training run for drone hover" in text
        assert "Algorithm:      SAC" in text
        assert "Date:           2026-04-13 14:15:30" in text
        assert "Timesteps:      1,500,000" in text
        assert "Duration:       2h 15m 30s" in text
        assert "Final eval:     -17.07 +/- 4.94" in text
        assert "Avg ep length:  89.3 +/- 13.8 steps (0.89s sim time)" in text
        assert "Best eval:      -5.23 +/- 3.14 (at 524,288 steps)" in text
        assert "Best Model Evaluation (3 episodes)" in text
        assert "-" * 40 in text
        assert "Reward:" in text
        assert "Ep length:" in text

    def test_velocity_lines_only_when_available(self):
        no_speed = format_stage_summary(
            project_name="BitCrazy",
            env_str="DroneHover",
            algorithm="SAC",
            total_timesteps=1_000,
            training_start=datetime(2026, 4, 13, 12, 0, 0),
            training_end=datetime(2026, 4, 13, 12, 1, 0),
            rollout=self._rollout(with_speed=False),
            eval_history=self._eval_history(),
            step_dt=0.01,
        )
        assert "Avg speed" not in no_speed
        assert "Speed:" not in no_speed

        with_speed = format_stage_summary(
            project_name="BitCrazy",
            env_str="DroneRacer",
            algorithm="SAC",
            total_timesteps=1_000,
            training_start=datetime(2026, 4, 13, 12, 0, 0),
            training_end=datetime(2026, 4, 13, 12, 1, 0),
            rollout=self._rollout(with_speed=True),
            eval_history=self._eval_history(),
            step_dt=0.01,
        )
        assert "Avg speed:" in with_speed
        assert "Speed:" in with_speed
        assert "m/s" in with_speed

    def test_omits_eval_block_when_history_missing(self):
        text = format_stage_summary(
            project_name="BitCrazy",
            env_str="DroneHover",
            algorithm="SAC",
            total_timesteps=1_000,
            training_start=datetime(2026, 4, 13, 12, 0, 0),
            training_end=datetime(2026, 4, 13, 12, 1, 0),
            rollout=self._rollout(),
            eval_history=None,
            step_dt=0.01,
        )
        assert "Final eval" not in text
        assert "Best eval" not in text
        # Best Model Evaluation section still present.
        assert "Best Model Evaluation" in text

    def test_omits_sim_time_when_step_dt_unknown(self):
        text = format_stage_summary(
            project_name="BitCrazy",
            env_str="DroneHover",
            algorithm="SAC",
            total_timesteps=1_000,
            training_start=datetime(2026, 4, 13, 12, 0, 0),
            training_end=datetime(2026, 4, 13, 12, 1, 0),
            rollout=self._rollout(),
            eval_history=self._eval_history(),
            step_dt=None,
        )
        assert "sim time" not in text


class TestWriteStageSummary:
    def test_writes_file(self, tmp_path):
        model = _FakeModel()
        env = _FakeVecEnv(ep_length=3)

        eval_file = tmp_path / "evaluations.npz"
        np.savez(
            eval_file,
            timesteps=np.array([500, 1_000]),
            results=np.array([[-10.0, -12.0], [-5.0, -7.0]]),
            ep_lengths=np.array([[80, 90], [95, 85]]),
        )

        out_path = write_stage_summary(
            save_dir=str(tmp_path),
            model=model,
            eval_env=env,
            project_name="BitCrazy",
            env_str="DroneHover",
            algorithm="SAC",
            total_timesteps=1_000,
            training_start=datetime(2026, 4, 13, 12, 0, 0),
            training_end=datetime(2026, 4, 13, 12, 30, 0),
            description="unit test run",
            eval_file=str(eval_file),
            n_eval_episodes=2,
            step_dt=0.01,
            verbose=False,
        )

        assert os.path.exists(out_path)
        assert os.path.basename(out_path) == "stage_summary.txt"
        contents = open(out_path).read()
        assert "BitCrazy: DroneHover Summary" in contents
        assert "Duration:       0h 30m 0s" in contents or "Duration:       30m 0s" in contents
        assert "Best Model Evaluation (2 episodes)" in contents
