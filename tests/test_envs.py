"""Tests for drone environments.

These tests verify the environment interfaces without requiring GPU rendering.
They check observation/action space shapes, reset behavior, and step outputs.
"""

import numpy as np
import pytest

try:
    from rl_drone.utils.model_xml import setup_mujoco_model
    setup_mujoco_model(sphere_size=0.25, target_height=1.0)
    from rl_drone.envs.drone_hover import DroneHoverEnv
    HAS_MUJOCO = True
except (ImportError, AttributeError, OSError):
    HAS_MUJOCO = False

pytestmark = pytest.mark.skipif(not HAS_MUJOCO, reason="MuJoCo not available")


class TestDroneHoverEnv:
    @pytest.fixture
    def env(self):
        env = DroneHoverEnv(sphere_size=0.25, episode_len=100)
        yield env
        env.close()

    def test_observation_space_shape(self, env):
        assert env.observation_space.shape == (19,)

    def test_action_space_shape(self, env):
        assert env.action_space.shape == (4,)

    def test_action_space_bounds(self, env):
        # Symmetric, normalized action space as recommended by SB3.
        for i in range(4):
            assert env.action_space.low[i] == -1.0
            assert env.action_space.high[i] == 1.0

    def test_action_rescaling(self, env):
        # The raw motor ranges are recovered by rescaling the normalized action.
        np.testing.assert_allclose(
            env._rescale_action(np.array([-1.0, -1.0, -1.0, -1.0])),
            np.array([0.0, -1.0, -1.0, -1.0]),
        )
        np.testing.assert_allclose(
            env._rescale_action(np.array([1.0, 1.0, 1.0, 1.0])),
            np.array([0.35, 1.0, 1.0, 1.0]),
        )
        np.testing.assert_allclose(
            env._rescale_action(np.array([0.0, 0.0, 0.0, 0.0])),
            np.array([0.175, 0.0, 0.0, 0.0]),
        )

    def test_reset_returns_correct_shape(self, env):
        obs, info = env.reset()
        assert obs.shape == (19,)
        assert isinstance(info, dict)

    def test_step_returns_correct_types(self, env):
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (19,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert "distance_to_target" in info
        assert "touch_reported" in info
        assert "sensor_reading" in info
        assert "drone_speed" in info
        assert info["drone_speed"] >= 0.0

    def test_observation_dtype(self, env):
        obs, _ = env.reset()
        assert obs.dtype == np.float64

    def test_step_increments(self, env):
        env.reset()
        assert env.step_number == 0
        env.step(env.action_space.sample())
        assert env.step_number == 1

    def test_reset_clears_state(self, env):
        env.reset()
        for _ in range(5):
            env.step(env.action_space.sample())
        env.reset()
        assert env.step_number == 0
        assert env.total_reward == 0

    def test_out_of_bounds_terminates(self, env):
        env.reset()
        # Force drone far away
        env.data.qpos[:3] = [100.0, 100.0, 100.0]
        _, reward, terminated, _, _ = env.step(env.action_space.sample())
        assert terminated
        assert reward == env.out_of_bounds_penalty

    def test_below_ground_terminates(self, env):
        env.reset()
        env.data.qpos[2] = 0.01  # Below ground threshold
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        assert terminated
