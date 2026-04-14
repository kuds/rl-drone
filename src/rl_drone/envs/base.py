"""Base MuJoCo environment for CF2 drone tasks.

Centralises the setup that every drone task shares so the task-specific
environments (:class:`~rl_drone.envs.drone_hover.DroneHoverEnv`,
:class:`~rl_drone.envs.drone_racer.DroneRacerEnv`) only implement their
reward shaping and target bookkeeping.

What this base class provides:

* A symmetric, normalized action space :math:`[-1, 1]^4` and the raw CF2
  motor ranges stored on the instance, so subclasses never rewrite the
  rescaling logic. See :meth:`_rescale_action`.
* A 19-dim observation vector :math:`(pos, quat, vec\\_to\\_target,
  lin\\_vel, ang\\_vel, target\\_pos)` built once in :meth:`_get_obs`.
* MuJoCo body / site / sensor handles resolved in ``__init__``.
* Reset noise and episode-bound thresholds as module constants so they
  are easy to tune consistently across tasks.

The class is abstract in spirit: ``step`` is intentionally left to
subclasses, which differ meaningfully between the hover and racer tasks.
"""

from __future__ import annotations

import os

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from robot_descriptions import cf2_mj_description


# ---------------------------------------------------------------------------
# Physical / geometric constants shared by every drone environment.
# ---------------------------------------------------------------------------

#: Raw CF2 motor control ranges. The first actuator is the collective
#: thrust (positive only); the other three are body-axis torques.
ACTION_LOW = np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float32)
ACTION_HIGH = np.array([0.35, 1.0, 1.0, 1.0], dtype=np.float32)

#: Dimensionality of the observation vector produced by :meth:`_get_obs`.
OBS_DIM = 19

#: Drone z-coordinate below which the episode terminates (ground contact).
GROUND_Z_MIN = 0.025

#: Minimum drone z-coordinate after reset noise (keeps the drone in the
#: air so physics does not immediately detect ground contact).
INIT_Z_MIN = 0.075

#: Half-range of the uniform xyz noise applied on ``reset``.
RESET_POS_NOISE = 0.1

#: Half-range of uniform qpos / qvel noise used by :meth:`reset_model`.
RESET_MODEL_NOISE = 0.01

#: Default MuJoCo simulation substeps per environment step.
DEFAULT_FRAME_SKIP = 5


class BaseDroneEnv(MujocoEnv, utils.EzPickle):
    """Abstract MuJoCo environment for CF2 drone navigation tasks.

    Subclasses must implement :meth:`step` and may override
    :meth:`_reset_task_state` to randomise targets, reset counters,
    etc.  The base class takes care of the action/observation spaces,
    MuJoCo handles, and reset noise so those are consistent across
    tasks.

    Args:
        frame_skip: MuJoCo substeps per environment step.
        max_distance: Distance threshold (drone-to-target) beyond which
            the episode is considered out of bounds.
        **kwargs: Forwarded to :class:`gymnasium.envs.mujoco.MujocoEnv`.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,
    }

    def __init__(
        self,
        *,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        max_distance: float = 2.0,
        **kwargs,
    ):
        utils.EzPickle.__init__(self, **kwargs)

        self.model_path = os.path.join(cf2_mj_description.PACKAGE_PATH, "scene.xml")
        self.max_distance = float(max_distance)

        # The observation space is fixed by :meth:`_get_obs`; publish it
        # before calling ``MujocoEnv.__init__`` so the parent keeps it.
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            self.model_path,
            frame_skip=frame_skip,
            observation_space=self.observation_space,
            **kwargs,
        )

        # MujocoEnv.__init__ calls :meth:`_set_action_space`, which we
        # override below to publish a symmetric, normalized space instead
        # of MuJoCo's raw actuator range.  The raw range is still
        # available for :meth:`_rescale_action`.
        self._action_low = ACTION_LOW.copy()
        self._action_high = ACTION_HIGH.copy()

        # Resolve body / site / sensor handles once so ``step`` and
        # ``_get_obs`` never pay the string-lookup cost per call.
        self.target_pos_id = self.model.site("fly_sensor").id
        self.drone_body_id = self.model.body("cf2").id
        self.gyro_sensor_id = self.model.sensor("body_gyro").id
        self.fly_sensor_id = self.model.sensor("touch_sensor").id
        self.target_pos = self.model.site_pos[self.target_pos_id]
        self.original_site_pos = self.model.site_pos[self.target_pos_id].copy()

        self.step_number = 0
        self.total_reward = 0.0

    # ------------------------------------------------------------------
    # Action / observation plumbing
    # ------------------------------------------------------------------

    def _set_action_space(self):
        """Publish a symmetric, normalized action space.

        Overrides :meth:`MujocoEnv._set_action_space`, which would
        otherwise replace our space with the raw CF2 actuator ranges.
        Using the normalized space is recommended by SB3 and makes the
        policy output scale independent of the physical motor limits.
        """
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        return self.action_space

    def _rescale_action(self, action: np.ndarray) -> np.ndarray:
        """Map a normalized action in :math:`[-1, 1]` to the motor range."""
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        return self._action_low + (action + 1.0) * 0.5 * (
            self._action_high - self._action_low
        )

    def _apply_action(self, action: np.ndarray) -> None:
        """Rescale the action and advance MuJoCo by ``frame_skip`` substeps."""
        self.data.ctrl[:] = self._rescale_action(action)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

    def _get_obs(self) -> np.ndarray:
        """Construct the 19-dim observation vector.

        Layout: ``[drone_pos(3), drone_quat(4), vec_to_target(3),
        drone_lin_vel(3), drone_ang_vel(3), target_pos(3)]``.
        """
        drone_pos = self.data.qpos[:3]
        drone_quat = self.data.qpos[3:7]
        drone_lin_vel = self.data.qvel[:3]
        drone_ang_vel = self.data.sensor(self.gyro_sensor_id).data
        vec_to_target = self.target_pos - drone_pos

        return np.concatenate(
            [
                drone_pos,
                drone_quat,
                vec_to_target,
                drone_lin_vel,
                drone_ang_vel,
                self.target_pos,
            ]
        ).astype(np.float64)

    # ------------------------------------------------------------------
    # Convenience queries used by subclass ``step`` methods.
    # ------------------------------------------------------------------

    def _drone_position(self) -> np.ndarray:
        return self.data.qpos[:3]

    def _drone_speed(self) -> float:
        return float(np.linalg.norm(self.data.qvel[:3]))

    def _distance_to_target(self) -> float:
        return float(np.linalg.norm(self._drone_position() - self.target_pos))

    def _touch_sensor_reading(self) -> float:
        return float(self.data.sensor(self.fly_sensor_id).data[0])

    def _is_out_of_bounds(self, distance_to_target: float) -> bool:
        """Return ``True`` when the drone is too far or has crashed."""
        drone_z = float(self._drone_position()[2])
        return distance_to_target > self.max_distance or drone_z < GROUND_Z_MIN

    def _is_episode_over(self) -> bool:
        """``True`` when the configured episode length has been exceeded.

        Subclasses that want a ``truncated=True`` signal at the episode
        budget can call this helper and remain consistent with the base
        class's step counter.
        """
        episode_len = getattr(self, "episode_len", None)
        if episode_len is None:
            episode_len = getattr(self, "episode_length", None)
        if episode_len is None:
            return False
        return self.step_number > int(episode_len)

    # ------------------------------------------------------------------
    # Reset helpers (shared noise + bookkeeping).
    # ------------------------------------------------------------------

    def _apply_position_reset(self) -> None:
        """Reset MuJoCo data to initial state and add xyz position noise."""
        mujoco.mj_resetData(self.model, self.data)
        noise = self.np_random.uniform(
            low=-RESET_POS_NOISE, high=RESET_POS_NOISE, size=3
        )
        self.data.qpos[:3] += noise
        self.data.qpos[2] = max(INIT_Z_MIN, self.data.qpos[2])

    def _reset_base_state(self) -> None:
        """Reset counters and restore the target site to its original pose.

        Called from :meth:`reset` before any task-specific bookkeeping.
        """
        self.model.site_pos[self.target_pos_id] = self.original_site_pos.copy()
        self.target_pos = self.model.site_pos[self.target_pos_id]
        self.step_number = 0
        self.total_reward = 0.0

    def _reset_task_state(self) -> None:
        """Hook for subclass-specific reset logic.

        Override in a subclass to randomise targets, reset lap counters,
        etc.  The default implementation is a no-op so simple tasks do
        not have to override it.
        """

    def reset(self, seed=None, options=None):
        """Generic reset: seed, apply position noise, delegate to hook."""
        super().reset(seed=seed)
        self._apply_position_reset()
        self._reset_base_state()
        self._reset_task_state()
        return self._get_obs(), {}

    def reset_model(self):
        """Reset internal model state (called by ``MujocoEnv.reset``).

        Applies small uniform noise to the full ``qpos`` / ``qvel``
        vectors so ``MujocoEnv.reset`` leaves a physically-valid state
        behind, even though :meth:`reset` normally overwrites it with
        :meth:`_apply_position_reset`.
        """
        self.step_number = 0
        self.total_reward = 0.0

        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-RESET_MODEL_NOISE, high=RESET_MODEL_NOISE
        )
        qpos[2] = max(INIT_Z_MIN, qpos[2])

        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-RESET_MODEL_NOISE, high=RESET_MODEL_NOISE
        )
        self.set_state(qpos, qvel)
        return self._get_obs()
