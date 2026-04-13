"""Drone hovering environment - fly to and maintain position at a target point."""

import numpy as np

from rl_drone.envs.base import BaseDroneEnv, DEFAULT_FRAME_SKIP
from rl_drone.utils.rewards import modified_tanh_final


class DroneHoverEnv(BaseDroneEnv):
    """Gymnasium environment for drone hovering / random-point / multi-target tasks.

    The drone must navigate to a target sphere in 3D space. Depending on
    configuration, the target may be fixed (hover), randomized per episode
    (random point), or moved after each contact (multiple targets).

    The observation, action, and reset behavior are inherited from
    :class:`~rl_drone.envs.base.BaseDroneEnv`; this class only implements
    the task-specific ``step``/target-randomisation logic.

    Observation space (19-dim):
        ``[drone_pos(3), drone_quat(4), vec_to_target(3),
        drone_lin_vel(3), drone_ang_vel(3), target_pos(3)]``

    Action space (4-dim, normalized to ``[-1, 1]``):
        ``[thrust, roll, pitch, yaw]``.  Actions are rescaled internally
        to the physical motor ranges (``thrust -> [0, 0.35]``,
        roll/pitch/yaw ``-> [-1, 1]``).

    Args:
        sphere_size: Radius of the target sphere for contact detection.
        episode_len: Maximum steps per episode.
        frame_skip: Number of physics substeps per action.
        randomize_target: If ``True``, randomize the target position each episode.
        noise_magnitude: Std dev for target randomization noise.
        move_on_contact: If ``True``, move the target after each contact.
        max_distance: Distance threshold for early termination.
        contact_reward: Reward for reaching the target.
        out_of_bounds_penalty: Penalty for going out of bounds.
    """

    def __init__(
        self,
        sphere_size: float = 0.2,
        episode_len: int = 1_000,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        randomize_target: bool = False,
        noise_magnitude: float = 0.25,
        move_on_contact: bool = False,
        max_distance: float = 2.0,
        contact_reward: float = 1.0,
        out_of_bounds_penalty: float = -10.0,
        **kwargs,
    ):
        self.sphere_size = sphere_size
        self.episode_len = episode_len
        self.randomize_target = randomize_target
        self.noise_magnitude = noise_magnitude
        self.move_on_contact = move_on_contact
        self.contact_reward = contact_reward
        self.out_of_bounds_penalty = out_of_bounds_penalty

        super().__init__(
            frame_skip=frame_skip,
            max_distance=max_distance,
            **kwargs,
        )

        self.total_contacts = 0

        if self.randomize_target:
            self._randomize_fly_zone()

    def _randomize_fly_zone(self) -> None:
        """Move the target to a random position near the original site."""
        noise = self.noise_magnitude * np.random.randn(3)
        new_pos = self.original_site_pos + noise
        new_pos[2] = max(0.5, new_pos[2])
        self.model.site_pos[self.target_pos_id] = new_pos
        self.target_pos = self.model.site_pos[self.target_pos_id]

    def _reset_task_state(self) -> None:
        """Reset contact counter and optionally re-randomize the target."""
        self.total_contacts = 0
        if self.randomize_target:
            self._randomize_fly_zone()

    def step(self, action):
        """Apply action, step simulation, compute reward."""
        self._apply_action(action)
        self.step_number += 1

        distance_to_target = self._distance_to_target()
        drone_speed = self._drone_speed()

        distance_reward = modified_tanh_final(distance_to_target)
        reward = distance_reward
        contact_bonus = 0.0
        oob_penalty = 0.0

        made_contact = 0
        touch_reported = False
        sensor_reading = self._touch_sensor_reading()

        if sensor_reading > 0:
            touch_reported = True
            contact_bonus = self.contact_reward
            reward = contact_bonus
        elif distance_to_target <= self.sphere_size:
            made_contact = 1
            contact_bonus = self.contact_reward
            reward = contact_bonus
            if self.move_on_contact:
                self._randomize_fly_zone()

        truncated = False
        terminated = False
        if self._is_out_of_bounds(distance_to_target):
            oob_penalty = self.out_of_bounds_penalty
            reward = oob_penalty
            terminated = True
        elif self._is_episode_over():
            truncated = True

        self.total_reward += reward
        self.total_contacts += made_contact
        success = touch_reported or made_contact > 0

        observation = self._get_obs()
        info = {
            "distance_to_target": distance_to_target,
            "touch_reported": touch_reported,
            "sensor_reading": sensor_reading,
            "made_contact": made_contact,
            "total_contacts": self.total_contacts,
            "drone_speed": drone_speed,
            "success": success,
            "reward_distance": distance_reward,
            "reward_contact": contact_bonus,
            "reward_oob_penalty": oob_penalty,
        }

        return observation, reward, terminated, truncated, info
