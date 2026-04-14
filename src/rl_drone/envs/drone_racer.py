"""Drone racer environment - follow a 3D track through checkpoints."""

from rl_drone.envs.base import BaseDroneEnv, DEFAULT_FRAME_SKIP
from rl_drone.utils.rewards import get_reward_function
from rl_drone.utils.track import (
    add_radial_noise_to_points_rng,
    generate_equidistant_points,
    get_next_clockwise_point,
    round_tuple_elements,
)


# Keys accepted in ``env_config`` along with their defaults.  Centralising
# them here means typos in a notebook config raise a clear ``ValueError``
# instead of silently creating an attribute that ``step`` never reads.
_DEFAULT_CONFIG = {
    "episode_length": 1_000,
    "sphere_size": 0.2,
    "track_size": 2.0,
    "number_of_checkpoints": 8,
    "track_height": 1.0,
    "reward_function": "modified_tanh",
    "terminate_without_contact": 200,
    "speed_factor": 0.0,
    "max_distance": 3.0,
    "time_penalty": 0.0,
    "out_of_bounds_penalty": -10.0,
    "no_contact_penalty": -10.0,
    "complete_bonus": 10.0,
    "contact_bonus": 1.0,
    "frame_stack": 1,
}


class DroneRacerEnv(BaseDroneEnv):
    """Gymnasium environment for drone racing along a circular track.

    The drone must fly through checkpoints arranged in a circle,
    collecting rewards for reaching each checkpoint and completing laps.

    Observation, action, and reset logic are inherited from
    :class:`~rl_drone.envs.base.BaseDroneEnv`; this subclass only
    implements the track state and the racing reward.

    Observation space:
        A flat vector of ``19 * frame_stack`` values.  Each 19-dim frame
        is laid out as ``[drone_pos(3), drone_quat(4), vec_to_target(3),
        drone_lin_vel(3), drone_ang_vel(3), target_pos(3)]``; with the
        default ``frame_stack=1`` the observation is the raw 19-dim
        vector.

    Action space (4-dim, normalized to ``[-1, 1]``):
        ``[thrust, roll, pitch, yaw]``.  Actions are rescaled internally
        to the physical motor ranges (``thrust -> [0, 0.35]``,
        roll/pitch/yaw ``-> [-1, 1]``).

    Args:
        env_config: Dictionary with environment configuration.  Any
            missing key falls back to the default defined in
            ``_DEFAULT_CONFIG``.  Unknown keys raise ``ValueError`` to
            catch typos early.  Expected keys:

            - ``episode_length``: Max steps per episode.
            - ``sphere_size``: Checkpoint contact radius.
            - ``track_size``: Radius of the circular track.
            - ``number_of_checkpoints``: Number of waypoints on the track.
            - ``track_height``: Z-height of the track.
            - ``reward_function``: One of the keys registered in
              :data:`~rl_drone.utils.rewards.REWARD_FUNCTIONS`
              (``"multiplicative_inverse"``, ``"modified_tanh"``,
              ``"modified_tanh_final"``, or ``"none"``).
            - ``terminate_without_contact``: Steps without contact before
              termination.
            - ``speed_factor``: Weight for speed reward component.
            - ``max_distance``: Distance threshold for out-of-bounds
              termination.
            - ``time_penalty``: Per-step penalty to encourage speed.
            - ``out_of_bounds_penalty``: Penalty for going out of bounds.
            - ``no_contact_penalty``: Penalty for no contact timeout.
            - ``complete_bonus``: Bonus for completing a lap.
            - ``contact_bonus``: Bonus for reaching a checkpoint.
            - ``frame_stack``: Number of consecutive raw observations to
              concatenate into the agent's observation.  Defaults to
              ``1`` (no stacking); larger values let an MLP policy
              recover temporal information directly from the input.
    """

    def __init__(self, env_config: dict, **kwargs):
        self._apply_env_config(env_config)

        super().__init__(
            frame_skip=DEFAULT_FRAME_SKIP,
            max_distance=self.max_distance,
            frame_stack=self.frame_stack,
            **kwargs,
        )

        self._reward_fn = get_reward_function(self.reward_function)

        self.base_points = generate_equidistant_points(
            self.track_size, self.number_of_checkpoints, self.track_height
        )
        self.track_points = add_radial_noise_to_points_rng(
            self.base_points, 0.25, 0.25, skip_origin=True, seed=0
        )

        self.current_point = (0, 0, self.track_height)
        self.next_point = get_next_clockwise_point(self.current_point, self.track_points)
        self.laps = 0
        self.total_contacts = 0
        self.steps_without_contact = 0

    # ------------------------------------------------------------------
    # Config / target plumbing
    # ------------------------------------------------------------------

    def _apply_env_config(self, env_config: dict) -> None:
        """Validate ``env_config`` and copy values onto ``self``.

        Missing keys fall back to :data:`_DEFAULT_CONFIG`; unknown keys
        raise ``ValueError`` so typos are surfaced before training
        starts.
        """
        unknown = set(env_config) - set(_DEFAULT_CONFIG)
        if unknown:
            known = ", ".join(sorted(_DEFAULT_CONFIG))
            raise ValueError(
                f"Unknown env_config keys: {sorted(unknown)}. Known: {known}."
            )
        for key, default in _DEFAULT_CONFIG.items():
            setattr(self, key, env_config.get(key, default))

    def _update_track_position(self) -> None:
        """Advance the target to the next checkpoint on the track."""
        self.current_point = self.next_point
        self.model.site_pos[self.target_pos_id] = self.current_point
        self.next_point = get_next_clockwise_point(
            self.current_point, self.track_points
        )
        self.target_pos = self.model.site_pos[self.target_pos_id]

    def _reset_task_state(self) -> None:
        """Reset track bookkeeping at the start of an episode."""
        self.current_point = (0, 0, self.track_height)
        self.next_point = get_next_clockwise_point(
            self.current_point, self.track_points
        )
        self.total_contacts = 0
        self.steps_without_contact = 0
        self.laps = 0

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        """Apply action, step simulation, compute reward."""
        self._apply_action(action)
        self.step_number += 1

        truncated = False
        terminated = False

        distance_to_target = self._distance_to_target()
        drone_speed = self._drone_speed()

        distance_reward = self._reward_fn(distance_to_target)
        speed_reward = self.speed_factor * drone_speed
        reward = distance_reward + speed_reward

        made_contact = 0
        touch_reported = False
        contact_bonus = 0.0
        complete_bonus = 0.0
        oob_penalty = 0.0
        no_contact_penalty = 0.0
        time_penalty_applied = 0.0
        sensor_reading = self._touch_sensor_reading()

        if sensor_reading > 0:
            touch_reported = True
            contact_bonus = self.contact_bonus
            reward += contact_bonus
        elif distance_to_target <= self.sphere_size:
            made_contact = 1
            contact_bonus = self.contact_bonus
            reward += contact_bonus
            rounded_current = round_tuple_elements(self.current_point)
            if self.total_contacts > 0 and rounded_current == (0, 0, self.track_height):
                self.laps += 1
                terminated = True
                complete_bonus = self.complete_bonus
                reward += complete_bonus

            self._update_track_position()
            self.total_contacts += 1
            self.steps_without_contact = 0
        else:
            self.steps_without_contact += 1
            if self.steps_without_contact > self.terminate_without_contact:
                oob_penalty = self.out_of_bounds_penalty
                reward = oob_penalty
                terminated = True

        if self._is_out_of_bounds(distance_to_target):
            no_contact_penalty = self.no_contact_penalty
            reward = no_contact_penalty
            terminated = True
        elif self._is_episode_over():
            truncated = True

        if not terminated and not truncated:
            time_penalty_applied = self.time_penalty
            reward -= time_penalty_applied

        self.total_reward += reward
        observation = self._get_obs()

        info = {
            "distance_to_target": distance_to_target,
            "touch_reported": touch_reported,
            "sensor_reading": sensor_reading,
            "made_contact": made_contact,
            "total_reward": self.total_reward,
            "total_contacts": self.total_contacts,
            "drone_speed": drone_speed,
            "steps_without_contact": self.steps_without_contact,
            "laps": self.laps,
            "success": self.laps > 0,
            "reward_distance": distance_reward,
            "reward_speed": speed_reward,
            "reward_contact": contact_bonus,
            "reward_complete": complete_bonus,
            "reward_oob_penalty": oob_penalty,
            "reward_no_contact_penalty": no_contact_penalty,
            "reward_time_penalty": time_penalty_applied,
        }

        return observation, reward, terminated, truncated, info
