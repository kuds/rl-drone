"""Drone racer environment - follow a 3D track through checkpoints."""

import os

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from robot_descriptions import cf2_mj_description

from rl_drone.utils.rewards import modified_tanh, modified_tanh_final, multiplicative_inverse
from rl_drone.utils.track import (
    add_radial_noise_to_points_rng,
    generate_equidistant_points,
    get_next_clockwise_point,
    round_tuple_elements,
)


class DroneRacerEnv(MujocoEnv):
    """Gymnasium environment for drone racing along a circular track.

    The drone must fly through checkpoints arranged in a circle, collecting
    rewards for reaching each checkpoint and completing laps.

    Observation space (19-dim):
        [drone_pos(3), drone_quat(4), vec_to_target(3),
         drone_lin_vel(3), drone_ang_vel(3), target_pos(3)]

    Action space (4-dim):
        [thrust(0..0.35), roll(-1..1), pitch(-1..1), yaw(-1..1)]

    Args:
        env_config: Dictionary with environment configuration. Expected keys:
            - episode_length: Max steps per episode.
            - sphere_size: Checkpoint contact radius.
            - track_size: Radius of the circular track.
            - number_of_checkpoints: Number of waypoints on the track.
            - track_height: Z-height of the track.
            - reward_function: One of "multiplicative_inverse", "modified_tanh",
              "modified_tanh_final", or "none".
            - terminate_without_contact: Steps without contact before termination.
            - speed_factor: Weight for speed reward component.
            - max_distance: Distance threshold for out-of-bounds termination.
            - time_penalty: Per-step penalty to encourage speed.
            - out_of_bounds_penalty: Penalty for going out of bounds.
            - no_contact_penalty: Penalty for no contact timeout.
            - complete_bonus: Bonus for completing a lap.
            - contact_bonus: Bonus for reaching a checkpoint.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,
    }

    REWARD_FUNCTIONS = {
        "multiplicative_inverse": multiplicative_inverse,
        "modified_tanh": modified_tanh,
        "modified_tanh_final": modified_tanh_final,
        "none": lambda x: 0,
    }

    def __init__(self, env_config: dict, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)

        for key, value in env_config.items():
            setattr(self, key, value)

        self.model_path = os.path.join(cf2_mj_description.PACKAGE_PATH, "scene.xml")
        self.total_reward = 0

        self.action_space = Box(
            low=np.array([0.0, -1.0, -1.0, -1.0]),
            high=np.array([0.35, 1.0, 1.0, 1.0]),
            dtype=np.float64,
        )

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            self.model_path,
            frame_skip=5,
            observation_space=self.observation_space,
            **kwargs,
        )

        if self.reward_function not in self.REWARD_FUNCTIONS:
            raise ValueError(f"Unknown reward function: {self.reward_function}")
        self._reward_fn = self.REWARD_FUNCTIONS[self.reward_function]

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
        self.step_number = 0
        self.target_pos_id = self.model.site("fly_sensor").id
        self.target_pos = self.model.site_pos[self.target_pos_id]
        self.drone_body_id = self.model.body("cf2").id
        self.gyro_sensor_id = self.model.sensor("body_gyro").id
        self.fly_sensor_id = self.model.sensor("touch_sensor").id
        self.original_site_pos = self.model.site_pos[self.target_pos_id].copy()

    def _update_track_position(self):
        """Advance the target to the next checkpoint on the track."""
        self.current_point = self.next_point
        self.model.site_pos[self.target_pos_id] = self.current_point
        self.next_point = get_next_clockwise_point(self.current_point, self.track_points)
        self.target_pos = self.model.site_pos[self.target_pos_id]

    def _get_obs(self):
        """Construct the 19-dim observation vector."""
        drone_pos = self.data.qpos[:3]
        drone_quat = self.data.qpos[3:7]
        drone_lin_vel = self.data.qvel[:3]
        drone_ang_vel = self.data.sensor(self.gyro_sensor_id).data
        vec_to_target = self.target_pos - drone_pos

        return np.concatenate(
            [drone_pos, drone_quat, vec_to_target, drone_lin_vel, drone_ang_vel, self.target_pos]
        ).astype(np.float64)

    def step(self, action):
        """Apply action, step simulation, compute reward."""
        truncated = False
        terminated = False
        self.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_number += 1
        drone_pos = self.data.qpos[:3]
        distance_to_target = np.linalg.norm(drone_pos - self.target_pos)

        distance_reward = self._reward_fn(distance_to_target)
        reward = distance_reward

        drone_speed = np.linalg.norm(self.data.qvel[:3])
        speed_reward = self.speed_factor * drone_speed
        reward += speed_reward

        made_contact = 0
        touch_reported = False
        contact_bonus = 0.0
        complete_bonus = 0.0
        oob_penalty = 0.0
        no_contact_penalty = 0.0
        time_penalty_applied = 0.0
        sensor_reading = self.data.sensor(self.fly_sensor_id).data[0]

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

        if distance_to_target > self.max_distance or drone_pos[2] < 0.025:
            no_contact_penalty = self.no_contact_penalty
            reward = no_contact_penalty
            terminated = True
        elif self.step_number > self.episode_length:
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

    def reset(self, seed=None, options=None):
        """Reset to initial state."""
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        noise = self.np_random.uniform(low=-0.1, high=0.1, size=3)
        self.data.qpos[:3] += noise
        self.data.qpos[2] = max(0.075, self.data.qpos[2])

        self.model.site_pos[self.target_pos_id] = self.original_site_pos.copy()
        self.target_pos = self.model.site_pos[self.target_pos_id]
        self.current_point = (0, 0, self.track_height)
        self.next_point = get_next_clockwise_point(self.current_point, self.track_points)
        self.total_contacts = 0
        self.step_number = 0
        self.steps_without_contact = 0
        self.total_reward = 0
        self.laps = 0
        return self._get_obs(), {}

    def reset_model(self, seed=None):
        """Reset internal model state (called by MujocoEnv)."""
        self.step_number = 0
        self.total_reward = 0
        self.total_contacts = 0
        self.laps = 0
        self.steps_without_contact = 0

        self.current_point = (0, 0, self.track_height)
        self.next_point = get_next_clockwise_point(self.current_point, self.track_points)

        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qpos[2] = max(0.075, qpos[2])

        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()
