"""Drone hovering environment - fly to and maintain position at a target point."""

import os

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from robot_descriptions import cf2_mj_description

from rl_drone.utils.rewards import modified_tanh_final


class DroneHoverEnv(MujocoEnv):
    """Gymnasium environment for drone hovering / random-point / multi-target tasks.

    The drone must navigate to a target sphere in 3D space. Depending on
    configuration, the target may be fixed (hover), randomized per episode
    (random point), or moved after each contact (multiple targets).

    Observation space (19-dim):
        [drone_pos(3), drone_quat(4), vec_to_target(3),
         drone_lin_vel(3), drone_ang_vel(3), target_pos(3)]

    Action space (4-dim):
        [thrust(0..0.35), roll(-1..1), pitch(-1..1), yaw(-1..1)]

    Args:
        sphere_size: Radius of the target sphere for contact detection.
        episode_len: Maximum steps per episode.
        frame_skip: Number of physics substeps per action.
        randomize_target: If True, randomize the target position each episode.
        noise_magnitude: Std dev for target randomization noise.
        move_on_contact: If True, move the target after each contact.
        max_distance: Distance threshold for early termination.
        contact_reward: Reward for reaching the target.
        out_of_bounds_penalty: Penalty for going out of bounds.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,
    }

    def __init__(
        self,
        sphere_size=0.2,
        episode_len=1_000,
        frame_skip=5,
        randomize_target=False,
        noise_magnitude=0.25,
        move_on_contact=False,
        max_distance=2.0,
        contact_reward=1.0,
        out_of_bounds_penalty=-10.0,
        **kwargs,
    ):
        utils.EzPickle.__init__(self, **kwargs)

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
            frame_skip=frame_skip,
            observation_space=self.observation_space,
            **kwargs,
        )

        self.step_number = 0
        self.sphere_size = sphere_size
        self.episode_len = episode_len
        self.frame_skip = frame_skip
        self.randomize_target = randomize_target
        self.noise_magnitude = noise_magnitude
        self.move_on_contact = move_on_contact
        self.max_distance = max_distance
        self.contact_reward = contact_reward
        self.out_of_bounds_penalty = out_of_bounds_penalty

        self.target_pos_id = self.model.site("fly_sensor").id
        self.target_pos = self.model.site_pos[self.target_pos_id]
        self.drone_body_id = self.model.body("cf2").id
        self.gyro_sensor_id = self.model.sensor("body_gyro").id
        self.fly_sensor_id = self.model.sensor("touch_sensor").id
        self.original_site_pos = self.model.site_pos[self.target_pos_id].copy()
        self.total_contacts = 0

        if self.randomize_target:
            self._randomize_fly_zone()

    def _randomize_fly_zone(self):
        """Move the target to a random position near the original site."""
        noise = self.noise_magnitude * np.random.randn(3)
        self.model.site_pos[self.target_pos_id] = self.original_site_pos + noise
        self.model.site_pos[self.target_pos_id][2] = max(
            0.5, self.model.site_pos[self.target_pos_id][2]
        )
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
        self.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_number += 1
        drone_pos = self.data.qpos[:3]
        distance_to_target = np.linalg.norm(drone_pos - self.target_pos)
        distance_reward = modified_tanh_final(distance_to_target)
        reward = distance_reward
        contact_bonus = 0.0
        oob_penalty = 0.0

        made_contact = 0
        touch_reported = False
        sensor_reading = self.data.sensor(self.fly_sensor_id).data[0]

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
        if distance_to_target > self.max_distance or drone_pos[2] < 0.025:
            oob_penalty = self.out_of_bounds_penalty
            reward = oob_penalty
            terminated = True
        elif self.step_number > self.episode_len:
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
            "success": success,
            "reward_distance": distance_reward,
            "reward_contact": contact_bonus,
            "reward_oob_penalty": oob_penalty,
        }

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset to initial state with small position noise."""
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        noise = self.np_random.uniform(low=-0.1, high=0.1, size=3)
        self.data.qpos[:3] += noise
        self.data.qpos[2] = max(0.075, self.data.qpos[2])

        self.model.site_pos[self.target_pos_id] = self.original_site_pos.copy()
        self.target_pos = self.model.site_pos[self.target_pos_id]
        self.total_contacts = 0
        self.step_number = 0
        self.total_reward = 0

        if self.randomize_target:
            self._randomize_fly_zone()

        return self._get_obs(), {}

    def reset_model(self, seed=None):
        """Reset internal model state (called by MujocoEnv)."""
        self.step_number = 0
        self.total_reward = 0

        if self.randomize_target:
            self._randomize_fly_zone()

        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qpos[2] = max(0.075, qpos[2])

        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()
