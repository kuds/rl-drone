"""Callback for periodically recording agent behavior during training."""

import csv
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder


class VideoRecordCallback(BaseCallback):
    """Records video of the agent at regular intervals during training.

    Args:
        make_env_fn: Callable that creates a single environment instance.
        csv_header: List of column names for the CSV log.
        save_path: Directory to save videos and CSVs.
        video_length: Maximum number of steps per recording.
        save_freq: Record every N training steps.
        name_prefix: Filename prefix for saved files.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        make_env_fn,
        csv_header,
        save_path: str,
        video_length: int,
        save_freq: int = 5_000,
        name_prefix: str = "rl_model",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.make_env_fn = make_env_fn
        self.csv_header = csv_header
        self.save_freq = save_freq
        self.video_length = video_length
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            name_prefix = f"{self.name_prefix}_{self.num_timesteps}"

            rec_val = make_vec_env(self.make_env_fn, n_envs=1)
            rec_val = VecNormalize(rec_val, training=False, norm_obs=True, norm_reward=True)
            rec_val = VecVideoRecorder(
                rec_val,
                self.save_path,
                video_length=self.video_length,
                record_video_trigger=lambda x: x == 0,
                name_prefix=name_prefix,
            )

            obs = rec_val.reset()
            session_length = 0
            total_reward = 0.0
            csv_file_name = os.path.join(self.save_path, f"{name_prefix}.csv")

            with open(csv_file_name, "w", newline="") as csvfile:
                csv_writer = csv.writer(csvfile, delimiter=",")
                csv_writer.writerow(self.csv_header)
                for _ in range(self.video_length):
                    action, _states = self.model.predict(obs)
                    obs_norm, rewards_norm, done, info = rec_val.step(action)
                    obs = rec_val.get_original_obs()
                    rewards = rec_val.get_original_reward()
                    total_reward += rewards

                    row_data = np.concatenate(
                        [
                            [
                                int(session_length),
                                obs[0][0],
                                obs[0][1],
                                obs[0][2],
                                rewards[0],
                                rewards_norm[0],
                                total_reward[0],
                                info[0].get("touch_reported", False),
                                info[0].get("sensor_reading", 0),
                                info[0].get("distance_to_target", 0),
                                done[0],
                            ]
                        ]
                    )
                    row_data = np.round(row_data, decimals=4)
                    csv_writer.writerow(row_data)
                    rec_val.render()
                    session_length += 1
                    if done:
                        break

            print(
                f"Step: {self.num_timesteps} | "
                f"Session Length: {session_length} | "
                f"Total Reward: {int(total_reward[0])}"
            )
            rec_val.close()

        return True
