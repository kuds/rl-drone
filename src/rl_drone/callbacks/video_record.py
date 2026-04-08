"""Callback for periodically recording agent behavior during training."""

import csv
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder


class VideoRecordCallback(BaseCallback):
    """Records video of the agent at regular intervals during training.

    At each recording interval, the callback creates a fresh environment,
    copies the current VecNormalize statistics from the training environment
    (so observations and rewards match what the policy expects), and records
    a video plus a per-step CSV log.

    The CSV columns are determined dynamically from the environment's ``info``
    dict so this callback works with any environment (hover, racer, etc.).

    Args:
        make_env_fn: Callable that creates a single environment instance.
        save_path: Directory to save videos and CSVs.
        video_length: Maximum number of steps per recording.
        save_freq: Record every N training steps.
        name_prefix: Filename prefix for saved files.
        verbose: Verbosity level.
    """

    # Base columns always written before info-dict keys.
    _BASE_COLUMNS = [
        "step",
        "drone_pos_x",
        "drone_pos_y",
        "drone_pos_z",
        "reward",
        "reward_normalized",
        "total_reward",
        "done",
    ]

    def __init__(
        self,
        make_env_fn,
        save_path: str,
        video_length: int,
        save_freq: int = 5_000,
        name_prefix: str = "rl_model",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.make_env_fn = make_env_fn
        self.save_freq = save_freq
        self.video_length = video_length
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True

        name_prefix = f"{self.name_prefix}_{self.num_timesteps}"

        rec_val = make_vec_env(self.make_env_fn, n_envs=1)
        rec_val = VecNormalize(rec_val, training=False, norm_obs=True, norm_reward=True)

        # Copy normalisation statistics from the training environment so the
        # recorded rollout uses the same observation / reward scaling that the
        # policy was trained with.
        self._sync_vec_normalize(rec_val)

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
        info_keys_written = False
        info_keys: list[str] = []

        with open(csv_file_name, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=",")

            for _ in range(self.video_length):
                action, _states = self.model.predict(obs)
                obs_norm, rewards_norm, done, info = rec_val.step(action)
                obs = rec_val.get_original_obs()
                rewards = rec_val.get_original_reward()
                total_reward += rewards

                # On the first step, discover info keys and write the header.
                if not info_keys_written:
                    info_keys = sorted(info[0].keys())
                    csv_writer.writerow(self._BASE_COLUMNS + info_keys)
                    info_keys_written = True

                base_data = [
                    int(session_length),
                    round(float(obs[0][0]), 4),
                    round(float(obs[0][1]), 4),
                    round(float(obs[0][2]), 4),
                    round(float(rewards[0]), 4),
                    round(float(rewards_norm[0]), 4),
                    round(float(total_reward[0]), 4),
                    int(done[0]),
                ]
                info_data = [
                    round(float(info[0][k]), 4)
                    if isinstance(info[0][k], (int, float, np.floating, np.integer))
                    else info[0][k]
                    for k in info_keys
                ]
                csv_writer.writerow(base_data + info_data)

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

    def _sync_vec_normalize(self, target_env: VecNormalize) -> None:
        """Copy running mean/var from the training VecNormalize to *target_env*."""
        training_env = self.training_env
        # Walk wrapper stack to find VecNormalize.
        while training_env is not None and not isinstance(training_env, VecNormalize):
            training_env = getattr(training_env, "venv", None)

        if training_env is None or not isinstance(training_env, VecNormalize):
            if self.verbose:
                print(
                    "VideoRecordCallback: could not find VecNormalize in "
                    "training env — using default statistics."
                )
            return

        target_env.obs_rms = training_env.obs_rms.copy()
        target_env.ret_rms = training_env.ret_rms.copy()
        target_env.clip_obs = training_env.clip_obs
        target_env.clip_reward = training_env.clip_reward
        target_env.epsilon = training_env.epsilon
