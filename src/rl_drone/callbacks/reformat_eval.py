"""Callback for converting evaluation .npz files to CSV summaries."""

import csv
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ReformatEvalCallback(BaseCallback):
    """Periodically converts evaluation data from .npz to a summary CSV.

    Args:
        save_path: Directory to save the CSV.
        eval_file: Path to the evaluations.npz file.
        save_freq: Convert every N training steps.
        csv_file_name: Name (without extension) for the output CSV.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        save_path: str,
        eval_file: str,
        save_freq: int = 5_000,
        csv_file_name: str = "rl_model",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_file = eval_file
        self.csv_file_name = csv_file_name
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            try:
                with np.load(self.eval_file) as data_archive:
                    if not data_archive.files:
                        return True

                    timesteps = data_archive["timesteps"]
                    ep_lengths = data_archive["ep_lengths"]
                    results = data_archive["results"]

                    row_avg_ep = np.mean(ep_lengths, axis=1)
                    row_std_ep = np.std(ep_lengths, axis=1)
                    row_avg_reward = np.mean(results, axis=1)
                    row_std_reward = np.std(results, axis=1)

                    csv_file_path = os.path.join(
                        self.save_path, f"{self.csv_file_name}.csv"
                    )
                    with open(csv_file_path, "w", newline="") as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(
                            [
                                "steps",
                                "reward_average",
                                "reward_standard_deviation",
                                "episode_average",
                                "episode_standard_deviation",
                            ]
                        )
                        for row in zip(
                            timesteps, row_avg_reward, row_std_reward, row_avg_ep, row_std_ep
                        ):
                            csv_writer.writerow(row)

            except FileNotFoundError:
                if self.verbose:
                    print(f"Eval file not found: {self.eval_file}")

        return True
