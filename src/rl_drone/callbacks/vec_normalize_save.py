"""Callback for saving VecNormalize statistics alongside the best model."""

import os

from stable_baselines3.common.callbacks import BaseCallback


class VecNormalizeSaveCallback(BaseCallback):
    """Saves the VecNormalize running statistics on every step.

    Intended for use as ``callback_on_new_best`` inside an ``EvalCallback``,
    so that the normalization state is saved whenever a new best model is found.

    Args:
        save_path: Directory to save the pickle file.
        file_name: Filename for the saved normalization state.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        save_path: str,
        file_name: str = "best_model_vec_normalize.pkl",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.file_name = file_name
        self.save_path = save_path

    def _on_step(self) -> bool:
        self.training_env.save(os.path.join(self.save_path, self.file_name))
        return True
