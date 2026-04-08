"""Callback for automatically generating and saving training plots."""

import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrainingPlotsCallback(BaseCallback):
    """Periodically reads evaluation data and saves learning-curve plots as PNG.

    Generates three plots (when enough data exists):
      1. **Reward curve** — mean reward +/- std over training timesteps.
      2. **Episode length curve** — mean episode length +/- std.
      3. **Success rate curve** — fraction of eval episodes where ``info["success"]``
         was ``True`` at the end of the episode (requires the environment to
         populate that key; skipped otherwise).

    It also logs a one-line summary each time a new best model is detected.

    Args:
        eval_file: Path to the ``evaluations.npz`` file written by
            ``EvalCallback``.
        save_path: Directory to write the PNG files into.
        save_freq: Generate plots every *save_freq* training steps.
        name_prefix: Filename prefix for saved images.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        eval_file: str,
        save_path: str,
        save_freq: int = 25_000,
        name_prefix: str = "training",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_file = eval_file
        self.save_path = save_path
        self.save_freq = save_freq
        self.name_prefix = name_prefix
        self._best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True

        try:
            with np.load(self.eval_file) as data:
                if not data.files:
                    return True
                timesteps = data["timesteps"]
                results = data["results"]
                ep_lengths = data["ep_lengths"]
        except FileNotFoundError:
            return True

        if len(timesteps) < 2:
            return True

        os.makedirs(self.save_path, exist_ok=True)

        # Lazy import so matplotlib is only required when this callback fires.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mean_rewards = np.mean(results, axis=1)
        std_rewards = np.std(results, axis=1)
        mean_ep_len = np.mean(ep_lengths, axis=1)
        std_ep_len = np.std(ep_lengths, axis=1)

        # --- Log best model detection ---
        current_best = np.max(mean_rewards)
        if current_best > self._best_mean_reward:
            best_idx = int(np.argmax(mean_rewards))
            self._best_mean_reward = current_best
            print(
                f"TrainingPlots: new best mean reward {current_best:.2f} "
                f"at step {timesteps[best_idx]}"
            )

        # --- 1. Reward curve ---
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(timesteps, mean_rewards, label="Mean reward")
        ax.fill_between(
            timesteps,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.25,
            label="\u00b11 std",
        )
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Reward")
        ax.set_title("Evaluation Reward over Training")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            os.path.join(self.save_path, f"{self.name_prefix}_reward_curve.png"),
            dpi=150,
        )
        plt.close(fig)

        # --- 2. Episode length curve ---
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(timesteps, mean_ep_len, label="Mean episode length", color="tab:orange")
        ax.fill_between(
            timesteps,
            mean_ep_len - std_ep_len,
            mean_ep_len + std_ep_len,
            alpha=0.25,
            color="tab:orange",
            label="\u00b11 std",
        )
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Episode Length (steps)")
        ax.set_title("Episode Length over Training")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            os.path.join(self.save_path, f"{self.name_prefix}_episode_length.png"),
            dpi=150,
        )
        plt.close(fig)

        # --- 3. Success rate curve (optional) ---
        # EvalCallback stores per-episode successes when the env populates
        # info["is_success"].  SB3 also accepts "success" as a key in newer
        # versions.  We attempt both conventions.
        success_data = None
        for key in ("successes", "is_success"):
            if key in np.load(self.eval_file).files:
                success_data = np.load(self.eval_file)[key]
                break

        if success_data is not None and success_data.size > 0:
            mean_success = np.mean(success_data, axis=1)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(timesteps, mean_success * 100, label="Success rate", color="tab:green")
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Success Rate (%)")
            ax.set_title("Success Rate over Training")
            ax.set_ylim(-5, 105)
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(
                os.path.join(self.save_path, f"{self.name_prefix}_success_rate.png"),
                dpi=150,
            )
            plt.close(fig)

        return True
