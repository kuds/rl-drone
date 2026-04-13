"""Callback for saving experiment configuration at the start of training."""

import json
import os
from datetime import datetime, timezone
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback


class ConfigSaveCallback(BaseCallback):
    """Saves experiment metadata and hyperparameters as a JSON file.

    The file is written once, at the very first training step, so the
    configuration is recorded before any training occurs.  It captures:

    * All user-supplied hyperparameters.
    * Algorithm name, policy class, observation/action space shapes.
    * A UTC timestamp and optional run-name tag.

    Args:
        save_path: Directory to write the config JSON file.
        hyperparams: Arbitrary dict of hyperparameters / env config to persist.
        run_name: Optional human-readable name for this training run.
        file_name: Name of the output file.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        save_path: str,
        hyperparams: dict[str, Any] | None = None,
        run_name: str = "",
        file_name: str = "config.json",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.hyperparams = hyperparams or {}
        self.run_name = run_name
        self.file_name = file_name
        self._saved = False

    def _on_step(self) -> bool:
        if self._saved:
            return True

        os.makedirs(self.save_path, exist_ok=True)

        model = self.model
        config: dict[str, Any] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "run_name": self.run_name,
            "algorithm": type(model).__name__,
            "policy": str(model.policy_class.__name__),
            "observation_space": str(model.observation_space),
            "action_space": str(model.action_space),
            "n_envs": model.n_envs,
            "device": str(model.device),
            "gpu_device": _describe_gpu_device(model),
            "learning_rate": _extract_lr(model),
            "gamma": getattr(model, "gamma", None),
            "batch_size": getattr(model, "batch_size", None),
            "buffer_size": getattr(model, "buffer_size", None),
            "tau": getattr(model, "tau", None),
            "ent_coef": _format_ent_coef(model),
            "total_timesteps": getattr(model, "_total_timesteps", None),
            "hyperparams": self.hyperparams,
        }

        # Remove None values for a cleaner file.
        config = {k: v for k, v in config.items() if v is not None}

        path = os.path.join(self.save_path, self.file_name)
        with open(path, "w") as f:
            json.dump(config, f, indent=2, default=str)

        if self.verbose:
            print(f"ConfigSaveCallback: saved experiment config to {path}")

        self._saved = True
        return True


def _extract_lr(model) -> float | str | None:
    lr = getattr(model, "learning_rate", None)
    if callable(lr):
        try:
            return float(lr(1.0))
        except Exception:
            return str(lr)
    return lr


def _format_ent_coef(model) -> str | float | None:
    ent = getattr(model, "ent_coef", None)
    if ent is None:
        return None
    if isinstance(ent, str):
        return ent
    return ent


def _describe_gpu_device(model) -> dict[str, Any] | None:
    """Return a dict describing the GPU the model is running on, if any."""
    try:
        import torch
    except ImportError:
        return None

    device = getattr(model, "device", None)
    if device is None:
        return None

    device_str = str(device)
    if not device_str.startswith("cuda"):
        return None

    if not torch.cuda.is_available():
        return None

    index = device.index if hasattr(device, "index") and device.index is not None else 0
    try:
        name = torch.cuda.get_device_name(index)
    except Exception:
        return None

    info: dict[str, Any] = {
        "index": index,
        "name": name,
        "cuda_version": torch.version.cuda,
    }

    try:
        props = torch.cuda.get_device_properties(index)
        info["total_memory_bytes"] = int(props.total_memory)
        info["multi_processor_count"] = int(props.multi_processor_count)
        info["compute_capability"] = f"{props.major}.{props.minor}"
    except Exception:
        pass

    return info
