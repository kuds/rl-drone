"""Reward shaping functions for drone navigation tasks."""

from typing import Callable

import numpy as np


def modified_tanh(x):
    """Smooth sigmoid-like reward that maps distance to [0, 1].

    Returns ~1.0 for small distances, ~0.0 for large distances.
    Transition centered around x=0.575 with steepness k=5.406.
    """
    k = 5.406
    x0 = 0.575
    return 0.5 * (1 - np.tanh(k * (x - x0)))


def modified_tanh_final(x, factor=0.5):
    """Smooth reward that maps distance to [-factor, factor].

    Returns ~+factor for small distances, ~-factor for large distances.
    Transition centered around x=1.1 with steepness k=2.941.
    """
    k = 2.941
    x0 = 1.1
    return -np.tanh(k * (x - x0)) * factor


def multiplicative_inverse(x, factor=2):
    """Inverse-distance reward: 1 / (1 + x * factor).

    Returns 1.0 at x=0 and decays smoothly toward 0.
    """
    return 1.0 / (1.0 + x * factor)


def _no_reward(x):  # pragma: no cover - trivial
    return 0.0


# Centralized registry so every environment resolves reward-function names
# the same way. Add new functions here rather than in each env class.
REWARD_FUNCTIONS: dict[str, Callable[[float], float]] = {
    "multiplicative_inverse": multiplicative_inverse,
    "modified_tanh": modified_tanh,
    "modified_tanh_final": modified_tanh_final,
    "none": _no_reward,
}


def get_reward_function(name: str) -> Callable[[float], float]:
    """Look up a reward function by name from :data:`REWARD_FUNCTIONS`.

    Raises ``ValueError`` with the list of known names if the key is
    unknown, which gives config typos a clearer failure mode than a
    plain ``KeyError``.
    """
    try:
        return REWARD_FUNCTIONS[name]
    except KeyError as exc:
        known = ", ".join(sorted(REWARD_FUNCTIONS))
        raise ValueError(
            f"Unknown reward function: {name!r}. Known: {known}."
        ) from exc
