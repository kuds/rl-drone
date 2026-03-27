"""Reward shaping functions for drone navigation tasks."""

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
