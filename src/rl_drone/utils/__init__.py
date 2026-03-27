"""Utility functions for reward shaping, track generation, and model setup."""

from rl_drone.utils.rewards import modified_tanh, modified_tanh_final, multiplicative_inverse
from rl_drone.utils.track import (
    generate_equidistant_points,
    get_next_clockwise_point,
    add_radial_noise_to_points_rng,
)
from rl_drone.utils.model_xml import setup_mujoco_model

__all__ = [
    "modified_tanh",
    "modified_tanh_final",
    "multiplicative_inverse",
    "generate_equidistant_points",
    "get_next_clockwise_point",
    "add_radial_noise_to_points_rng",
    "setup_mujoco_model",
]
