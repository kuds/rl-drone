"""Utility functions for reward shaping, track generation, model setup, plotting, and paths."""

from rl_drone.utils.rewards import (
    REWARD_FUNCTIONS,
    get_reward_function,
    modified_tanh,
    modified_tanh_final,
    multiplicative_inverse,
)
from rl_drone.utils.track import (
    generate_equidistant_points,
    get_next_clockwise_point,
    add_radial_noise_to_points_rng,
)
from rl_drone.utils.model_xml import setup_mujoco_model
from rl_drone.utils.plotting import (
    plot_cross_run_comparison,
    plot_learning_curves,
    plot_reward_breakdown,
    plot_training_losses,
    plot_training_reward_over_time,
    plot_trajectory_3d,
)
from rl_drone.utils.paths import (
    RunPaths,
    TIMESTAMP_FORMAT,
    build_run_paths,
    default_timestamp,
)
from rl_drone.utils.summary import (
    format_duration,
    format_stage_summary,
    read_eval_history,
    run_best_model_evaluation,
    write_stage_summary,
)
from rl_drone.utils.versioning import get_git_version_info

__all__ = [
    "REWARD_FUNCTIONS",
    "get_reward_function",
    "modified_tanh",
    "modified_tanh_final",
    "multiplicative_inverse",
    "generate_equidistant_points",
    "get_next_clockwise_point",
    "add_radial_noise_to_points_rng",
    "setup_mujoco_model",
    "plot_cross_run_comparison",
    "plot_learning_curves",
    "plot_reward_breakdown",
    "plot_training_losses",
    "plot_training_reward_over_time",
    "plot_trajectory_3d",
    "RunPaths",
    "TIMESTAMP_FORMAT",
    "build_run_paths",
    "default_timestamp",
    "format_duration",
    "format_stage_summary",
    "read_eval_history",
    "run_best_model_evaluation",
    "write_stage_summary",
    "get_git_version_info",
]
