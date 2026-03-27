"""Track generation utilities for the drone racer environment."""

import math

import numpy as np


def generate_equidistant_points(
    radius: float, num_points: int, z_height: float
) -> list[tuple[float, float, float]]:
    """Generate equally spaced points around a circle.

    Points start from the positive x-axis and move counter-clockwise.
    The circle center is offset so that one point sits at the origin (x=0, y=0).

    Args:
        radius: Radius of the circle.
        num_points: Number of points on the circumference.
        z_height: Constant z-coordinate for all points.

    Returns:
        List of (x, y, z) tuples.
    """
    if radius <= 0:
        raise ValueError("Radius must be a positive number.")
    if num_points <= 0:
        raise ValueError("Number of points must be a positive integer.")

    points = []
    angle_step = 2 * math.pi / num_points

    for i in range(num_points):
        angle = angle_step * i
        x = radius * math.cos(angle) + radius
        y = radius * math.sin(angle)
        points.append((x, y, z_height))

    return points


def get_next_clockwise_point(
    current_point: tuple[float, float, float],
    all_points: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Find the next point in the clockwise direction.

    Args:
        current_point: The (x, y, z) starting point.
        all_points: All equidistant points on the circle.

    Returns:
        The next (x, y, z) point clockwise.

    Raises:
        ValueError: If current_point is not found in all_points.
    """
    idx = -1
    for i, p in enumerate(all_points):
        if np.isclose(p[0], current_point[0]) and np.isclose(p[1], current_point[1]):
            idx = i
            break

    if idx == -1:
        raise ValueError(
            f"The provided current_point {current_point} is not in the list of points."
        )

    num_points = len(all_points)
    next_index = (idx - 1 + num_points) % num_points
    return all_points[next_index]


def add_radial_noise_to_points_rng(
    points,
    r_noise_std_dev: float,
    z_noise_std_dev: float,
    skip_origin: bool = False,
    seed: int | None = None,
) -> list[tuple[float, float, float]]:
    """Apply radial and z-axis noise to 3D points using a seeded RNG.

    Noise is added to the radius (distance from center) in polar space,
    preserving the original angle. The z-axis gets independent noise.

    Args:
        points: Input (N, 3) coordinates.
        r_noise_std_dev: Std deviation of radius noise.
        z_noise_std_dev: Std deviation of z-axis noise.
        skip_origin: If True, skip the point at (0, 0, z).
        seed: Random seed for reproducibility.

    Returns:
        List of noisy (x, y, z) tuples.
    """
    rng = np.random.default_rng(seed)

    points_arr = np.array(points)
    original_points = points_arr.copy()
    origin_index = 0

    if skip_origin:
        is_x_zero = np.isclose(points_arr[:, 0], 0, atol=1e-05)
        is_y_zero = np.isclose(points_arr[:, 1], 0, atol=1e-05)
        mask = is_x_zero & is_y_zero
        origin_index = np.argmax(mask)
        points_arr = points_arr[~mask]

    num_points = len(points_arr)

    z_noise = rng.normal(0, z_noise_std_dev, num_points)
    new_z = points_arr[:, 2] + z_noise

    r_noise = rng.normal(0, r_noise_std_dev, num_points)

    x = points_arr[:, 0]
    y = points_arr[:, 1]

    r = np.hypot(x, y)
    theta = np.arctan2(y, x)

    new_r = r + r_noise

    new_x = new_r * np.cos(theta)
    new_y = new_r * np.sin(theta)

    new_points_arr = np.stack((new_x, new_y, new_z), axis=1)
    if skip_origin:
        new_points_arr = np.insert(
            new_points_arr, origin_index, original_points[origin_index], axis=0
        )

    return [tuple(row) for row in new_points_arr]


def round_tuple_elements(numeric_tuple: tuple) -> tuple:
    """Round each element of a numeric tuple to two decimal places."""
    return tuple(round(item, 2) for item in numeric_tuple)
