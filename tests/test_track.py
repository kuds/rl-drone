"""Tests for track generation utilities."""

import math

import numpy as np
import pytest

from rl_drone.utils.track import (
    add_radial_noise_to_points_rng,
    generate_equidistant_points,
    get_next_clockwise_point,
    round_tuple_elements,
)


class TestGenerateEquidistantPoints:
    def test_correct_count(self):
        points = generate_equidistant_points(1.0, 8, 1.0)
        assert len(points) == 8

    def test_all_at_correct_height(self):
        points = generate_equidistant_points(1.0, 6, 2.5)
        for _, _, z in points:
            assert z == 2.5

    def test_first_point_at_double_radius(self):
        r = 1.5
        points = generate_equidistant_points(r, 4, 1.0)
        # First point: x = r*cos(0) + r = 2r, y = 0
        assert np.isclose(points[0][0], 2 * r)
        assert np.isclose(points[0][1], 0.0)

    def test_points_on_circle(self):
        r = 2.0
        points = generate_equidistant_points(r, 12, 0.5)
        for x, y, _ in points:
            # Center is at (r, 0), so distance from center = r
            dist = math.sqrt((x - r) ** 2 + y**2)
            assert np.isclose(dist, r, atol=1e-10)

    def test_invalid_radius_raises(self):
        with pytest.raises(ValueError):
            generate_equidistant_points(0, 5, 1.0)
        with pytest.raises(ValueError):
            generate_equidistant_points(-1, 5, 1.0)

    def test_invalid_num_points_raises(self):
        with pytest.raises(ValueError):
            generate_equidistant_points(1.0, 0, 1.0)


class TestGetNextClockwisePoint:
    def test_wraps_around(self):
        points = generate_equidistant_points(1.0, 4, 1.0)
        # Going clockwise from index 0 should wrap to the last point
        nxt = get_next_clockwise_point(points[0], points)
        assert np.isclose(nxt[0], points[-1][0])
        assert np.isclose(nxt[1], points[-1][1])

    def test_normal_progression(self):
        points = generate_equidistant_points(1.0, 4, 1.0)
        nxt = get_next_clockwise_point(points[2], points)
        assert np.isclose(nxt[0], points[1][0])
        assert np.isclose(nxt[1], points[1][1])

    def test_invalid_point_raises(self):
        points = generate_equidistant_points(1.0, 4, 1.0)
        with pytest.raises(ValueError):
            get_next_clockwise_point((999, 999, 999), points)


class TestAddRadialNoiseRng:
    def test_reproducible_with_seed(self):
        points = generate_equidistant_points(1.0, 6, 1.0)
        result1 = add_radial_noise_to_points_rng(points, 0.1, 0.1, seed=42)
        result2 = add_radial_noise_to_points_rng(points, 0.1, 0.1, seed=42)
        for p1, p2 in zip(result1, result2):
            assert np.allclose(p1, p2)

    def test_different_seeds_differ(self):
        points = generate_equidistant_points(1.0, 6, 1.0)
        result1 = add_radial_noise_to_points_rng(points, 0.5, 0.5, seed=1)
        result2 = add_radial_noise_to_points_rng(points, 0.5, 0.5, seed=2)
        assert not all(np.allclose(p1, p2) for p1, p2 in zip(result1, result2))

    def test_preserves_count(self):
        points = generate_equidistant_points(1.0, 8, 1.0)
        result = add_radial_noise_to_points_rng(points, 0.1, 0.1, seed=0)
        assert len(result) == 8

    def test_skip_origin(self):
        points = generate_equidistant_points(1.0, 8, 1.0)
        result = add_radial_noise_to_points_rng(
            points, 0.1, 0.1, skip_origin=True, seed=0
        )
        assert len(result) == 8


class TestRoundTupleElements:
    def test_rounds_correctly(self):
        assert round_tuple_elements((1.556, 2.444, 3.999)) == (1.56, 2.44, 4.0)

    def test_integers_unchanged(self):
        assert round_tuple_elements((1, 2, 3)) == (1, 2, 3)
