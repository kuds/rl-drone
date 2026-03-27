"""Tests for reward shaping functions."""

import numpy as np

from rl_drone.utils.rewards import modified_tanh, modified_tanh_final, multiplicative_inverse


class TestModifiedTanh:
    def test_returns_near_one_for_zero_distance(self):
        assert modified_tanh(0.0) > 0.95

    def test_returns_near_zero_for_large_distance(self):
        assert modified_tanh(5.0) < 0.05

    def test_monotonically_decreasing(self):
        distances = np.linspace(0, 3, 50)
        values = [modified_tanh(d) for d in distances]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]

    def test_vectorized(self):
        distances = np.array([0.0, 0.5, 1.0, 2.0])
        result = modified_tanh(distances)
        assert result.shape == (4,)


class TestModifiedTanhFinal:
    def test_positive_for_small_distance(self):
        assert modified_tanh_final(0.0) > 0

    def test_negative_for_large_distance(self):
        assert modified_tanh_final(5.0) < 0

    def test_factor_scales_output(self):
        val_default = modified_tanh_final(0.5)
        val_scaled = modified_tanh_final(0.5, factor=1.0)
        assert abs(val_scaled) > abs(val_default)


class TestMultiplicativeInverse:
    def test_returns_one_at_zero(self):
        assert multiplicative_inverse(0.0) == 1.0

    def test_decreasing(self):
        assert multiplicative_inverse(1.0) < multiplicative_inverse(0.5)

    def test_always_positive(self):
        for d in [0, 0.1, 1.0, 10.0, 100.0]:
            assert multiplicative_inverse(d) > 0

    def test_factor_changes_decay(self):
        val_default = multiplicative_inverse(1.0)
        val_steep = multiplicative_inverse(1.0, factor=10)
        assert val_steep < val_default
