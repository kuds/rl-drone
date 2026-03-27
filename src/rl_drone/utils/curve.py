"""3D curve fitting for creating smooth racing tracks through waypoints."""

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import minimize_scalar


class Curve3D:
    """Smooth 3D curve through a set of points using B-spline interpolation.

    Supports closed (looping) and open curves, with nearest-point queries
    for tracking tasks.

    Args:
        points: Array of shape (n, 3) with 3D waypoints.
        degree: Spline degree (1=linear, 2=quadratic, 3=cubic).
        closed: If True, creates a closed curve connecting last point to first.
        num_samples: Number of points to generate for visualization.
    """

    def __init__(self, points, degree=3, closed=True, num_samples=100):
        self.original_points = np.array(points)
        self.degree = degree
        self.closed = closed

        if self.original_points.shape[0] < 2:
            raise ValueError("At least 2 points are required")
        if self.original_points.shape[1] != 3:
            raise ValueError("Points must be 3-dimensional")

        self._build_splines()

        if self.closed:
            original_t = self.t_params[:-1]
        else:
            original_t = self.t_params

        t_uniform = np.linspace(0, 1, num_samples)
        t_combined = np.unique(np.concatenate([original_t, t_uniform]))
        self.curve_points = self.evaluate(t_combined)

    def _build_splines(self):
        """Build the spline representation of the curve."""
        points = self.original_points.copy()

        if self.closed:
            points = np.vstack([points, points[0:1]])

        n_points = len(points)
        degree = min(self.degree, n_points - 1)

        distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
        cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
        self.t_params = cumulative_distances / cumulative_distances[-1]

        self.splines = []
        for dim in range(3):
            if self.closed and n_points > degree + 1:
                bc_type = "periodic"
            else:
                bc_type = "not-a-knot"

            try:
                spline = make_interp_spline(
                    self.t_params, points[:, dim], k=degree, bc_type=bc_type
                )
            except ValueError:
                spline = make_interp_spline(
                    self.t_params, points[:, dim], k=degree, bc_type="not-a-knot"
                )

            self.splines.append(spline)

    def evaluate(self, t):
        """Evaluate the curve at parameter value(s) t in [0, 1]."""
        t = np.atleast_1d(t)
        points = np.column_stack([spline(t) for spline in self.splines])
        return points.squeeze() if len(t) == 1 else points

    def find_nearest_point(self, query_point, num_samples=1000, refine=True):
        """Find the nearest point on the curve to a query point.

        Args:
            query_point: 3D point to query.
            num_samples: Sampling density for initial search.
            refine: If True, refine with scipy optimization.

        Returns:
            Tuple of (nearest_point, distance, t_parameter).
        """
        query_point = np.array(query_point)

        t_samples = np.linspace(0, 1, num_samples)
        curve_samples = self.evaluate(t_samples)
        distances = np.linalg.norm(curve_samples - query_point, axis=1)
        min_idx = np.argmin(distances)
        t_initial = t_samples[min_idx]

        if not refine:
            return curve_samples[min_idx], distances[min_idx], t_initial

        def distance_function(t):
            t = np.clip(t, 0, 1)
            point_on_curve = self.evaluate(t)
            return np.linalg.norm(point_on_curve - query_point)

        window = 0.1
        bounds = (max(0, t_initial - window), min(1, t_initial + window))
        result = minimize_scalar(distance_function, bounds=bounds, method="bounded")

        t_nearest = result.x
        nearest_point = self.evaluate(t_nearest)
        return nearest_point, result.fun, t_nearest

    def get_points(self, num_samples=None, include_original=True):
        """Get points along the curve.

        Args:
            num_samples: Number of points (None returns cached points).
            include_original: Ensure original waypoints are included.

        Returns:
            Array of shape (N, 3).
        """
        if num_samples is None:
            return self.curve_points.copy()

        if not include_original:
            t_samples = np.linspace(0, 1, num_samples)
            return self.evaluate(t_samples)

        if self.closed:
            original_t = self.t_params[:-1]
        else:
            original_t = self.t_params

        t_uniform = np.linspace(0, 1, num_samples)
        t_combined = np.unique(np.concatenate([original_t, t_uniform]))
        return self.evaluate(t_combined)

    def get_length(self, num_samples=1000):
        """Estimate the total arc length of the curve."""
        t_samples = np.linspace(0, 1, num_samples)
        curve_samples = self.evaluate(t_samples)
        segment_lengths = np.linalg.norm(np.diff(curve_samples, axis=0), axis=1)
        return np.sum(segment_lengths)
