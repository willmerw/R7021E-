"""Likelihood field (endpoint) measurement model.

Scores a batch of candidate poses against ONE particle's map. Called N times
per filter step, once per particle, with K candidates each time -- so the K
dimension must be vectorised, never looped.

This is a proper likelihood, not a correlation. Summing occupancy
probabilities at the beam endpoints would be an unnormalized correlation
score; in Grid-FastSLAM 2.0 that quantity enters tau, so it would corrupt the
proposal mean and covariance, not merely the weights.
"""
import numpy as np

from .utils import transform_points



def measurement_log_likelihood(poses, endpoints, grid_map, cfg):
    world_points = transform_points(poses, endpoints)
    grid_points = grid_map.world_to_grid(world_points)
    good_points = grid_map.in_bounds(grid_points)

    likelihood_field = grid_map.likelihood_field(
        cfg.occ_thresh,
        cfg.max_dist,
    )

    distances = np.full(
        good_points.shape,
        cfg.max_dist,
        dtype=float,
    )

    gx = grid_points[..., 0]
    gy = grid_points[..., 1]

    distances[good_points] = likelihood_field[
        gy[good_points],
        gx[good_points],
    ]

    p_beam = (
        cfg.z_hit
        * np.exp(-0.5 * (distances / cfg.sigma_hit) ** 2)
        + cfg.z_rand / cfg.range_max
    )

    log_likelihood = np.sum(np.log(p_beam), axis=1)
    return log_likelihood