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


def measurement_log_likelihood(poses, endpoints, grid_map, cfg) -> np.ndarray:
    """log p(z | x, m) for each of K candidate poses.

    Args:
        poses:     (3,) or (K, 3) in the map frame.
        endpoints: (B, 2) beam endpoints in the robot base frame.
        grid_map:  the particle's GridMap.
        cfg:       Config; uses z_hit, z_rand, sigma_hit, max_dist, occ_thresh
                and range_max.

    Returns:
        (K,) array of log-likelihoods for each candidate pose.
    
    TODO:
        Implement the likelihood field measurement model as described in Task 3.
        Transform the beam endpoints into the map frame, look up the distance
        to the nearest occupied cell, and compute the log-likelihood using
        the Gaussian and uniform mixture model.
    NOTE:
        1) Use the transform_points function to convert beam endpoints from the 
        robot base frame to the map frame.
        2) Use the grid_map.world_to_grid and grid_map.in_bounds methods to handle 
        the conversion from world coordinates to grid coordinates and to check 
        if the points are within the map bounds.
        3) Use the distance transform provided by grid_map.likelihood_field to efficiently 
        compute distances for all beam endpoints at once.
        4) Compute the log probability for each beam using the Gaussian and uniform mixture model:
            p_b = z_hit * np.exp(-0.5 * (dist / sigma) ** 2) + z_rand / range_max
        5) Sum the log probabilities over all beams to get the final log-likelihood for each candidate pose.

    """
    raise NotImplementedError
