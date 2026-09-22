"""Odometry motion model: sampling from it, and evaluating its density.

The three-parameter model of hello-slam's `4_grid_based_slam.ipynb`: the
odometry increment is decomposed into (rot1, trans, rot2) and each component
gets its own Gaussian, with the standard deviation given directly.

    sigmas = [sd_rot1 (rad), sd_trans (m), sd_rot2 (rad)]

The decomposition itself, `odometry_increment`, is plain SE(2) bookkeeping and
lives in `utils.py`. What belongs here is the modelling: putting noise on those
three numbers, and writing down the density that noise implies.

FastSLAM 1.0 only ever samples from this model. Grid-FastSLAM 2.0 also needs
to *evaluate* it at poses that were not drawn from it, because the improved
proposal weights each candidate by likelihood x motion density.
"""
import numpy as np

from .utils import TRANS_EPS, wrap_angle

## Never divide by a zero standard deviation in the density.
_SIGMA_FLOOR = 1e-4


def sample_motion_model_odometry(u, poses, sigmas, rng) -> np.ndarray:
    """Draw one perturbed pose per input pose. 
    We need to sample from the odometry motion model for each particle.

    Args:
        u:      (3,) [d_rot1, d_trans, d_rot2]
        poses:  (N, 3) current particle poses
        sigmas: (3,) [sd_rot1, sd_trans, sd_rot2]
        rng:    numpy Generator

    Returns:
        (N, 3) perturbed particle poses

    TODO: 
        Implement the odometry motion model sampling. 
        Add Gaussian noise to each component of the odometry 
        increment and apply it to the current particle poses.
    """
    raise NotImplementedError


def motion_model_log_pdf(x_cand, x_prev, u, sigmas) -> np.ndarray:
    """log p(x_cand | x_prev, u) for a batch of candidate poses.

    Args:
        x_cand: (N, 3) candidate poses
        x_prev: (3,) previous odometry pose [x, y, theta]
        u:      (3,) [d_rot1, d_trans, d_rot2]
        sigmas: (3,) [sd_rot1, sd_trans, sd_rot2]

    Returns:
        (N,) log probabilities of the candidate poses

    TODO:
        Implement the odometry motion model log density calculation.
        Compute the errors between the predicted and actual odometry increments,
        normalize by the standard deviations, and return the log probability
        for each candidate pose.
    """
    raise NotImplementedError
