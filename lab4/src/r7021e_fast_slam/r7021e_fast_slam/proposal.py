"""The improved proposal distribution of Grid-FastSLAM 2.0.

The Gaussian approximation derived in hello-slam's `4_grid_based_slam.ipynb`:
sample K poses around the scan match, weight each by

    tau_j = p(z_t | x_j, m_t-1) * p(x_j | x_t-1, u_t),

and fit a Gaussian to the weighted cloud,

    mu = (1/eta) sum_j x_j tau_j,   Sigma = (1/eta) sum_j (x_j - mu)(x_j - mu)^T tau_j.

The normalizer eta = sum_j tau_j of that same product is the particle's
importance weight, which is why it is returned alongside the moments.

Everything is in log space. The raw products underflow within a handful of
beams.
"""
import numpy as np

from .measurement_model import measurement_log_likelihood
from .motion_model import motion_model_log_pdf
from .utils import logsumexp, wrap_angle

## Regularization of the fitted covariance. With all the tau mass on a single
## candidate the scatter matrix is exactly singular and the sampler's Cholesky
## would fail. SIGMA_REG is an additive variance floor per axis (x, y, theta);
## SIGMA_EIG_MIN floors the eigenvalues afterwards.
SIGMA_REG = (1.0e-4, 1.0e-4, 1.0e-5)
SIGMA_EIG_MIN = 1.0e-6


def proposal_candidates(x_star, cfg) -> np.ndarray:
    """K candidate poses around the scan-matched pose.

    A regular lattice: it covers the small search window evenly.
    K is rounded down to a perfect cube, e.g. 27 gives 3 x 3 x 3.

    Args:
        x_star:     (3,) [x, y, theta] the scan-matched pose at the current time step.
        cfg:        configuration object with proposal parameters.

    Returns:
        (K, 3) numpy array of candidate poses, where K is the number of candidates.

    TODO:
        Implement the generation of candidate poses around the scan-matched pose `x_star`
        using a regular lattice within the specified proposal window.
    """

    center = np.asarray(x_star, dtype=float) #center of lattice

    k = cfg.num_candidates
    w = cfg.proposal_window_xy
    wTheta = cfg.proposal_window_theta

    rounded_n = int(np.floor(np.cbrt(k))) # k is first taken cubed root then rounded down

    if rounded_n == 1:
        d_xy = d_theta = [0.0]
    else:
        d_xy = np.linspace(-w, w, rounded_n)
        d_theta = np.linspace(-wTheta, wTheta, rounded_n)

    dx, dy, dt = np.meshgrid(d_xy, d_xy, d_theta, indexing="ij")

    candi_poses = np.stack([center[0] + dx.ravel(),
                            center[1] + dy.ravel(),
                            wrap_angle(center[2] + dt.ravel())], axis=1)


    return candi_poses


def improved_proposal(x_star, x_prev, u, endpoints, grid_map, cfg):
    """Return (log_eta, mu, Sigma) for one particle.

    Args:
        x_star:     (3,) [x, y, theta] the scan-matched pose at the current time step.
        x_prev:     (3,) [x, y, theta] the previous pose of the particle.
        u:          (3,) [rot1, trans, rot2] the odometry increment.
        endpoints:  (N, 2) array of laser scan endpoints in the robot frame.
        grid_map:   occupancy grid map.
        cfg:        configuration object with proposal parameters.

    Returns:
        log_eta:    float, the importance weight factor for the particle.
        mu:        (3,) numpy array, the mean of the candidate poses.
        Sigma:     (3, 3) numpy array, the covariance of the candidate poses.

    log_eta is log sum_j tau_j -- the importance weight factor the filter
    applies to the particle. It is the sum, NOT the likelihood at the sampled
    pose: it approximates the integral p(z | x_prev, m, u), which is what the
    weight is supposed to be.

    TODO:
        Use `proposal_candidates` to generate candidate poses around `x_star`,
        then compute the importance weights `log_tau` for each candidate using
        the measurement likelihood and motion model.
        Normalize the weights to get `w`, and use them to compute the weighted
        mean `mu` and covariance `Sigma`.
    """
    candidates = proposal_candidates(x_star, cfg)
    sig = cfg.odometry_sigmas
    log_tau = measurement_log_likelihood(candidates, endpoints, grid_map, cfg) + motion_model_log_pdf(candidates, x_prev, u, sig)
    log_eta = logsumexp(log_tau)

    w = np.exp(log_tau - log_eta)

    #d = candidates - x_star



    #d[:,2] = wrap_angle(d[:,2])
    d_mu = np.empty(3)
    d_mu[:2] = w @ candidates[:, :2]
    d_mu[2] = float(np.arctan2(w @ np.sin(candidates[:, 2]), w @ np.cos(candidates[:, 2])))

    #mu = x_star + d_mu


    #mu[2] = wrap_angle(mu[2])
    #e = d - d_mu
    e = np.empty_like(candidates)
    e[:,:2] = candidates[:,:2] - d_mu[:2]
    e[:, 2] = wrap_angle(candidates[:, 2] - d_mu[2])

    sigma = (w[:, None] * e).T @ e + np.diag(SIGMA_REG)
    sigma = 0.5 * (sigma + sigma.T)
    vals,vecs = np.linalg.eigh(sigma)
    vals = np.maximum(vals, SIGMA_EIG_MIN)
    sigma = vecs @ np.diag(vals) @ vecs.T
    sigma = 0.5 * (sigma + sigma.T)

    return float(log_eta), d_mu, sigma
