"""Correlative coarse-to-fine scan matcher.

This is the "pose correction using scan matching" of hello-slam's
`4_grid_based_slam.ipynb`: transform the beam endpoints by a candidate pose,
look up the particle's own occupancy probabilities there, and keep the
candidate with the highest agreement.

Correlation is legitimate here because the result is only ever used as an
argmax -- the location of the maximum, not its value. It must never be used as
tau in the proposal, where an unnormalized score would corrupt the proposal
mean and covariance as well as the weights.
"""
import numpy as np

from .utils import transform_points, wrap_angle

## Scale of the Gaussian motion prior relative to the correlation score.
##
## This is a heuristic, and deliberately so: the score mixes an UNNORMALIZED
## correlation with a proper log-density, so there is no principled exchange
## rate between them. Larger values pin the match to the odometry prediction,
## smaller ones let the scan pull it further. The prior's sigmas are not set
## here -- they come from `cfg.scan_match_prior_sigmas`, i.e. from the motion
## model. The prior shapes which candidate wins but is excluded from the
## reported score, so a fixed `match_score_min` keeps its meaning.
PRIOR_WEIGHT = 0.1


def candidate_grid(center, win_xy, step_xy, win_th, step_th) -> np.ndarray:
    """Regular lattice of candidate poses centred on `center`."""
    center = np.asarray(center, dtype=float)
    n_xy = int(np.floor(win_xy / (step_xy+1e-6)))
    n_th = int(np.floor(win_th / (step_th+1e-6)))
    d_xy = np.arange(-n_xy, n_xy + 1) * step_xy
    d_th = np.arange(-n_th, n_th + 1) * step_th
    dx, dy, dth = np.meshgrid(d_xy, d_xy, d_th, indexing='ij')
    return np.stack([center[0] + dx.ravel(),
                     center[1] + dy.ravel(),
                     wrap_angle(center[2] + dth.ravel())], axis=1)


def correlation(grid_map, cands, endpoints) -> np.ndarray:
    """Mean occupancy probability at each candidate's beam endpoints.

    Normalizing by the in-bounds beam count keeps the result in [0, 1] and
    comparable across scans, which is what makes a fixed match threshold
    meaningful.
    """
    world = transform_points(cands, endpoints)
    cells = grid_map.world_to_grid(world)
    inb = grid_map.in_bounds(cells)
    probs = grid_map.probs()
    vals = np.zeros(world.shape[:2], dtype=float)
    gx, gy = cells[..., 0], cells[..., 1]
    vals[inb] = probs[gy[inb], gx[inb]]
    n = np.maximum(inb.sum(axis=1), 1)
    return vals.sum(axis=1) / n


def match(pose_pred, grid_map, endpoints, cfg):
    """Maximise scan-map agreement near `pose_pred`.

    Searches the stages of `cfg.scan_match_stages`, coarse first, each one
    centred on the previous winner, with a Gaussian prior pulling the match
    back toward `pose_pred` at the strength implied by the motion model.

    Returns (pose_star, score), score being the correlation at the winner,
    in [0, 1].
    """
    pose_pred = np.asarray(pose_pred, dtype=float)
    endpoints = np.asarray(endpoints, dtype=float)
    if endpoints.shape[0] == 0:
        return pose_pred.copy(), 0.0

    sig_trans, sig_rot = cfg.scan_match_prior_sigmas
    center = pose_pred.copy()
    score = 0.0
    for win_xy, step_xy, win_th, step_th in cfg.scan_match_stages:
        cands = candidate_grid(center, win_xy, step_xy, win_th, step_th)
        corr = correlation(grid_map, cands, endpoints)

        dx = cands[:, 0] - pose_pred[0]
        dy = cands[:, 1] - pose_pred[1]
        dth = wrap_angle(cands[:, 2] - pose_pred[2])
        penalty = (0.5 * (dx * dx + dy * dy) / sig_trans ** 2
                   + 0.5 * dth * dth / sig_rot ** 2)

        best = int(np.argmax(corr - PRIOR_WEIGHT * penalty))
        center = cands[best].copy()
        score = float(corr[best])
    return center, score
