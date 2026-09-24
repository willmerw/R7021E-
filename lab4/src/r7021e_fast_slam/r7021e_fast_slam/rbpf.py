"""Grid-FastSLAM 2.0: the filter itself.

The algorithm of hello-slam's `4_grid_based_slam.ipynb`. Each particle carries
a pose hypothesis and its own occupancy grid. Per particle and per step:
predict from odometry (3), scan-match against that particle's OWN map (5),
build a Gaussian proposal around the match (6-10), sample the new pose (11),
update the weight with the proposal's normalizer (12), and only then integrate
the scan (22). Then normalize (24), compute N_eff (25) and resample
selectively (26-27).

No rclpy in here. The ROS node feeds this class odometry increments and beam
endpoints and does its own publishing.

Occupancy mapping without a filter -- `use_measurement_update: false` -- does
NOT go through this class at all. The node runs `MappingOnlyFilter` for that,
so the very first task works before a single line of this file exists.
"""
from dataclasses import dataclass
from time import perf_counter
from typing import List

import numpy as np

from .evaluation import StepInfo
from .grid_map import GridMap
from .measurement_model import measurement_log_likelihood
from .motion_model import sample_motion_model_odometry
from .proposal import improved_proposal
from .resampling import effective_sample_size, systematic_resample
from .scan_matcher import match as scan_match
from .utils import logsumexp, wrap_angle


@dataclass
class Particle:
    """A single particle in the Grid-FastSLAM 2.0 filter.
    Holds the robot's pose, the particle's log-weight, and its own occupancy grid.
    """
    pose: np.ndarray
    log_weight: float
    grid: GridMap


class GridFastSLAM:
    def __init__(self, cfg, rng):
        """Create N particles, all at the origin with equal log-weights.

        Each particle gets its OWN GridMap; they must never share one, or
        resampling would corrupt every hypothesis at once.
        """
        self.cfg = cfg
        self.rng = rng
        n = int(cfg.num_particles)
        ## Initialise the particle set with N particles at the origin, each with its own GridMap.
        self.particles: List[Particle] = [
            Particle(np.zeros(3), -np.log(n), GridMap(cfg)) for _ in range(n)
        ]

    def best_particle(self) -> Particle:
        """The heaviest particle -- the hypothesis published as the estimate."""
        return self.particles[int(np.argmax([p.log_weight
                                             for p in self.particles]))]

    def step(self, u, endpoints, ranges) -> StepInfo:
        """Run one filter update over the whole particle set.

        The map is integrated LAST on purpose. Every likelihood evaluated
        during a step must see m_{t-1}: scoring a scan against a map that
        already contains it is a different, far more optimistic estimator.

        `use_improved_proposal=False` samples from the motion model and
        weights by the measurement likelihood alone -- FastSLAM 1.0, the
        baseline the improved proposal is measured against.

        Args:
            u:         (3,) odometry increment [d_rot1, d_trans, d_rot2].
            endpoints: (B, 2) beam endpoints in the robot base frame.
            ranges:    (B,) beam ranges, used to tell a hit from a max-range
                    return.

        Returns a StepInfo carrying N_eff, whether resampling fired, the
        scan-match fallback count, the per-stage timings and the index of the
        best particle. The particle set is updated in place; read it back
        through `self.particles` and `best_particle()`.

        TODO:
            Implement the full Grid-FastSLAM 2.0 step:
            1. Predict each particle's pose using the odometry model.
            2. Scan-match against the particle's own map.
            3. Build the improved proposal distribution.
            4. Sample the new pose from the proposal.
            5. Update the particle's weight with the proposal's normalization constant.
            6. Integrate the scan into the particle's map.
            7. Normalize weights and resample if necessary.
        """

        ## Sketch of one step.
        ##
        cfg = self.cfg
        timings = {'scan_match': 0.0, 'likelihood': 0.0, 'map_integrate': 0.0}
        n_fallback = 0
        #
        # Split the beams before the loop, not inside it -- the same arrays
        # are reused by every particle:
        hits = ranges < cfg.range_max - 1e-6  # (a max-range return
        #              says "nothing out to here"; it maps free space but
        #              must never score a pose)
        ep_hit   = endpoints[hits]
        ep_match = ep_hit[::cfg.scan_match_stride]   #(for the matcher)
        ep_w     = ep_hit[::cfg.likelihood_stride]   #(for the weight)
        #
        for p in self.particles:
            x_bar = sample_motion_model_odometry(u, p.pose[None, :], cfg.odometry_sigmas, self.rng)[0]
            if cfg.use_improved_proposal:
                t_sm_start = perf_counter()
                x_star, score = scan_match(x_bar, p.grid, ep_match, cfg,)
                timings['scan_match'] += perf_counter() - t_sm_start
                #time it into timings['scan_match']
            else:
                #FastSLAM 1.0 baseline:
                x_star, score = x_bar, -np.inf
            if score >= cfg.match_score_min:
                log_eta, mu, sigma = improved_proposal(x_star, p.pose, u, ep_w, p.grid, cfg)
                x_new = self.rng.multivariate_normal(mu, sigma) #wrap_angle the heading
                x_new[2] = wrap_angle(x_new[2])
                p.log_weight += log_eta
            else:
                x_new = x_bar
                x_new[2] = wrap_angle(x_new[2])
                t_lik_start = perf_counter()
                p.log_weight += measurement_log_likelihood(x_bar, ep_w, p.grid, cfg)[0]
                timings['likelihood'] += perf_counter() - t_lik_start
                if cfg.use_improved_proposal is True:
                    n_fallback += 1
            p.pose = x_new
            t_map_start = perf_counter()
            p.grid.integrate_scan(x_new, endpoints, ranges, cfg.range_max)
            timings['map_integrate'] += perf_counter() - t_map_start

        weights, n_eff = self._normalize()
        best_idx = int(np.argmax(p.log_weight for p in self.particles)) # Calculate before resampling
        resampled = self._maybe_resample(weights, n_eff)

        return StepInfo(n_eff=n_eff, resampled=resampled, n_fallback=n_fallback, timings=timings, best_index=best_idx)

        ## Sketch of one step.
        ##
        ## cfg = self.cfg
        ## timings = {'scan_match': 0.0, 'likelihood': 0.0, 'map_integrate': 0.0}
        ## n_fallback = 0
        ##
        ## Split the beams before the loop, not inside it -- the same arrays
        ## are reused by every particle:
        ##   hits     = ranges < cfg.range_max - 1e-6   (a max-range returntimings
        ##              says "nothing out to here"; it maps free space but
        ##              must never score a pose)
        ##   ep_hit   = endpoints[hits]
        ##   ep_match = ep_hit[::cfg.scan_match_stride]   (for the matcher)
        ##   ep_w     = ep_hit[::cfg.likelihood_stride]   (for the weight)
        ##
        ## for p in self.particles:
        ##     (3)  x_bar = sample_motion_model_odometry(u, p.pose[None, :],
        ##                      cfg.odometry_sigmas, self.rng)[0]
        ##     (5)  if cfg.use_improved_proposal:
        ##              x_star, score = scan_match(x_bar, p.grid, ep_match, cfg)
        ##              time it into timings['scan_match']
        ##          else:
        ##              FastSLAM 1.0 baseline: x_star, score = x_bar, -inf
        ##     if score >= cfg.match_score_min:
        ##         (6-10) log_eta, mu, sigma = improved_proposal(
        ##                    x_star, p.pose, u, ep_w, p.grid, cfg)
        ##         (11)   x_new = self.rng.multivariate_normal(mu, sigma),
        ##                wrap_angle the heading
        ##         (12)   p.log_weight += log_eta
        ##     else:
        ##         The match is unreliable (a featureless corridor, say).
        ##         Skip 5-12: keep x_bar and weight by the likelihood alone,
        ##         p.log_weight += measurement_log_likelihood(x_bar, ep_w,
        ##                             p.grid, cfg)[0]
        ##         count it in n_fallback when the improved proposal was on.
        ##         Time this whole branch into timings['likelihood'].
        ##     p.pose = x_new
        ##     (22) p.grid.integrate_scan(x_new, endpoints, ranges,
        ##              cfg.range_max)  -- LAST, so every likelihood above saw
        ##              m_{t-1}. Time it into timings['map_integrate'].
        ##
        ## (24-25) weights, n_eff = self._normalize()
        ## (26-27) resampled = self._maybe_resample(weights, n_eff)
        ##
        ## Report the PRE-resample n_eff: it is the quantity that made the
        ## resampling decision. Overwriting it with N afterwards pins the trace
        ## at N on every resampled step and hides the depletion.
        ## return StepInfo(n_eff=n_eff, resampled=resampled,
        ##                 n_fallback=n_fallback, timings=timings,
        ##                 best_index=argmax of the log-weights)
        #raise NotImplementedError

    def _normalize(self):
        """Normalize the particle weights and compute the effective sample size.

        Returns:

            weights : np.ndarray
                The normalized weights of the particles.
            n_eff : float
                The effective sample size.
        """
        log_w = np.array([p.log_weight for p in self.particles], dtype=float)
        log_w -= logsumexp(log_w)
        for p, lw in zip(self.particles, log_w):
            p.log_weight = float(lw)
        weights = np.exp(log_w)
        return weights, effective_sample_size(weights)

    def _maybe_resample(self, weights, n_eff) -> bool:
        """Resample the particle set if the diversity has collapsed.

        Args:
            weights: np.ndarray
                The normalized particle weights.
            n_eff: float
                The effective sample size before resampling.

        Returns:
            bool: True when a resample step was performed, otherwise False.
        """
        cfg = self.cfg
        n = len(self.particles)
        due = cfg.resample_every_step or n_eff < cfg.resample_threshold * n
        if not due:
            return False

        idx = systematic_resample(weights, self.rng)
        ## Every survivor gets a full grid copy, including the ones selected
        ## exactly once. This is the straightforward reading of the algorithm
        ## and it is what makes a large particle set expensive.
        self.particles = [
            Particle(self.particles[i].pose.copy(), -np.log(n),
                     self.particles[i].grid.copy())
            for i in idx
        ]
        return True