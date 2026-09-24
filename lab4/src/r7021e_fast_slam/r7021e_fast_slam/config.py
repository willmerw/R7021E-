"""Every tunable of the SLAM system, in one place.

This mirrors the `class Config` of hello-slam's `4_grid_based_slam.ipynb`: one
object, passed to the grid map, the scan matcher, the proposal and the filter.
Nothing here imports rclpy except `config_from_node`, which is the single
bridge between ROS parameters and the filter.

This file owns the STRUCTURE of the configuration: the field names, their
types, and the quantities derived from them. `config/params.yaml` owns the
VALUES the node actually runs with, and it wins over anything here --
change a value there, not in this file.

Only a subset of these fields appears in `params.yaml` -- the ones the lab
asks you to tune. Every field is still a declared ROS parameter, so anything
can be overridden from a YAML file or the command line.

The derived properties at the bottom are the values the code actually uses.
They are computed, never configured, so that the parameters you set stay the
ones you learned about in the notebooks.
"""
from dataclasses import dataclass, fields
from typing import Sequence

import numpy as np

from .utils import logit


@dataclass
class Config:
    ## --- particle filter ---
    num_particles: int = 1
    seed: int = 1
    use_measurement_update: bool = True    # False: occupancy mapping only
    use_improved_proposal: bool = True     # False: FastSLAM 1.0 baseline
    resample_threshold: float = 1.0        # resample when N_eff < thr * N
    resample_every_step: bool = False

    ## --- motion model (4_grid_based_slam.ipynb) ---
    ## Standard deviations of the odometry increment, [rot1, trans, rot2]
    ## in [rad, m, rad]. THE FILTER'S BELIEF about how noisy the odometry is:
    ## it sets the particle spread in the prediction and shapes tau in the proposal.
    odometry_sigmas: Sequence[float] = (1.0, 1.0, 1.0)
    ## Simulation only, and the OPPOSITE role: noise injected into the
    ## odometry before the filter ever sees it, because Gazebo's odometry is
    ## too good to be interesting. Leave at zeros on hardware.
    odom_noise_sigma: Sequence[float] = (0.0, 0.0, 0.0)

    ## --- occupancy grid (1_grid_maps.ipynb) ---
    ## The prior p0 is fixed at 0.5 and is NOT tunable: that choice makes
    ## l_0 = logit(0.5) = 0, so an unobserved cell is exactly 0.0
    map_size_m: float = 20.0               # square map, centred on the start
    map_resolution: float = 1.0            # meters per cell
    p_free: float = 0.5                    # the notebook's P_FREE
    p_occ: float = 0.5                     # the notebook's P_HIT
    hit_window_cells: float = 0.0          # the notebook's R_WINDOW_CELLS.
    log_odds_limit: float = 1.0            # the notebook's LO_MAX
    use_clamping: bool = True

    ## --- measurement model (likelihood field) (tuned) ---
    ## range_max is NOT set here: it is read from LaserScan.range_max on the
    ## first scan, so the sensor is the single source of truth. The value
    ## below is only a placeholder until the first scan arrives.
    range_max: float = 0.0
    z_hit: float = 0.8                     # weight of the hit measurement component
    z_rand: float = 0.2                    # weight of the random measurement component
    sigma_hit: float = 0.10                # standard deviation of the hit measurement component
    max_dist: float = 0.15                 # likelihood field clip distance
    occ_thresh: float = 0.6                # p above which a cell is occupied

    ## Use every Nth hit beam when WEIGHTING a pose. 1 is every beam, which
    ## is the honest reading of the model.
    likelihood_stride: int = 1

    ## --- scan matching ---
    ## Use every Nth hit beam when MATCHING.
    scan_match_stride: int = 1
    ## One window per axis. The coarse and fine stages are derived from it,
    ## see `scan_match_stages` below.
    scan_match_window_xy: float = 1.0
    scan_match_window_theta: float = 1.0
    ## Below this correlation score the match is not trusted and the filter
    ## falls back to the plain odometry proposal.
    match_score_min: float = 0.00

    ## --- improved proposal ---
    num_candidates: int = 100               # K, rounded down to a perfect cube
    proposal_window_xy: float = 1.0
    proposal_window_theta: float = 1.0

    ## --- update gating (tuned) ---
    ## Upper bound on filter steps per second, 0 for no limit. This is the
    ## knob that decides the update rate; set it to the scanner's rate to
    ## step on every scan. The two below only keep a PARKED robot from
    ## burning the same scan into its map over and over, so they belong at
    ## the smallest motion still worth a step -- not at the motion you want
    ## between steps, which is what they meant before the rate limit existed.
    update_max_rate: float = 10.0
    update_min_dist: float = 0.05
    update_min_angle: float = 0.0175
    ## Depth of the /scan queue.
    scan_queue_depth: int = 1

    ## --- run log ---
    ## The node writes <log_dir>/<run_name>.npz at shutdown; every figure in
    ## the report is built from those files by `plot_results.py`.
    log_dir: str = 'runs'
    run_name: str = 'run'

    ## -- derived ---
    @property
    def l_free(self) -> float:
        """Log-odds increment for a cell the beam passed through.

        The prior is p = 0.5, so l_0 = 0 and the notebook's
        `logit(p) - L0` reduces to `logit(p)`.
        """
        return logit(self.p_free)

    @property
    def l_occ(self) -> float:
        """Log-odds increment for the cell the beam ended in."""
        return logit(self.p_occ)

    @property
    def hit_half_m(self) -> float:
        """Half the occupied window, in metres: the notebook's R_HALF.

        Cells within +-this distance of the measured range are pushed toward
        occupied; everything nearer is pushed toward free.
        """
        return 0.5 * self.hit_window_cells * self.map_resolution

    @property
    def map_origin(self) -> np.ndarray:
        """Bottom-left corner in metres. The map is square and centred on
        the robot's start pose, which is the origin of the odom frame."""
        return np.full(2, -0.5 * self.map_size_m)

    @property
    def scan_match_prior_sigmas(self):
        """(sd_trans, sd_rot) for the scan-match motion prior, in (m, rad).

        The scan-match prior answers the same question the motion model
        already answers -- how far can the true pose be from the odometry
        prediction -- so it is derived from `odometry_sigmas` rather than
        configured separately. Tightening the motion model therefore also
        tightens the scan match, which is what you would expect.

        The heading uncertainty over one increment is rot1 and rot2
        combined, hence the hypot.
        """
        s = np.asarray(self.odometry_sigmas, dtype=float)
        floor = 1e-4                       # never divide by zero
        return (max(float(s[1]), floor),
                max(float(np.hypot(s[0], s[2])), floor))

    @property
    def scan_match_stages(self):
        """((window_xy, step_xy, window_theta, step_theta), ...), coarse first.

        Each stage searches a window at half its own width, and the next one
        searches that step. Five values per axis, so 125 candidates a stage
        and 250 in total, ending at a step of a quarter of the window.

        Refining further is not free and buys nothing: `correlation` reads
        `probs()` at CELL CENTRES, so two candidates inside the same cell
        score identically however finely they are spaced, and the argmax
        between them is decided by array order. Keep the final step near the
        cell size and spend the candidates on a wider window instead.
        """
        w_xy, w_th = self.scan_match_window_xy, self.scan_match_window_theta
        return ((w_xy, w_xy / 2.0, w_th, w_th / 2.0),
                (w_xy / 2.0, w_xy / 4.0, w_th / 2.0, w_th / 4.0))


def config_from_node(node) -> Config:
    """Declare every Config field as a ROS parameter and read it back."""
    cfg = Config()
    for f in fields(Config):
        default = getattr(cfg, f.name)
        is_seq = isinstance(default, (tuple, list))
        node.declare_parameter(f.name, list(default) if is_seq else default)
        value = node.get_parameter(f.name).value
        setattr(cfg, f.name, tuple(value) if is_seq else value)
    return cfg
