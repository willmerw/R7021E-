"""Occupancy grid in log-odds -- the inverse sensor model of hello-slam's
`1_grid_maps.ipynb`, plus the likelihood field used to score poses.

Each cell holds l = log(p / (1 - p)) for "this cell is occupied". The prior is
p = 0.5, so an unobserved cell is exactly 0.0 and the recursion of the
static-state binary Bayes filter is a plain sum:

    l_t,i = l_t-1,i + inv_sensor_model(m_i, x_t, z_t) - l_0,   l_0 = 0.

One of these belongs to each particle, so `copy()` being a genuine deep copy
is what makes resampling correct, and its cost is what makes the number of
particles expensive.
"""
import copy as copy_module

import numpy as np

from .utils import distance_transform, expit

## How a no-return beam is treated. It means only that nothing came back, and
## that has two causes the sensor cannot tell apart: the space really is empty,
## or a surface is there which is too dark or too steeply angled to reflect.
## So it carries free-space evidence, but discounted, and not all the way out
## to range_max -- the far end of a no-return is exactly where an unseen
## surface is most likely to be. Fixed here rather than exposed as parameters:
## they describe the sensor, not the map.
NO_RETURN_FREE_SCALE = 0.5      # fraction of l_free deposited per cell
NO_RETURN_REACH_SCALE = 0.75    # fraction of range_max marked free at all


class GridMap:
    def __init__(self, cfg):
        self.res = float(cfg.map_resolution)
        self.origin = np.asarray(cfg.map_origin, dtype=float)
        self.W = int(round(cfg.map_size_m / self.res))
        self.H = self.W
        ## Increments derived from the probabilities of the inverse sensor
        ## model, not configured directly. See Config.l_free / Config.l_occ.
        self.l_free = float(cfg.l_free)
        self.l_occ = float(cfg.l_occ)
        self.l_limit = float(cfg.log_odds_limit)
        self.r_half = float(cfg.hit_half_m)   # the notebook's R_HALF
        self.use_clamping = bool(cfg.use_clamping)
        self.log_odds = np.zeros((self.H, self.W), dtype=np.float32)

    ## --- derived views ---
    def probs(self) -> np.ndarray:
        """sigma(log_odds): the occupancy probability of every cell.

        `expit` rather than `1 / (1 + exp(-l))`: the naive form overflows for
        large |l| and prints a RuntimeWarning on every call. With
        `use_clamping: false` the log-odds are unbounded, so that is exactly
        the configuration the lab asks you to compare against.
        """
        return expit(self.log_odds).astype(np.float32)

    def likelihood_field(self, occ_thresh: float, max_dist: float) -> np.ndarray:
        """Distance in metres from each cell to the nearest occupied cell.

        Clipped at `max_dist`. The Euclidean distance transform runs over the
        whole grid on every call, once per particle per filter step.
        """
        occ = self.probs() > occ_thresh
        if not occ.any():
            return np.full((self.H, self.W), max_dist, dtype=np.float32)
        return distance_transform(occ, self.res, max_dist)

    ## --- indexing ---
    def world_to_grid(self, pts) -> np.ndarray:
        """(..., 2) world metres -> (..., 2) int cells, [.., 0]=col, [.., 1]=row."""
        pts = np.asarray(pts, dtype=float)
        return np.floor((pts - self.origin) / self.res).astype(np.int32)

    def in_bounds(self, cells) -> np.ndarray:
        cells = np.asarray(cells)
        gx, gy = cells[..., 0], cells[..., 1]
        return (gx >= 0) & (gx < self.W) & (gy >= 0) & (gy < self.H)

    ## --- inverse sensor model ---
    def integrate_scan(self, pose, endpoints, ranges, max_range: float):
        """Mapping with known poses, via the step-window inverse sensor model.

        For each beam of measured range z, a cell at distance s along the ray
        gets (1_grid_maps.ipynb):

            s <  z - r/2   ->  free      (l_free)
            s <= z + r/2   ->  occupied  (l_occ)
            s >  z + r/2   ->  no information, keep the prior

        with r = 2 * self.r_half. A beam that returned nothing saw no obstacle,
        so it carries free-space evidence and never an occupied mark -- but
        discounted by NO_RETURN_FREE_SCALE, and only out to
        NO_RETURN_REACH_SCALE of its range.

        pose:      (3,) robot pose in the map frame.
        endpoints: (B, 2) beam endpoints in the robot base frame.
        ranges:    (B,) raw beam ranges, used only to tell a hit from a
                   max-range return.
        max_range: the scanner's maximum range, from LaserScan.range_max.
        """
        endpoints = np.asarray(endpoints, dtype=float)
        if endpoints.shape[0] == 0:
            return

        x, y, th = np.asarray(pose, dtype=float)
        bearing = np.arctan2(endpoints[:, 1], endpoints[:, 0]) + th
        reach = np.hypot(endpoints[:, 0], endpoints[:, 1])
        cos_b, sin_b = np.cos(bearing), np.sin(bearing)
        hit = np.asarray(ranges, dtype=float) < (max_range - 1e-6)

        ## March every ray at half-cell steps, all beams at once. The march has
        ## to run past the endpoint to cover the far half of the hit window.
        step = 0.5 * self.res
        n_steps = int(np.ceil((reach.max() + self.r_half) / step)) + 1
        t = np.arange(n_steps) * step                            # (S,)
        px = x + t[None, :] * cos_b[:, None]                     # (B, S)
        py = y + t[None, :] * sin_b[:, None]
        gx = np.floor((px - self.origin[0]) / self.res).astype(np.int64)
        gy = np.floor((py - self.origin[1]) / self.res).astype(np.int64)

        inb = (gx >= 0) & (gx < self.W) & (gy >= 0) & (gy < self.H)
        flat = gy * self.W + gx

        ## The step window, exactly as in the notebook's p_occ_beam.
        T = t[None, :]
        z = reach[:, None]
        ## A hit frees everything up to the near edge of its window. A
        ## no-return frees only the near part of its reach: the far end is
        ## precisely where a surface it failed to see would be.
        free = np.where(hit[:, None], T < z - self.r_half,
                        T < NO_RETURN_REACH_SCALE * z)
        occ = hit[:, None] & (T <= z + self.r_half) & ~free

        ## A cell is sampled two or three times at half-cell steps, so each one
        ## has to be collapsed to a single update per ray -- otherwise a cell
        ## the ray merely crosses at a shallow angle gets several times the
        ## evidence of one it crosses head on.
        ##
        ## Collapse per LABEL, not just per cell. The classification is
        ## monotone in t (a run of free, then the hit window), so exactly one
        ## cell per ray can straddle the boundary, and in that cell the
        ## occupied evidence has to win: it is the cell the wall is in.
        ## Taking the first sample of the cell instead marks the wall FREE and
        ## drops the hit entirely, which erases walls seen at ranges whose
        ## hit window happens to open mid-cell.
        prev_same = np.zeros(flat.shape, dtype=bool)
        prev_same[:, 1:] = flat[:, 1:] == flat[:, :-1]

        def _first_of_run(label):
            """Drop every sample whose predecessor was the same cell AND
            carried the same label."""
            prev_label = np.zeros(label.shape, dtype=bool)
            prev_label[:, 1:] = label[:, :-1]
            return label & ~(prev_same & prev_label)

        occ_mark = _first_of_run(occ)
        free_mark = _first_of_run(free)

        ## Clear the free mark on the straddling cell: the one holding the
        ## first sample of the hit window.
        rows = np.arange(flat.shape[0])
        straddle = np.where(occ.any(axis=1), flat[rows, occ.argmax(axis=1)], -1)
        free_mark &= flat != straddle[:, None]

        ## Discount the free evidence of a no-return: with the full weight of
        ## a hit beam, a handful of them erases a wall the scanner has simply
        ## stopped seeing.
        lo = self.log_odds.ravel()
        is_hit = hit[:, None]
        np.add.at(lo, flat[inb & free_mark & is_hit], np.float32(self.l_free))
        np.add.at(lo, flat[inb & free_mark & ~is_hit],
                  np.float32(NO_RETURN_FREE_SCALE * self.l_free))
        np.add.at(lo, flat[inb & occ_mark], np.float32(self.l_occ))

        if self.use_clamping:
            np.clip(self.log_odds, -self.l_limit, self.l_limit,
                    out=self.log_odds)

    def copy(self) -> 'GridMap':
        """Deep copy of the map.

        Resampling duplicates particles, and two particles must never share a
        grid, so the log-odds array is genuinely copied. That copy is the
        dominant cost of a large particle set.
        """
        other = copy_module.copy(self)      # scalars are fine to share
        other.log_odds = self.log_odds.copy()
        other.origin = self.origin.copy()
        return other

    def nbytes(self) -> int:
        return int(self.log_odds.nbytes)

    ## --- ROS integration ---
    def to_occupancy_grid(self, frame_id: str, stamp):
        from nav_msgs.msg import OccupancyGrid
        msg = OccupancyGrid()
        msg.header.frame_id = frame_id
        msg.header.stamp = stamp
        msg.info.resolution = self.res
        msg.info.width = self.W
        msg.info.height = self.H
        msg.info.origin.position.x = float(self.origin[0])
        msg.info.origin.position.y = float(self.origin[1])
        msg.info.origin.orientation.w = 1.0
        data = np.full((self.H, self.W), -1, dtype=np.int8)
        known = self.log_odds != 0.0     # l_0 = 0, so 0 means never observed
        data[known] = np.round(100.0 * self.probs()[known]).astype(np.int8)
        msg.data = data.ravel().tolist()
        return msg
