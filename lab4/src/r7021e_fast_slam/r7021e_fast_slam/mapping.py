"""Occupancy mapping with known poses: no particles, no weights, no filter.

This is what `use_measurement_update: false` runs, and it is the whole of the
lab's first task -- drive the maze, watch the map smear as the odometry
drifts. It exists as its own module for two reasons:

  * it must work on a freshly unpacked package, before a single line of
    `motion_model.py`, `measurement_model.py`, `proposal.py`,
    `resampling.py` or `rbpf.py` has been written;
  * it must be testable without ROS installed.

The pose is the odometry pose the node already has, used verbatim. Nothing is
sampled, so the map degrades from odometry drift alone -- which is what the
task claims to show.
"""
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from .evaluation import StepInfo
from .grid_map import GridMap


class MappingOnlyFilter:
    """A stand-in for `GridFastSLAM` that only builds a map.

    It presents the three members the node uses of the real filter --
    `particles`, `best_particle()` and `step()` -- so nothing downstream has
    to know which of the two is running.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        ## A SimpleNamespace rather than `rbpf.Particle`: importing the
        ## student's filter here would defeat the purpose of this class.
        self.particles = [SimpleNamespace(pose=np.zeros(3), log_weight=0.0,
                                          grid=GridMap(cfg))]

    def best_particle(self):
        return self.particles[0]

    def step(self, pose, endpoints, ranges) -> StepInfo:
        """Integrate one scan at `pose`.

        Note the first argument is a POSE, not an odometry increment: there is
        no motion model here, which is the entire difference from `rbpf.step`.
        """
        p = self.particles[0]
        p.pose = np.asarray(pose, dtype=float).copy()
        t0 = perf_counter()
        p.grid.integrate_scan(p.pose, endpoints, ranges, self.cfg.range_max)
        return StepInfo(n_eff=1.0, timings={
            'scan_match': 0.0, 'likelihood': 0.0,
            'map_integrate': perf_counter() - t0})
