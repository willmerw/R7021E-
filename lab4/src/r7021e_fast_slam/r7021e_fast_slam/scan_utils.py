"""LaserScan handling: validity and conversion to beam endpoints.

Everything downstream of this module works with Cartesian beam endpoints in
the robot's base frame, never with the raw LaserScan.

A scan contains two kinds of useful beam and one useless one:

  hit         a real range reading. It says "free up to here, occupied here".
  no-return   the beam went out and nothing came back, because the nearest
              surface is beyond range_max. It says "free all the way out to
              range_max", which is weaker evidence but not no evidence: it is
              how open space gets mapped at all. Reported as inf.
  invalid     a reading below range_min, ZERO INCLUDED. The LDS-02 reports a
              dropout -- a surface too dark or too oblique to reflect -- as
              0.0, and it is indistinguishable from a genuine out-of-range
              return.

`scan_to_endpoints` returns the first two, with a no-return placed at
range_max. Callers tell them apart by the returned range: a no-return has
`range == scan.range_max`, which is what `GridMap.integrate_scan` tests.
"""
import numpy as np

from .utils import transform_points


def classify(ranges, range_min: float, range_max: float):
    """Split beams into (hit, usable), both boolean masks over the raw scan.

    `hit` marks a real range reading. `usable` additionally includes the
    no-returns, and excludes every invalid reading -- below range_min, zero
    included.
    """
    r = np.asarray(ranges, dtype=float)
    with np.errstate(invalid='ignore'):
        hit = np.isfinite(r) & (r >= range_min) & (r <= range_max)
        ## Zero is included here, and that is the whole point: see `invalid`
        ## in the module docstring.
        invalid = np.isfinite(r) & (r < range_min)
    return hit, ~invalid


def scan_to_endpoints(scan, base_T_laser):
    """Convert a LaserScan to beam endpoints in the base frame.

    scan:         object with ranges, angle_min, angle_increment,
                  range_min, range_max (a sensor_msgs/LaserScan, or a stub).
    base_T_laser: (3,) pose of the laser in the base frame.

    Returns (endpoints (B, 2) in base frame, ranges (B,), angles (B,) in the
    laser frame) for every usable beam. A no-return beam is reported at
    range_max, so its endpoint is NOT an obstacle observation -- use it for
    mapping, never for scoring a pose.
    """
    r = np.asarray(scan.ranges, dtype=float)
    ang = scan.angle_min + np.arange(r.size) * scan.angle_increment

    hit, usable = classify(r, scan.range_min, scan.range_max)
    r = np.where(hit, r, scan.range_max)[usable]
    ang = ang[usable]

    pts_laser = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
    if pts_laser.shape[0] == 0:
        return np.zeros((0, 2)), r, ang
    pts_base = transform_points(base_T_laser, pts_laser)[0]
    return pts_base, r, ang
