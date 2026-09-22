"""Small geometry and numerics helpers shared by the whole package.

Nothing in here knows about ROS message types except the quaternion
converters, which are the boundary between the filter's (x, y, theta)
convention and the ROS pose convention.
"""
import numpy as np

def wrap_angle(a):
    """Wrap an angle, or an array of angles, to [-pi, pi)."""
    return (np.asarray(a, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def logit(p, eps: float = 1e-6) -> float:
    """log(p / (1 - p)), clipped so that p = 0 or 1 does not give +-inf.

    The log-odds representation of the occupancy grid: it turns the products
    of the binary Bayes filter into sums. Same helper as in hello-slam.
    """
    p = np.clip(p, eps, 1.0 - eps)
    return float(np.log(p / (1.0 - p)))


## Constant shared by `odometry_increment` and the motion model: below this
## translation the heading of the motion is numerically meaningless.
TRANS_EPS = 1e-6


def odometry_increment(pose_prev, pose_cur) -> np.ndarray:
    """Decompose two consecutive odometry poses into (rot1, trans, rot2).

    Pure SE(2) bookkeeping, so it is provided: turn to face the motion, drive,
    turn to the final heading. The modelling work -- putting noise on these
    three numbers and evaluating the resulting density -- is `motion_model.py`.

    Args:
        pose_prev: (3,) previous odometry pose [x, y, theta]
        pose_cur:  (3,) current odometry pose [x, y, theta]

    Returns:
        (3,) [d_rot1, d_trans, d_rot2].

    When the robot barely translates (TRANS_EPS), the heading of the motion is
    numerically meaningless, so the whole rotation is attributed to rot2.
    Without that guard a stationary robot produces a rot1 driven purely by
    odometry jitter.
    """
    pose_prev = np.asarray(pose_prev, dtype=float)
    pose_cur = np.asarray(pose_cur, dtype=float)
    dx = pose_cur[0] - pose_prev[0]
    dy = pose_cur[1] - pose_prev[1]
    d_trans = float(np.hypot(dx, dy))
    if d_trans < TRANS_EPS:
        d_rot1 = 0.0
    else:
        d_rot1 = float(wrap_angle(np.arctan2(dy, dx) - pose_prev[2]))
    d_rot2 = float(wrap_angle(pose_cur[2] - pose_prev[2] - d_rot1))
    return np.array([d_rot1, d_trans, d_rot2])


def yaw_from_quaternion(q) -> float:
    """Yaw of a geometry_msgs Quaternion, assuming planar motion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def yaw_to_quaternion(yaw: float):
    """geometry_msgs Quaternion for a rotation of `yaw` about +Z."""
    from geometry_msgs.msg import Quaternion
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = float(np.sin(0.5 * yaw))
    q.w = float(np.cos(0.5 * yaw))
    return q


def pose_to_matrix(pose) -> np.ndarray:
    """SE(2) homogeneous matrix for pose = (x, y, theta)."""
    x, y, th = np.asarray(pose, dtype=float)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def matrix_to_pose(T) -> np.ndarray:
    """Inverse of `pose_to_matrix`."""
    return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])])


def transform_points(poses, pts) -> np.ndarray:
    """Transform 2-D points by one or many SE(2) poses.

    poses: (3,) or (K, 3).  pts: (B, 2).  Returns (K, B, 2).
    Broadcasting over K is what lets the measurement model score a whole
    batch of candidate poses without a Python loop.
    """
    poses = np.atleast_2d(np.asarray(poses, dtype=float))
    pts = np.asarray(pts, dtype=float)
    c = np.cos(poses[:, 2])[:, None]
    s = np.sin(poses[:, 2])[:, None]
    x = pts[None, :, 0]
    y = pts[None, :, 1]
    out = np.empty((poses.shape[0], pts.shape[0], 2), dtype=float)
    out[..., 0] = c * x - s * y + poses[:, 0][:, None]
    out[..., 1] = s * x + c * y + poses[:, 1][:, None]
    return out


## --- numerics that used to come from SciPy ---

def expit(x):
    """Logistic sigma(x), written so that large |x| cannot overflow.

    `0.5 * (1 + tanh(x / 2))` is algebraically identical to
    `1 / (1 + exp(-x))` and saturates instead of overflowing, which matters
    with `use_clamping: false` where the log-odds are unbounded.
    """
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float64)))


def logsumexp(a) -> float:
    """log(sum(exp(a))), shifted by max(a) so that exp() cannot overflow."""
    a = np.asarray(a, dtype=float)
    m = float(np.max(a))
    if not np.isfinite(m):
        return m                      # all -inf (or an inf): nothing to shift
    return m + float(np.log(np.sum(np.exp(a - m))))


def _shift(a, k: int, axis: int, fill):
    """`out[i] = a[i + k]` along `axis`, with `fill` where that runs off."""
    out = np.full_like(a, fill)
    n = a.shape[axis]
    if abs(k) >= n:
        return out
    dst = [slice(None)] * a.ndim
    src = [slice(None)] * a.ndim
    dst[axis] = slice(max(-k, 0), n - max(k, 0))
    src[axis] = slice(max(k, 0), n + min(k, 0))
    out[tuple(dst)] = a[tuple(src)]
    return out


def distance_transform(mask, res: float, max_dist: float) -> np.ndarray:
    """Euclidean distance in metres from every cell to the nearest True cell.

    Clipped at `max_dist`, which is what makes it cheap: only cells within
    R = ceil(max_dist / res) can matter, so the exact separable transform
    reduces to 2R + 1 shifted minima per axis. Inside the clip the result is
    the same as `scipy.ndimage.distance_transform_edt` on the complement.
    """
    mask = np.asarray(mask, dtype=bool)
    r = int(np.ceil(max_dist / res))
    inf = (r + 1) ** 2                # one cell beyond anything we keep

    ## Pass 1: squared distance, in cells, to the nearest True cell in the
    ## same column. Pass 2: combine columns, which is exact because the
    ## squared Euclidean distance is separable.
    g = np.full(mask.shape, inf, dtype=np.int64)
    for k in range(-r, r + 1):
        np.minimum(g, np.where(_shift(mask, k, 0, False), k * k, inf), out=g)
    d2 = np.full(mask.shape, inf, dtype=np.int64)
    for k in range(-r, r + 1):
        np.minimum(d2, _shift(g, k, 1, inf) + k * k, out=d2)

    return np.minimum(np.sqrt(d2) * res, max_dist).astype(np.float32)
