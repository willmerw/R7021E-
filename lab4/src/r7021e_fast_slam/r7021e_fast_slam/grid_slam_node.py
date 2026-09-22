"""ROS 2 wrapper around the Grid-FastSLAM 2.0 filter.

This is the only module that imports rclpy. It owns the subscriptions, the
update gating, the frame bookkeeping and the run log; the filter itself
knows nothing about ROS.

A scan and the odometry it is paired with must be read at the SAME instant.
Odometry samples are therefore kept in a short history and the one closest in
time to the scan's own stamp is used, rather than the newest pose being used
for whatever scan happens to come out of the queue.

The SLAM parameters live in `config.py` and are what `config/params.yaml`
sets. The dict below is the ROS wiring -- topics and frames --
"""
from bisect import bisect_left
from collections import deque

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from nav_msgs.msg import Path as PathMsg
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from sensor_msgs.msg import LaserScan

from .config import config_from_node
from .evaluation import RunLogger
from .mapping import MappingOnlyFilter
from .scan_utils import scan_to_endpoints
from .utils import (matrix_to_pose, odometry_increment, pose_to_matrix,
                    wrap_angle, yaw_from_quaternion, yaw_to_quaternion)


def _stamp_seconds(stamp) -> float:
    """A builtin_interfaces/Time as float seconds, in the message's own base."""
    return rclpy.time.Time.from_msg(stamp).nanoseconds * 1e-9


ROS_DEFAULTS = {
    'scan_topic': '/scan',
    'odom_topic': '/odom',
    'ground_truth_topic': '',          # empty: no external ground truth
    'map_frame': 'map',
    'odom_frame': 'odom',
    'base_frame': 'base_footprint',
    'laser_frame': 'base_scan',
    'laser_offset_xyt': [0.0, 0.0, 0.0],   # fallback if TF is unavailable
}

## /scan is best_effort on the robot and in the bag. The depth is the
## `scan_queue_depth` parameter, not a constant: 1 on hardware so a slow step
## drops the backlog instead of falling further behind, deep for bag replay
## where every scan must be processed.
MAP_PUBLISH_PERIOD = 0.1    # [s]
## Odometry samples kept for the lookup. At 50 Hz this is four seconds,
## far more than any filter step can fall behind before the queue drops it.
ODOM_HISTORY = 100


class GridSlamNode(Node):
    def __init__(self):
        super().__init__('grid_slam_node')
        for name, value in ROS_DEFAULTS.items():
            self.declare_parameter(name, value)
        ros = {k: self.get_parameter(k).value for k in ROS_DEFAULTS}

        self.cfg = config_from_node(self)
        self.rng = np.random.default_rng(int(self.cfg.seed))
        ## `GridFastSLAM` is imported lazily so that mapping-only mode runs on
        ## a package whose five student modules are still empty stubs.
        self.mapping_only = not self.cfg.use_measurement_update
        if self.mapping_only:
            self.filter = MappingOnlyFilter(self.cfg)
        else:
            from .rbpf import GridFastSLAM
            self.filter = GridFastSLAM(self.cfg, self.rng)

        self.map_frame = ros['map_frame']
        self.odom_frame = ros['odom_frame']
        self.base_frame = ros['base_frame']
        self.laser_frame = ros['laser_frame']
        self.base_T_laser = np.asarray(ros['laser_offset_xyt'], dtype=float)
        self._laser_tf_resolved = False
        self._range_max_from_scan = False

        self.odom_pose = None        # what the filter sees
        self.gt_pose = None          # clean pose, simulation only
        ## The newest RAW odometry pose. This is the one the odom -> base
        ## transform on the TF tree actually carries, which is why the
        ## map -> odom correction is computed against it and not against the
        ## corrupted pose the filter sees.
        self.odom_raw = None
        self.odom_origin = None      # first odom pose, defines the map frame
        self.gt_origin = None        # first external ground-truth pose
        self.ext_gt_pose = None
        self.last_update_pose = None
        ## Stamp of the newest odometry message. Everything this node
        ## publishes is stamped from the DATA, never from `now()`: with a bag
        ## replayed without `--clock` the node's own clock is wall time while
        ## the messages carry bag time, and a message stamped in the wrong
        ## time base cannot be transformed into any other frame.
        self.last_stamp = None
        ## map -> odom correction (x, y, theta). Updated at every filter step,
        ## rebroadcast at every odometry message; see `publish_tf`.
        self.map_T_odom = None
        self.corrupting = any(float(a) != 0.0
                              for a in self.cfg.odom_noise_sigma)
        ## (stamp [s], (2, 3) poses) of recent odometry, oldest first, for
        ## `_odom_at`. Row 0 is what the filter sees, row 1 the raw pose the
        ## robot's own odom -> base carries; they differ only in simulation,
        ## where this node corrupts the first. Both are needed at the scan's
        ## stamp: the filter consumes one and the TF correction the other.
        self.odom_hist = deque(maxlen=ODOM_HISTORY)
        self.last_update_time = None

        ## --- Publishers ---
        self.map_pub = self.create_publisher(
            OccupancyGrid, '/map',
            QoSProfile(depth=1, history=QoSHistoryPolicy.KEEP_LAST,
                       reliability=QoSReliabilityPolicy.RELIABLE,
                       durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
        self.particles_pub = self.create_publisher(PoseArray, '/particles', 10)
        self.path_pub = self.create_publisher(PathMsg, '/slam_path', 10)
        self.best_pub = self.create_publisher(Odometry, '/best_particle', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer(
            cache_time=rclpy.duration.Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.path_msg = PathMsg()
        self.path_msg.header.frame_id = self.map_frame

        ## --- Subscriptions ---
        scan_qos = QoSProfile(depth=max(int(self.cfg.scan_queue_depth), 1),
                              history=QoSHistoryPolicy.KEEP_LAST,
                              reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(LaserScan, ros['scan_topic'], self.scan_cb,
                                 scan_qos)
        self.create_subscription(Odometry, ros['odom_topic'], self.odom_cb, 100)
        ## Needed only if ground truth comes from a topic other than the clean
        ## /odom the node corrupts itself -- a Gazebo pose plugin, say.
        if ros['ground_truth_topic']:
            self.create_subscription(Odometry, ros['ground_truth_topic'],
                                     self.gt_cb, 100)

        self.create_timer(MAP_PUBLISH_PERIOD, self.publish_map)
        ## With use_sim_time and no /clock publisher the node's clock stays at
        ## zero, so the timer above never fires and nothing is ever published.
        ## That failure is silent, so check for it on the wall clock.
        if self.get_parameter('use_sim_time').value:
            self._clock_check = self.create_timer(
                5.0, self._warn_if_no_clock,
                clock=Clock(clock_type=ClockType.SYSTEM_TIME))

        self.logger = RunLogger(self.cfg)

        self.get_logger().info(
            f'grid_slam_node up: N={self.cfg.num_particles} '
            f'improved_proposal={self.cfg.use_improved_proposal} '
            f'measurement_update={self.cfg.use_measurement_update} '
            f'seed={self.cfg.seed}')
        ## Say where the run log will land before the run starts, not after.
        self.get_logger().info(
            f'run log -> {self.logger.dir / (self.cfg.run_name + ".npz")}')

    ## --- helpers ---
    @staticmethod
    def _pose_from_odom(msg) -> np.ndarray:
        return np.array([msg.pose.pose.position.x,
                         msg.pose.pose.position.y,
                         yaw_from_quaternion(msg.pose.pose.orientation)])

    def _resolve_laser_tf(self):
        """Read base->laser from TF once; keep the parameter as a fallback."""
        if self._laser_tf_resolved:
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.laser_frame, rclpy.time.Time())
        except tf2_ros.TransformException:
            return
        self.base_T_laser = np.array([
            tf.transform.translation.x, tf.transform.translation.y,
            yaw_from_quaternion(tf.transform.rotation)])
        self._laser_tf_resolved = True
        self.get_logger().info(
            f'{self.base_frame} -> {self.laser_frame} = {self.base_T_laser}')

    @staticmethod
    def _rebase(pose, origin) -> np.ndarray:
        """Express `pose` relative to `origin`.

        The particles start at the origin at the first odometry message, so
        the filter's map frame is the odometry frame with that first pose
        moved to (0, 0, 0). A bag recorded after the robot had already been
        driving starts its odometry at an arbitrary pose, so anything logged
        for comparison against the estimate has to be rebased or the two
        trajectories live in different frames.
        """
        if origin is None:
            return np.asarray(pose, dtype=float)
        return matrix_to_pose(np.linalg.inv(pose_to_matrix(origin))
                              @ pose_to_matrix(pose))

    def _corrupt(self, pose, u):
        """Apply one noisy odometry increment to `pose`. Simulation only.

        Deliberately NOT `motion_model.sample_motion_model_odometry`: the noise
        that defines the experiment must not depend on the code being graded,
        or a bug in the motion model would silently change the ground truth it
        is measured against.
        """
        sd = np.asarray(self.cfg.odom_noise_sigma, dtype=float)
        r1, tr, r2 = np.asarray(u, dtype=float) + self.rng.normal(0.0, sd)
        heading = pose[2] + r1
        return np.array([pose[0] + tr * np.cos(heading),
                         pose[1] + tr * np.sin(heading),
                         float(wrap_angle(heading + r2))])

    def _odom_at(self, t: float):
        """(filter pose, raw pose) of the sample CLOSEST to `t`, or None."""
        hist = self.odom_hist
        if not hist:
            return None
        i = bisect_left([s for s, _ in hist], t)
        if i == 0:
            return hist[0][1].copy()
        if i == len(hist):
            return hist[-1][1].copy()
        (t0, p0), (t1, p1) = hist[i - 1], hist[i]
        return (p0 if t - t0 <= t1 - t else p1).copy()

    def _gated(self, pose, t: float) -> bool:
        """True when this scan should drive a filter step.

        `update_max_rate` sets the rate; the distance and angle minimums
        below it only suppress steps while the robot is parked, where a step
        would re-integrate the same scan and let the map sharpen on nothing.
        """
        if self.last_update_pose is None:
            return True
        rate = float(self.cfg.update_max_rate)
        if (rate > 0.0 and self.last_update_time is not None
                and t - self.last_update_time < 1.0 / rate):
            return False
        d = np.linalg.norm(pose[:2] - self.last_update_pose[:2])
        a = abs(wrap_angle(pose[2] - self.last_update_pose[2]))
        return d >= self.cfg.update_min_dist or a >= self.cfg.update_min_angle

    def _warn_if_no_clock(self):
        self._clock_check.cancel()
        if self.count_publishers('/clock') == 0:
            self.get_logger().warn(
                'use_sim_time is true but nothing publishes /clock, so this '
                'node is frozen and will publish nothing. Replay the bag with '
                '`ros2 bag play --clock`, or launch with use_sim_time:=false.')

    ## --- callbacks ---
    def gt_cb(self, msg):
        self.ext_gt_pose = self._pose_from_odom(msg)
        if self.gt_origin is None:
            self.gt_origin = self.ext_gt_pose.copy()

    def odom_cb(self, msg):
        raw = self._pose_from_odom(msg)
        if self.odom_raw is None:
            self.odom_raw = raw
            self.odom_origin = raw.copy()
            self.odom_pose = raw.copy()
            self.gt_pose = raw.copy()
            return

        self.gt_pose = raw.copy()
        if not self.corrupting:
            ## Hardware: the filter sees the odometry exactly as published.
            self.odom_pose = raw.copy()
        else:
            ## Simulation: Gazebo odometry is near-perfect, which would make
            ## the lab trivial. Corrupt the increment; the clean pose is
            ## logged as ground truth and never reaches the filter.
            self.odom_pose = self._corrupt(
                self.odom_pose, odometry_increment(self.odom_raw, raw))
        self.odom_raw = raw
        self.last_stamp = msg.header.stamp
        self.odom_hist.append((_stamp_seconds(msg.header.stamp),
                               np.stack([self.odom_pose, raw])))

        ## Keep map -> odom as fresh as odom -> base, so RViz never has to
        ## extrapolate. Before the first filter step the estimate is still the
        ## origin, which seeds the correction with the inverse of the first
        ## odometry pose, and it makes the map frame exist from the
        ## very first message instead of from the first update.
        if self.map_T_odom is None:
            self.publish_tf(np.zeros(3), msg.header.stamp)
        else:
            self._send_tf(msg.header.stamp)

    def scan_cb(self, msg):
        if self.odom_pose is None:
            return
        self._resolve_laser_tf()
        if not self._range_max_from_scan:
            ## The scanner is the single source of truth for its own range.
            self.cfg.range_max = float(msg.range_max)
            self._range_max_from_scan = True
            self.get_logger().info(
                f'range_max = {self.cfg.range_max:.2f} m, from LaserScan')
        ## Everything below reads the odometry as it was WHEN THIS SCAN WAS
        ## TAKEN, never as it is now; see `_odom_at`.
        t = _stamp_seconds(msg.header.stamp)
        at_scan = self._odom_at(t)
        if at_scan is None:
            return
        odom_pose, odom_raw = at_scan
        if not self._gated(odom_pose, t):
            return

        prev = (self.last_update_pose if self.last_update_pose is not None
                else odom_pose)
        u = odometry_increment(prev, odom_pose)
        self.last_update_pose = odom_pose.copy()
        self.last_update_time = t

        endpoints, ranges, _ = scan_to_endpoints(msg, self.base_T_laser)
        ## MappingOnlyFilter maps at a known pose, so it is handed the
        ## odometry pose itself; the filter is handed the increment.
        info = self.filter.step(
            self._rebase(odom_pose, self.odom_origin)
            if self.mapping_only else u, endpoints, ranges)

        best = self.filter.best_particle()
        stamp = msg.header.stamp
        self.publish_tf(best.pose, stamp, odom_raw)
        self.publish_particles(stamp)
        self.publish_path(best.pose, stamp)
        self.publish_best_particle(best.pose, stamp)

        ## est is in the filter's frame, so odom and ground truth must be too.
        if self.ext_gt_pose is not None:
            gt = self._rebase(self.ext_gt_pose, self.gt_origin)
        elif self.corrupting:
            gt = self._rebase(self.gt_pose, self.odom_origin)
        else:
            gt = None
        self.logger.add(t, best.pose,
                        self._rebase(odom_pose, self.odom_origin),
                        gt, info,
                        sum(p.grid.nbytes() for p in self.filter.particles))

    ## --- publishing ---
    def publish_tf(self, est_pose, stamp, odom_raw=None):
        """Recompute and broadcast the map -> odom correction.

        `odom_raw` is the raw odometry pose AT THE SAME INSTANT as `est_pose`,
        picked by `_odom_at`; it defaults to the newest sample for the
        seeding call, which has no estimate to be consistent with yet. Pairing
        an estimate with newer odometry than it was computed from would leave
        the whole map lagging the robot by one filter step.

        map -> odom -> base composes to the estimate, which is why the
        correction is the inverse of the odometry pose times that estimate.

        That inverse must use the RAW odometry pose, because odom -> base is
        published by the robot (or by Gazebo) and carries the raw pose --
        never the corrupted one this node invents for the filter. Using the
        corrupted pose here cancels the corruption back out of the TF tree:
        RViz would then draw /scan at the true pose while the map was built at
        the corrupted one, and the live scan would float off the walls it just
        wrote by exactly the injected noise.
        """
        if odom_raw is None:
            odom_raw = self.odom_raw
        t_map_odom = pose_to_matrix(est_pose) @ np.linalg.inv(
            pose_to_matrix(odom_raw))
        self.map_T_odom = matrix_to_pose(t_map_odom)
        self._send_tf(stamp)

    def _send_tf(self, stamp):
        """Broadcast the cached map -> odom correction at `stamp`."""
        x, y, th = self.map_T_odom
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self.map_frame
        tf.child_frame_id = self.odom_frame
        tf.transform.translation.x = float(x)
        tf.transform.translation.y = float(y)
        tf.transform.rotation = yaw_to_quaternion(th)
        self.tf_broadcaster.sendTransform(tf)

    def publish_particles(self, stamp):
        msg = PoseArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame
        for particle in self.filter.particles:
            pose = Pose()
            pose.position.x = float(particle.pose[0])
            pose.position.y = float(particle.pose[1])
            pose.orientation = yaw_to_quaternion(float(particle.pose[2]))
            msg.poses.append(pose)
        self.particles_pub.publish(msg)

    def publish_best_particle(self, est_pose, stamp):
        """The heaviest particle as an Odometry message, for visualisation.

        Same type as /odom so the two can be overlaid in RViz or plotted
        against each other. The pose is the best particle;
        the covariance is the spread of the WHOLE weighted cloud, which is
        what RViz draws as an ellipse. 
        Twist is left at zero -- the filter does not estimate velocity,
        and filling it with a made-up number would be worse than a zero.
        """
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame
        msg.child_frame_id = self.base_frame
        msg.pose.pose.position.x = float(est_pose[0])
        msg.pose.pose.position.y = float(est_pose[1])
        msg.pose.pose.orientation = yaw_to_quaternion(float(est_pose[2]))

        poses = np.array([p.pose for p in self.filter.particles])
        w = np.exp(np.array([p.log_weight for p in self.filter.particles]))
        w = w / w.sum() if w.sum() > 0 else np.full(w.size, 1.0 / w.size)

        d = poses[:, :2] - (w @ poses[:, :2])
        cov_xy = (d * w[:, None]).T @ d
        mean_yaw = np.arctan2(w @ np.sin(poses[:, 2]), w @ np.cos(poses[:, 2]))
        d_yaw = wrap_angle(poses[:, 2] - mean_yaw)
        cov = np.zeros((6, 6))
        cov[:2, :2] = cov_xy
        cov[5, 5] = float(w @ (d_yaw ** 2))
        cov[0, 5] = cov[5, 0] = float(w @ (d[:, 0] * d_yaw))
        cov[1, 5] = cov[5, 1] = float(w @ (d[:, 1] * d_yaw))
        cov[np.diag_indices(6)] += 1e-9
        msg.pose.covariance = cov.ravel().tolist()
        self.best_pub.publish(msg)

    def publish_path(self, est_pose, stamp):
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self.map_frame
        ps.pose.position.x = float(est_pose[0])
        ps.pose.position.y = float(est_pose[1])
        ps.pose.orientation = yaw_to_quaternion(float(est_pose[2]))
        self.path_msg.poses.append(ps)
        self.path_msg.header.stamp = stamp
        self.path_pub.publish(self.path_msg)

    def publish_map(self):
        if self.last_stamp is None:
            return
        best = self.filter.best_particle()
        self.map_pub.publish(
            best.grid.to_occupancy_grid(self.map_frame, self.last_stamp))

    def shutdown(self):
        self.logger.set_final_map(self.filter.best_particle().grid)
        path = self.logger.save()
        if path is not None:
            self.get_logger().info(f'run log written to {path}')


def main(args=None):
    rclpy.init(args=args)
    node = GridSlamNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        ## Ctrl-C and SIGTERM alike: shut down quietly and still save the run
        ## log. Without the second case a SIGTERM'd node dies with a traceback
        ## and exit 1, which makes `ros2 launch` teardown look like a crash.
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
