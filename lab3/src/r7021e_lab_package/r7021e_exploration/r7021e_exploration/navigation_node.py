code#!/usr/bin/env python3

import math
from typing import Optional, Tuple, List

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Path, OccupancyGrid
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.time import Time

import tf2_ros
import numpy as np

from sklearn.cluster import DBSCAN

def quat_to_yaw(q: Quaternion) -> float:
    """Extract planar yaw (rad) from a Quaternion."""
    x, y, z, w = q.x, q.y, q.z, q.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion(yaw: float) -> Quaternion:
    """Convert planar yaw (rad) to Quaternion."""
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q

class TreeNode:
    def __init__(self, x, y, parent):
        self.pt = np.array([x,y])
        self.children = []
        self.parent = parent

    def get_closest_node(self, pt):
        pt = np.array(pt)

        closest_dist = np.linalg.norm(pt-self.pt)
        closest_node = self

        for node in self.children:
            child_dist, child_node = node.get_closest_node(pt)
            if child_dist < closest_dist:
                closest_dist = child_dist
                closest_node = child_node

        return closest_dist, closest_node

    def get_path_to_root(self):
        path = [self.pt]

        if self.parent == None:
            return path
        path = self.parent.get_path_to_root() + path
        return path



class PathPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__('path_planner_node')

        # Parameters
        self.declare_parameter('map_topic', 'map')
        self.declare_parameter('frontier_topic', 'frontier')
        self.declare_parameter('path_topic', 'path')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tf_timeout_sec', 0.5)

        map_topic: str = self.get_parameter('map_topic').get_parameter_value().string_value
        frontier_topic: str = self.get_parameter('frontier_topic').get_parameter_value().string_value
        path_topic: str = self.get_parameter('path_topic').get_parameter_value().string_value
        self.global_frame: str = self.get_parameter('global_frame').get_parameter_value().string_value
        self.base_frame: str = self.get_parameter('base_frame').get_parameter_value().string_value
        self.tf_timeout = Duration(seconds=self.get_parameter('tf_timeout_sec').get_parameter_value().double_value)

        default_qos = QoSProfile(depth=10)

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.node_max_dist = 1.0
        self.node_radius = 1.0 # for rrt-star
        self.obs_fid = 0.1 #obstacle fidelity
        self.goal_radius = 0.2

        self.tree = TreeNode(self.x,self.y, None)

        # Publishers (add path publisher here)
        self.path_pub = self.create_publisher(Path, 'path', 10)

        # Subscribers
        self.map_sub = self.create_subscription(OccupancyGrid, map_topic, self._on_map, default_qos)
        self.frontier_sub = self.create_subscription(OccupancyGrid, frontier_topic, self._on_frontier, default_qos)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # State
        self._latest_map: Optional[OccupancyGrid] = None
        self._latest_frontier: Optional[OccupancyGrid] = None

        self.get_logger().info('PathPlannerNode initialized.')

    # ----------------- TF Helper -----------------

    def get_robot_pose(self, target_frame: Optional[str] = None, source_frame: Optional[str] = None
                       ) -> Optional[Tuple[float, float, float]]:
        """
        Lookup TF to get robot pose (x, y, yaw) in target_frame.
        Returns None if not available within timeout.
        """
        tgt = target_frame or self.global_frame
        src = source_frame or self.base_frame
        self.plan_path((x, y, yaw),
                              msg,
                              self._latest_frontier)
        try:
            transform = self.tf_buffer.lookup_transform(
                tgt, src, Time(), timeout=self.tf_timeout
            )
        except Exception as e:
            self.get_logger().warn(f'TF lookup {tgt} <- {src} failed: {e}')
            return None

        t = transform.transform.translation
        r = transform.transform.rotation
        yaw = quat_to_yaw(r)

        return (t.x, t.y, yaw)

    # ----------------- Callbacks -----------------

    def _on_frontier(self, msg: OccupancyGrid) -> None:
        """Store the latest frontier map (could be used by your planner)."""
        self._latest_frontier = msg

    def _on_map(self, msg: OccupancyGrid) -> None:
        """Trigger planning when a new map arrives."""
        self._latest_map = msg

        pose = self.get_robot_pose()
        if pose is None:
            return

        x, y, yaw = pose
        path = self.plan_path((x, y, yaw),
                              msg,
                              self._latest_frontier)

        if path is None:
            return

    # ----------------- Planning -----------------

    def plan_path(self,
                  start: Tuple[float, float, float],
                  map_msg: OccupancyGrid,
                  frontier_msg: Optional[OccupancyGrid]):
        """
        Use this function to plan the path
        """

        map_res = map_msg.info.resolution
        map_height = map_msg.info.height * map_res
        map_width = map_msg.info.width * map_res
        map_values = map_msg.data

        w_ig = 1.0 #information gain weight
        w_d = 1.0 #distace weight
        w_th = 1.0 #heading weight

        frontier_grid = np.array(frontier_msg, dtype=np.int16).reshape(
            (frontier_msg.info.height, frontier_msg.info.width))

        rows, cols = np.where(frontier_grid == 100)

        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        x_coords = origin_x + (cols + 0.5) * resolution
        y_coords = origin_y + (rows + 0.5) * resolution
        points = np.column_stack((x_coords, y_coords))
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = db.labels_

        cluster_count = {}

        for label in labels:
            if label == -1:
                continue
            if label not in labels:
                cluster_count[label] = 1
                continue
            cluster_count[label] += 1

        best_c = -1
        best_cluster = -1
        for label in cluster_count.keys():
            count = cluster_count[label]
            if cluster_count[label] > best_c:
                best_c = count
                best_cluster = label


        goal_pt = np.array([])

        while True:
            r_x = np.random.rand(1) * map_height
            r_y = np.random.rand(1) * map_width
            random_pt = np.array([r_x, r_y])

            closest_node = np.array(self.tree.get_closest_node(random_pt))

            d = random_pt - closest_node.pt
            d_norm = np.linalg.norm(d)

            increment = d/d_norm)*self.obs_fid
            check_pt = closest_node.pt + increment

            obs = False
            while np.linalg.norm(check_pt - closest_node.pt) < d:
                check_pt_cell = int(check_pt // map_res)
                grid_index = check_pt_cell[1] * map_msg.info.width + check_pt_cell[0]
                if map_values[grid_index] >= 0.5:
                    obs = True
                    break
                check_pt += increment

            if obs:
                continue

            if d_norm < self.max_dist:
                new_node = TreeNode(pt=random_pt,parent=closest_node)
            else:
                new_pt = (d/d_norm) * self.node_max_dist
                new_node = TreeNode(pt=new_pt,parent=closest_node)

            if np.linalg.norm(new_node.pt-goal_pt) < self.goal_radius:
                goal_node = TreeNode(pt=goal_pt,parent=closest_node)
                path = goal_node.get_path_to_root()
                break

            closest_node.children.append(new_node)

        return None


def main() -> None:
    rclpy.init()
    node = PathPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
