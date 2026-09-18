#!/usr/bin/env python3

import math
from typing import Optional, Tuple, List

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, Quaternion,Point
from nav_msgs.msg import Path, OccupancyGrid
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.time import Time
from visualization_msgs.msg import Marker

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
    def __init__(self, pt, parent):
        self.pt = np.array(pt)
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
        self.declare_parameter('frontier_topic', 'frontiers')
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

        self.node_max_dist = 0.2
        self.obs_fid = 0.01 #obstacle fidelity
        self.goal_radius = 0.2

        self.map_inflation = 3

        self.tree = TreeNode([self.x,self.y], None)
        self.tree_pts = []

        self.goal_reached = True

        self.last_goal = None
        self.last_goal_rad = 0.1

        # Publishers (add path publisher here)
        self.path_pub = self.create_publisher(Path, 'path', 10)

        self.tree_pub = self.create_publisher(Marker, 'rrt_tree', 10)
        self.inf_map_pub = self.create_publisher(OccupancyGrid, 'inf_map', 10)
        self.upd_frontier_pub = self.create_publisher(OccupancyGrid, 'upd_frontiers', 10)

        self.goal_frontier_pub = self.create_publisher(Marker, 'goal_frontier', 10)

        # Subscribers
        self.map_sub = self.create_subscription(OccupancyGrid, map_topic, self._on_map, default_qos)
        self.frontier_sub = self.create_subscription(OccupancyGrid, frontier_topic, self._on_frontier, default_qos)
        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # State
        self._latest_map: Optional[OccupancyGrid] = None
        self._latest_frontier: Optional[OccupancyGrid] = None
        self.latest_path = None

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
        if self.goal_reached:
            path = self.plan_path((x, y, yaw),
                                msg,
                                self._latest_frontier)
            self.latest_path = path
            if path is None:
                return


        pos = np.array([x,y])
        if self.latest_path is None:
            return

        path = self.latest_path
        if np.linalg.norm(path[-1]-pos) <= self.goal_radius:
            self.goal_reached = True

        else:
            self.goal_reached = False



    # ----------------- Planning -----------------

    def plan_path(self,
                  start: Tuple[float, float, float],
                  map_msg: OccupancyGrid,
                  frontier_msg: Optional[OccupancyGrid]):
        """
        Use this function to plan the path
        """
        if frontier_msg is None:
            return None

        map_msg, frontier_msg = self.inflate_map(map_msg,frontier_msg)

        self.inf_map_pub.publish(map_msg)
        self.upd_frontier_pub.publish(frontier_msg)

        map_res = map_msg.info.resolution
        map_height = map_msg.info.height * map_res
        map_width = map_msg.info.width * map_res
        map_values = map_msg.data
        origin_x = map_msg.info.origin.position.x
        origin_y = map_msg.info.origin.position.y
        map_grid = np.array(map_msg.data, dtype=np.int16).reshape(
            (map_msg.info.height, map_msg.info.width))

        cell_x = int((start[0]-origin_x)/map_res)
        cell_y = int((start[1]-origin_y)/map_res)
        pos = np.array([start[0],start[1]])
        #check if robot is in occupied space
        if map_grid[cell_y,cell_x] >= 50:
            closest_free_d = float('inf')
            rows, cols = np.where(map_grid < 50)
            for row,col in zip(rows, cols):
                ptx = origin_x + (col + 0.5) * map_res
                pty = origin_y + (row + 0.5) * map_res
                pt = np.array([ptx,pty])
                d = np.linalg.norm(pt-pos)
                if d < closest_free_d:
                    closest_free = pt
                    closest_free_d = d
            self.tree = TreeNode([closest_free[0],closest_free[1]], None)
        else:
            self.tree = TreeNode([start[0],start[1]], None)

        self.tree_pts = []



        goal_pt = self.extract_frontier(frontier_msg)
        if goal_pt is None:
            return

        self.publish_marker(goal_pt)

        cell_x = int((goal_pt[0]-origin_x)/map_res)
        cell_y = int((goal_pt[1]-origin_y)/map_res)

        #check if goal is in occupied space
        if map_grid[cell_y,cell_x] >= 50:
            closest_free_d = float('inf')
            rows, cols = np.where(map_grid < 50)
            for row,col in zip(rows, cols):
                ptx = origin_x + (col + 0.5) * map_res
                pty = origin_y + (row + 0.5) * map_res
                pt = np.array([ptx,pty])
                d = np.linalg.norm(pt-goal_pt)
                if d < closest_free_d:
                    closest_free = pt
                    closest_free_d = d
            goal_pt = closest_free

        # check if new goal is very close to previous goal
        if self.last_goal is not None:
            d = np.linalg.norm(goal_pt - np.array(self.last_goal))
            if d < self.last_goal_rad:
                return None
        self.last_goal = goal_pt

        # check if direct path to goal exists
        start_pos = np.array([start[0],start[1]])
        obs = self.check_obs(start_pos,goal_pt,map_msg)
        if not obs:
            path_msg = Path()
            path = [start_pos,goal_pt]
            path_msg.header.frame_id = 'map'
            path_pt_ls = []
            for pt in path:
                path_pt = PoseStamped()
                path_pt.header.frame_id = 'map'
                path_pt.pose.position.x = pt[0]
                path_pt.pose.position.y = pt[1]
                path_pt_ls.append(path_pt)
            path_msg.poses = path_pt_ls
            self.path_pub.publish(path_msg)
            return path

        # RRT
        nodes_near_goal = []
        while len(nodes_near_goal) < 3:
            r_x = np.random.uniform(origin_x, origin_x+map_width)
            r_y = np.random.uniform(origin_y, origin_y+map_height)
            random_pt = np.array([r_x, r_y], dtype=float)

            _, closest_node = self.tree.get_closest_node(random_pt)
            d = np.array(random_pt - closest_node.pt)
            d_norm = np.linalg.norm(d)

            obs = self.check_obs(closest_node.pt,random_pt,map_msg)

            if obs:
                continue

            if d_norm < self.node_max_dist:
                new_node = TreeNode(pt=random_pt,parent=closest_node)
                self.tree_pts.append((closest_node.pt,random_pt))

            else:
                new_pt = closest_node.pt + ((d/d_norm) * self.node_max_dist)
                new_node = TreeNode(pt=new_pt,parent=closest_node)
                self.tree_pts.append((closest_node.pt, new_pt))
            if np.linalg.norm(closest_node.pt-goal_pt) < self.goal_radius:
                continue

            if np.linalg.norm(new_node.pt-goal_pt) < self.goal_radius:
                nodes_near_goal.append(new_node)
                self.publish_tree(self.tree_pts)
                break
            closest_node.children.append(new_node)

        shortest_path = float('inf')
        for goal_node in nodes_near_goal:
            path = goal_node.get_path_to_root()
            path_len = 0
            for i in range(len(path)-1):
                seg_len = np.linalg.norm(path[i]-path[i+1])
                path_len += seg_len
            if path_len < shortest_path:
                best_path = path.copy()

        path = best_path

        # trim path to avoid turning around
        pose = self.get_robot_pose()
        x,y,yaw = pose
        pos = np.array([x,y])
        closest = np.linalg.norm(path[0]-pos)
        for i,pt in enumerate(path[1:]):
            d = np.linalg.norm(pt-pos)
            if d<closest:
                continue
            else:
                path = path[i:]
                break

        path = self.optimize_path(path,map_msg)
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_pt_ls = []
        for pt in path:
            path_pt = PoseStamped()
            path_pt.header.frame_id = 'map'
            path_pt.pose.position.x = pt[0]
            path_pt.pose.position.y = pt[1]
            path_pt_ls.append(path_pt)
        path_msg.poses = path_pt_ls
        self.path_pub.publish(path_msg)

        return path

        return None

    def check_obs(self, start, goal, map_msg):
        map_res = map_msg.info.resolution
        map_values = map_msg.data
        origin_x = map_msg.info.origin.position.x
        origin_y = map_msg.info.origin.position.y

        d = goal - start
        d_norm = np.linalg.norm(d)

        increment = (d/d_norm)*self.obs_fid
        check_pt = start + increment
        obs = False
        while np.linalg.norm(check_pt - start) < d_norm:
            cell_x = int((check_pt[0]-origin_x)/map_res)
            cell_y = int((check_pt[1]-origin_y)/map_res)
            grid_index = int(cell_y * map_msg.info.width + cell_x)
            if map_values[grid_index] >= 0.5 or map_values[grid_index] == -1:
                obs = True
                break
            check_pt += increment
        return obs

    def publish_tree(self, tree_edges):

            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "rrt_tree"
            marker.id = 0
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD

            marker.scale.x = 0.02
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            for parent, child in tree_edges:
                p_start = Point(x=float(parent[0]), y=float(parent[1]), z=0.0)
                p_end = Point(x=float(child[0]), y=float(child[1]), z=0.0)
                marker.points.append(p_start)
                marker.points.append(p_end)

            self.tree_pub.publish(marker)

    def publish_marker(self, pt):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = 'goal_frontier'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = float(pt[0])
        marker.pose.position.y = float(pt[1])
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.goal_frontier_pub.publish(marker)

    def optimize_path(self, path, map_msg):
        i = 0
        while (i+2) < len(path)-1:
            child = path[i]
            gp = path[i+2]
            obs = self.check_obs(child,gp,map_msg)

            if not obs:
                path.pop(i+1)
            else:
                i+=1

        i = 0
        while (i+2) < len(path)-1:
            child = path[i]
            middle = path[i+1]
            gp = path[i+2]

            step_size = 0.1

            c_m = np.array(middle-child)
            d1 = np.linalg.norm(c_m)
            c_md = np.linalg.norm(c_m)
            c_m = (c_m/c_md)*step_size


            gp_m = np.array(middle-gp)
            d2 = np.linalg.norm(gp_m)
            gp_md = np.linalg.norm(gp_m)
            gp_m = (gp_m/gp_md)*step_size

            child_n = child.copy()
            gp_n = gp.copy()


            while (np.linalg.norm(child_n-child) < d1 and np.linalg.norm(gp_n-gp) < d2):
                child_n += c_m
                gp_n += gp_m
                obs = self.check_obs(child_n,gp_n,map_msg)
                if not obs:
                    path.pop(i+1)
                    path[i+1:i+1] = [child_n, gp_n]

                    break
            i = i+2
        segment_path = []
        for i in range(len(path)-1):
            segment_length = np.linalg.norm(path[i]-path[i+1])
            segment = list(np.linspace(path[i],path[i+1],int(segment_length/0.1)))
            segment_path = segment_path + segment
        path = segment_path

        return path

    def inflate_map(self,map_msg, frontier_msg):
        map_grid = np.array(map_msg.data, dtype=np.int16).reshape(
        (map_msg.info.height, map_msg.info.width))

        frontier_grid = np.array(frontier_msg.data, dtype=np.int16).reshape(
        (frontier_msg.info.height, frontier_msg.info.width))

        rows, cols = np.where(map_grid >= 50)

        for row,col in zip(rows,cols):
            infl = self.map_inflation
            for i in range(-infl,infl):
                for j in range(-infl,infl):
                    try:
                        map_grid[row+i][col+j] = 100
                    except Exception as e:
                        continue
            infl = infl+2
            for i in range(-infl,infl):
                for j in range(-infl,infl):
                    try:
                        frontier_grid[row+i][col+j] = 0
                    except Exception as e:
                        continue
        map_msg.data = map_grid.ravel().tolist()
        frontier_msg.data = frontier_grid.ravel().tolist()
        return map_msg, frontier_msg

    def extract_frontier(self, frontier_msg):
        frontier_grid = np.array(frontier_msg.data, dtype=np.int16).reshape(
            (frontier_msg.info.height, frontier_msg.info.width))

        rows, cols = np.where(frontier_grid == 100)

        frontier_res = frontier_msg.info.resolution

        f_origin_x = frontier_msg.info.origin.position.x
        f_origin_y = frontier_msg.info.origin.position.y
        frontiers_x = f_origin_x + (cols + 0.5) * frontier_res
        frontiers_y = f_origin_y + (rows + 0.5) * frontier_res
        points = np.column_stack((frontiers_x, frontiers_y))

        mean_pt = np.mean(points,axis=0)

        pose = self.get_robot_pose()
        x,y,yaw = pose
        pos = np.array([x,y])
        heading = np.array([np.cos(yaw),np.sin(yaw)])

        w_sim = 3.0
        w_dist = 0.5
        w_mean_dist = 1.0
        scores = []
        for pt in points:
            dpt = pt - pos
            cosine_sim = np.dot(dpt, heading) / (np.linalg.norm(heading) * np.linalg.norm(dpt))
            dist = np.linalg.norm(pt-pos)
            dist_mean = np.linalg.norm(pt-mean_pt)

            score = w_sim * cosine_sim - w_dist*dist - w_mean_dist * dist_mean
            scores.append(score)
        if len(scores) >= 1:
            idx = scores.index(max(scores))
            best_pt = points[idx]
        else:
            return None


        return best_pt


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
