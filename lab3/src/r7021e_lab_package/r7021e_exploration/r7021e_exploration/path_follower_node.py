#!/usr/bin/env python3
# frontier_detector_node.py

import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped, TransformStamped
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException


class PathFollower(Node):
    def __init__(self):
        super().__init__('path_follower')
        self.path = []
        self.path_header=""

        # Parameters
        self.declare_parameter('max_v', 0.15)  # m/s
        self.declare_parameter('kp_vel', 1.0)
        self.declare_parameter('max_w', 1.0)  # rad/s
        self.declare_parameter('kp_yaw', 2.0)
        self.declare_parameter('look_ahead', 0.2)  # m

        self.max_v = float(self.get_parameter('max_v').value)
        self.kp_vel = float(self.get_parameter('kp_vel').value)
        self.max_w = float(self.get_parameter('max_w').value)
        self.kp_yaw = float(self.get_parameter('kp_yaw').value)
        self.look_ahead = float(self.get_parameter('look_ahead').value)

        # subscriptions and publishers
        self.map_sub = self.create_subscription(
            Path, 'path', self.path_callback, 1)

        self.vel_pub = self.create_publisher(
            TwistStamped, 'cmd_vel', 1)

        # TransformListener
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)

        # Timers
        self.follow_path_time = self.create_timer(0.1, self.follow_path)

    def path_callback(self, msg: Path):
        self.path_header=msg.header.frame_id
        self.path=[]
        for point in msg.poses:
            self.path.append((point.pose.position.x, point.pose.position.y))

    def update_robot_pos(self):
        ts: TransformStamped = self.buffer.lookup_transform(
            'map',         # target frame
            'base_link',   # source frame
            rclpy.time.Time()
        )

        t = ts.transform.translation
        self.robot_pos = (t.x, t.y)

        q = ts.transform.rotation
        self.robot_yaw = self.yaw_from_quaternion(q.x, q.y, q.z, q.w)

    def yaw_from_quaternion(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def follow_path(self):
        if not self.path:
            return

        vel_msg = TwistStamped()
        vel_msg.header.stamp= self.get_clock().now().to_msg()
        vel_msg.header.frame_id= self.path_header

        self.update_robot_pos()

        while len(self.path) > 1:
            if self.dist(self.path[0], self.robot_pos) > self.look_ahead:
                break
            self.path.pop(0)

        dist_to_target = self.dist(self.path[0], self.robot_pos)
        if dist_to_target < self.look_ahead:
            vel_msg.twist.linear.x = self.kp_vel*dist_to_target
            vel_msg.twist.linear.x = min(self.max_v, vel_msg.twist.linear.x)
        else:
            vel_msg.twist.linear.x = self.max_v

        dx = self.path[0][0]-self.robot_pos[0]
        dy = self.path[0][1]-self.robot_pos[1]
        ang_to_target = math.atan2(dy, dx)
        dif_ang = self.ang_dist(self.robot_yaw, ang_to_target)

        if abs(dif_ang)>0.3:
            vel_msg.twist.linear.x = 0.0
        vel_msg.twist.angular.z = dif_ang * self.kp_yaw
        if abs(vel_msg.twist.angular.z) > self.max_w:
            vel_msg.twist.angular.z = math.copysign(self.max_w, vel_msg.twist.angular.z)

        self.vel_pub.publish(vel_msg)

    def dist(self, pos1, pos2):
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]

        return math.sqrt(dx*dx+dy*dy)

    def ang_dist(self, v1: float, v2: float):
        '''
        Get the  shortest distance between two angles,
        handles extreme values of angles.
        '''
        v1 = math.copysign(math.fmod(v1, (2*math.pi)), v1)
        v2 = math.copysign(math.fmod(v2, (2*math.pi)), v2)

        dif_ang = v2-v1

        if abs(dif_ang) < math.pi:
            return dif_ang
        else:
            dif_ang2 = 2*math.pi-abs(dif_ang)
            dif_ang2 = math.copysign(dif_ang2, -dif_ang)
            return dif_ang2


def main(args=None):
    rclpy.init(args=args)
    node = PathFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
