import rclpy # Import ROS2 Python client library
from rclpy.node import Node # Import Node object from the rclpy library
from std_msgs.msg import String # Import the string message-type from the std_msgs
from geometry_msgs.msg import TwistStamped, Pose, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
import math


class ClosestPoint(Node):
    def __init__(self):
        super().__init__('closest_point_node')

        #self.declare_parameter('cmd_vel', 'Test value')

        self.publisher = self.create_publisher(PoseStamped, '/wall_pt', 10)

        self.timer = self.create_timer(0.3, self.timer_callback)

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.closest_pt = (0.0,0.0)


        self.odom_subscriber = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.new_pos_subscriber = self.create_subscription(Odometry, '/odom', self.msg_callback, 1)

    def msg_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation

        self.yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

        x_points = ranges * np.cos(angles)
        y_points = ranges * np.sin(angles)

        min_index = np.argmin(ranges)
        x_close = x_points[min_index]
        y_close = y_points[min_index]

        x_wf = x_close * np.cos(self.yaw) - y_close * np.sin(self.yaw)
        y_wf = x_close * np.sin(self.yaw) + y_close * np.cos(self.yaw)

        x_wf = x_wf + self.x
        y_wf = y_wf + self.y

        pt_wf = np.array([x_wf, y_wf])

        self.closest_pt = pt_wf


    def timer_callback(self): # The timer calls this function.
        new_msg = PoseStamped() # Declare a new PoseStamped message.

        new_msg.header.frame_id = 'odom'
        new_msg.pose.position.x = self.closest_pt[0]
        new_msg.pose.position.y = self.closest_pt[1]

        self.publisher.publish(new_msg)


def main(): # Main Function
    rclpy.init()
    node = ClosestPoint()
    rclpy.spin(node) # Keep the node alive and runnings.
    node.destroy_node() # Destroy the node after completion (typically when exited).
    rclpy.shutdown() # Shutdown the ros client on interruption.


if __name__=='__main__': # Entry point
    main()