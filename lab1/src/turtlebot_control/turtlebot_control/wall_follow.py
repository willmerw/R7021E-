import rclpy # Import ROS2 Python client library
from rclpy.node import Node # Import Node object from the rclpy library
from std_msgs.msg import String # Import the string message-type from the std_msgs
from geometry_msgs.msg import TwistStamped, Pose, PoseStamped
from nav_msgs.msg import Odometry
import numpy as np
import math

class FollowWall(Node):
    def __init__(self):
        super().__init__('follow_wall_node')

        self.publisher = self.create_publisher(PoseStamped, '/new_position', 10)

        self.timer = self.create_timer(0.7, self.timer_callback)

        self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.msg_callback, 10)
        self.wallpt_subscriber = self.create_subscription(PoseStamped, '/wall_pt', self.wallpt_callback, 10)

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.wall_pt = (0.0,0.0)

    def msg_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation

        self.yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def wallpt_callback(self, msg):
        self.wall_pt = (msg.pose.position.x, msg.pose.position.y)

    def timer_callback(self):
        new_msg = PoseStamped()
        new_msg.header.frame_id = 'odom'

        rotate = np.pi/2
        wall_dist = 0.4

        dx = self.x - self.wall_pt[0]
        dy = self.y - self.wall_pt[1]

        d = np.array([dx, dy])
        d_norm = np.linalg.norm(d)
        d = d / (d_norm + 1e-6)
        print("Distance to wall:", d_norm)

        wall_pt = np.array(self.wall_pt)

        d = d * wall_dist

        wall_pt = wall_pt + d

        d_rot_x = d[0] * np.cos(rotate) - d[1] * np.sin(rotate)
        d_rot_y = d[0] * np.sin(rotate) + d[1] * np.cos(rotate)

        wall_pt = wall_pt + np.array([d_rot_x, d_rot_y]) * 2

        new_msg.pose.position.x = wall_pt[0]
        new_msg.pose.position.y = wall_pt[1]

        self.publisher.publish(new_msg)




def main(): # Main Function
    rclpy.init()
    node = FollowWall()
    rclpy.spin(node) # Keep the node alive and runnings.
    node.destroy_node() # Destroy the node after completion (typically when exited).
    rclpy.shutdown() # Shutdown the ros client on interruption.


if __name__=='__main__': # Entry point
    main()