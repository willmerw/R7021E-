import rclpy # Import ROS2 Python client library
from rclpy.node import Node # Import Node object from the rclpy library
from std_msgs.msg import String # Import the string message-type from the std_msgs
from geometry_msgs.msg import TwistStamped, Pose, PoseStamped
from nav_msgs.msg import Odometry
import numpy as np
import math
class Controller(Node):
    def __init__(self):
        super().__init__('controller_node')

        #self.declare_parameter('cmd_vel', 'Test value')

        self.publisher = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        # The timer definition has two arguments: Time period in seconds and Callback function.
        self.timer = self.create_timer(0.3, self.timer_callback)

        self.x_speed = 0.0
        self.z_speed = 0.0

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.goal_pos = None

        self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.msg_callback, 10)
        self.new_pos_subscriber = self.create_subscription(PoseStamped, '/new_position', self.new_pos_callback, 1)

    def msg_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation

        self.yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )


    def new_pos_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.goal_pos = (x,y)

    def timer_callback(self): # The timer calls this function.
        new_msg = TwistStamped() # Declare a new TwistStamped message.

        if self.goal_pos is None:
            return

        goal = np.array(self.goal_pos)
        pos = np.array([self.x,self.y])
        d = goal - pos

        ang_diff = np.arctan2(d[1],d[0])
        ang_diff = ang_diff - self.yaw
        ang_diff = (ang_diff + np.pi) % (2 * np.pi) - np.pi
        if np.linalg.norm(d) > 0.05:
            self.x_speed = 0.05
            self.z_speed = ang_diff
        else:
            self.x_speed = 0.0
            self.z_speed = 0.0
        new_msg.twist.linear.x = self.x_speed
        new_msg.twist.angular.z = self.z_speed

        self.publisher.publish(new_msg) # Publish the message.
        # Log about the published message (optional).


def main(): # Main Function
    rclpy.init()
    node = Controller()
    rclpy.spin(node) # Keep the node alive and runnings.
    node.destroy_node() # Destroy the node after completion (typically when exited).
    rclpy.shutdown() # Shutdown the ros client on interruption.


if __name__=='__main__': # Entry point
    main()