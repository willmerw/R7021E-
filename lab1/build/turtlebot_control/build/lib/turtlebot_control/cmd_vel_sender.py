import rclpy # Import ROS2 Python client library
from rclpy.node import Node # Import Node object from the rclpy library
from std_msgs.msg import String # Import the string message-type from the std_msgs
from geometry_msgs.msg import TwistStamped, Pose
from nav_msgs.msg import Odometry
import numpy as np
class Controller(Node):
    def __init__(self):
        super().__init__('controller_node')

        #self.declare_parameter('cmd_vel', 'Test value')
        self.declare_parameter('use_sim_time', True)
        self.publisher = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        # The timer definition has two arguments: Time period in seconds and Callback function.
        self.timer = self.create_timer(1.0, self.timer_callback)

        self.x_speed = 0.0
        self.z_speed = 0.0

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.goal_pos = (0.0,0.0)

        self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.msg_callback, 10)
        self.new_pos_subscriber = self.create_subscription(Pose, '/new_position', self.new_pos_callback, 1)

    def msg_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = msg.pose.pose.orientation.z
        print(msg)
        #print(f'Current position: x={self.x:<10.4f}, y={self.y:<10.4f}, yaw={self.yaw:<10.4f}')

    def new_pos_callback(self, msg):
        x = msg.position.x
        y = msg.position.y
        self.goal_pos = (x,y)

    def timer_callback(self): # The timer calls this function.
        new_msg = TwistStamped() # Declare a new TwistStamped message.

        goal = np.array(self.goal_pos)
        pos = np.array([self.x,self.y])
        d = goal - pos

        ang_diff = np.arctan2(d[1],d[0]) - self.yaw

        if np.linalg.norm(d) > 0.1:
            self.x_speed = 0.5
            self.z_speed = ang_diff
        else:
            self.x_speed = 0.0
            self.z_speed = 0.0

        new_msg.twist.linear.x = self.x_speed *0
        new_msg.twist.angular.z = self.z_speed *0
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