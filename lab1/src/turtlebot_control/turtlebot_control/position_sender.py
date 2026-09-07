import rclpy # Import ROS2 Python client library
from rclpy.node import Node # Import Node object from the rclpy library
from std_msgs.msg import String # Import the string message-type from the std_msgs
from geometry_msgs.msg import TwistStamped, Pose, PoseStamped
from nav_msgs.msg import Odometry
import numpy as np
import math

class PositionSender(Node):
    def __init__(self):
        super().__init__('position_sender_node')

        self.publisher = self.create_publisher(PoseStamped, '/new_position', 10)

        self.timer = self.create_timer(0.7, self.timer_callback)

        self.pts = self.draw_eight(0,0,0.5)

        self.index = 0


    def timer_callback(self):
        new_msg = PoseStamped()
        new_msg.header.frame_id = 'odom'
        if self.index >= len(self.pts):
            self.index = 0
        pt = self.pts[self.index]
        new_msg.pose.position.x = pt[0]
        new_msg.pose.position.y = pt[1]
        self.publisher.publish(new_msg)
        self.index += 1


    def draw_eight(self, x, y, diameter):
        pts = []
        fid = 10
        for i in range(-180,180,fid):
            rad = math.radians(i)
            x_pos = (x + diameter) + diameter*math.cos(rad)
            y_pos = y + diameter*math.sin(rad)
            pts.append((x_pos,y_pos))
        for i in range(360,0,-fid):
            rad = math.radians(i)
            x_pos = (x - diameter) + diameter*math.cos(rad)
            y_pos = y + diameter*math.sin(rad)
            pts.append((x_pos,y_pos))
        return pts


def main(): # Main Function
    rclpy.init()
    node = PositionSender()
    rclpy.spin(node) # Keep the node alive and runnings.
    node.destroy_node() # Destroy the node after completion (typically when exited).
    rclpy.shutdown() # Shutdown the ros client on interruption.


if __name__=='__main__': # Entry point
    main()