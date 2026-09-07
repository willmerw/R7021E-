import rclpy # Import ROS2 Python client library
from rclpy.node import Node # Import Node object from the rclpy library
from std_msgs.msg import String # Import the string message-type from the std_msgs

class Receiver(Node):
    def __init__(self):
        super().__init__('receiver_node')

        self.subscriber = self.create_subscription(String, '/short_msg', self.msg_callback, 10)
    def msg_callback(self, msg):
        self.get_logger().info(f'Received new message: "{msg.data}" in the topic "/short_msg"')


def main(args=None):
    rclpy.init(args=args)
    node = Receiver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__=='__main__':
    main()