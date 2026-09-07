import rclpy # Import ROS2 Python client library
from rclpy.node import Node # Import Node object from the rclpy library
from std_msgs.msg import String # Import the string message-type from the std_msgs

class Sender(Node):
    def __init__(self):
        super().__init__('sender_node')

        self.declare_parameter('message', 'Test value')

        self.publisher = self.create_publisher(String, '/short_msg', 10)
        # The timer definition has two arguments: Time period in seconds and Callback function.
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self): # The timer calls this function.
        new_msg = String() # Declare a new string message.

        new_msg.data = self.get_parameter('message').get_parameter_value().string_value

        self.publisher.publish(new_msg) # Publish the message.
        # Log about the published message (optional).
        self.get_logger().info(f'Publishing: "{new_msg.data}" in the topic "/short_msg"')

def main(): # Main Function
    rclpy.init() # Initialize the ros client
    node = Sender() # Declare an object of the Sender class.
    rclpy.spin(node) # Keep the node alive and runnings.
    node.destroy_node() # Destroy the node after completion (typically when exited).
    rclpy.shutdown() # Shutdown the ros client on interruption.


if __name__=='__main__': # Entry point
    main()