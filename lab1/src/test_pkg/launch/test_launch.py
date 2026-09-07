from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description(): # Launch description function
    return LaunchDescription([ # It returns the LaunchDescription Container.
        Node( # Node object defines the executable.
            package='test_pkg', # Package name.
            executable='sender_node', # Executable name (the one mentioned in setup.py).
            name='sender', # Unique name.
            output='screen',
            parameters=[{'message':'R7021E Advanced Robotics'}] # The parameter is set here.
        ),
        Node(
            package='test_pkg',
            executable='receiver_node',
            name='receiver',
            output='screen',
        )])