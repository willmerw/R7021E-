from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description(): # Launch description function
    return LaunchDescription([ # It returns the LaunchDescription Container.
        Node( # Node object defines the executable.
            package='turtlebot_control', # Package name.
            executable='position_sender_node', # Executable name (the one mentioned in setup.py).
            name='position_sender', # Unique name.
            output='screen',
        ),
        Node(
            package='turtlebot_control',
            executable='controller_node',
            name='controller',
            output='screen',
        ),
        Node(
            package='turtlebot_control',
            executable='wall_pt_node',
            name='wall_pt',
            output='screen',
        )])