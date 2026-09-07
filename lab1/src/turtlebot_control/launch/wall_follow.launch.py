from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description(): # Launch description function
    return LaunchDescription([ # It returns the LaunchDescription Container.
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
        ),
        Node(
            package='turtlebot_control',
            executable='wall_follow_node',
            name='wall_follow',
            output='screen',
        )])