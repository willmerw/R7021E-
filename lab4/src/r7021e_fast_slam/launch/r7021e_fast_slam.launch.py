"""Launch the Grid-FastSLAM 2.0 node and RViz.

Values come from `config/params.yaml`. The four sweep arguments below are
OPTIONAL overrides, for a shell loop if you need it:

    for n in 1 5 20 50; do for s in 1 2 3; do
        ros2 launch r7021e_fast_slam r7021e_fast_slam.launch.py \
            num_particles:=$n seed:=$s run_name:=n${n}_s${s}
    done; done

Each defaults to empty, meaning "do not override". Pass one and it wins over
the YAML; leave it out and the YAML is used. A launch argument with a real
default value would shadow the YAML entry permanently, which is why the
override dict is built at launch time instead of being declared statically.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

## name -> how to parse the string a launch argument always gives us
SWEEP_ARGS = {
    'num_particles': int,
    'seed': int,
    'use_improved_proposal': lambda v: v.strip().lower() in ('true', '1', 'yes'),
    'run_name': str,
}


def launch_setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory('r7021e_fast_slam')
    default_rviz = os.path.join(pkg_share, 'config', 'grid_slam.rviz')

    overrides = {}
    for name, parse in SWEEP_ARGS.items():
        value = LaunchConfiguration(name).perform(context)
        if value != '':
            overrides[name] = parse(value)

    return [
        Node(
            package='r7021e_fast_slam',
            executable='grid_slam_node',
            name='grid_slam_node',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
                overrides,
            ],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', default_rviz],
            condition=IfCondition(LaunchConfiguration('use_rviz')),
        ),
    ]


def generate_launch_description():
    pkg_share = get_package_share_directory('r7021e_fast_slam')
    default_params = os.path.join(pkg_share, 'config', 'params.yaml')

    args = [
        DeclareLaunchArgument('params_file', default_value=default_params,
                              description='YAML file with node parameters.'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        ## Only set this true if the bag is played with `ros2 bag play --clock`.
        ## Without a /clock publisher the node's clock never advances, the map
        ## timer never fires, and nothing is published at all.
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='true only with `ros2 bag play '
                                          '--clock`.'),
    ]
    args += [
        DeclareLaunchArgument(
            name, default_value='',
            description=f'Optional override for {name}; empty uses params_file.')
        for name in SWEEP_ARGS
    ]
    return LaunchDescription(args + [OpaqueFunction(function=launch_setup)])
