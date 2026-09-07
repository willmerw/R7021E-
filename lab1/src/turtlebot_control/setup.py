from setuptools import find_packages, setup

package_name = 'turtlebot_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/eight.launch.py']),
        ('share/' + package_name + '/launch', ['launch/wall_follow.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='willmer@wohlen.se',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'position_sender_node=turtlebot_control.position_sender:main',
            'controller_node=turtlebot_control.cmd_vel_sender:main',
            'wall_pt_node=turtlebot_control.closest_wallpt:main',
            'wall_follow_node=turtlebot_control.wall_follow:main',
        ],
    },
)
