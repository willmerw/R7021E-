from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'r7021e_exploration'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*launch.py'))),
        (os.path.join('share', package_name, 'launch', 'rviz'),
         glob(os.path.join('launch', 'rviz', '*.rviz'))),
        (os.path.join('share', package_name, 'config'),
         glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='name',
    maintainer_email='name@users.org',
    description='Exploration template package for R7021E',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'frontier_detector_node = r7021e_exploration.frontier_detector_node:main',
            'navigation_node = r7021e_exploration.navigation_node:main',
            'path_follower_node = r7021e_exploration.path_follower_node:main', 
        ],
    },
)
