from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'shell_simulation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']), # This will find 'shell_simulation' and its sub-packages like 'shell_simulation.agents'
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'transforms3d', # Often useful with geometry_msgs, though your control_node does manual quaternion math
        'PyYAML',       # For loading waypoints if done outside ROS params (though your planner uses ROS params for waypoints_yaml path)
        'opencv-python',# For cv_bridge and perception_node (cv_bridge itself has system deps)
    ],
    zip_safe=True,
    maintainer='Sadiq Saidu',
    maintainer_email='sadiqsaidu1221@gmail.com',
    description='Shell Eco-marathon APC 2025 Simulation Package for Team MKBardi.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'planning_node = shell_simulation.planning_node:main',
            'perception_node = shell_simulation.perception_node:main',
            'control_node = shell_simulation.control_node:main',
        ],
    },
)