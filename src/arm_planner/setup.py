from setuptools import find_packages, setup
import glob
import sys
import os
from glob import glob

package_name = 'arm_planner'

python_version = f'{sys.version_info.major}.{sys.version_info.minor}'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'scipy'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='ROS2 service nodes for arm planning in simulation and real robot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'arm_planner_sim_node = arm_planner.arm_planner_sim_node:main',
            'arm_planner_real_robot_node = arm_planner.arm_planner_real_robot_node:main',
        ],
    },
)

