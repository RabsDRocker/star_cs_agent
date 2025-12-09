from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'star_cs_agent'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'data'), glob('data/.*')),
        (os.path.join('share', package_name, 'data'), glob('data/*.json')),
        (os.path.join('share', package_name, 'data/assets'), glob('data/assets/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='The STAR-CS Agent Package for ROS 2',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'agent_simulator = star_cs_agent.agent_simulator:main',
            'command_center = star_cs_agent.command_center_simple:main',
        ],
    },
)