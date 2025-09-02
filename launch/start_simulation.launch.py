import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    package_share_dir = get_package_share_directory('star_cs_agent')
    rviz_config_file = os.path.join(package_share_dir, 'rviz', 'simple_config.rviz')

    return LaunchDescription([
        Node(
            package='star_cs_agent',
            executable='command_center',
            name='command_center_node',
            output='screen'
        ),
        Node(
            package='star_cs_agent',
            executable='agent_simulator',
            name='agent1_simulator',
            output='screen',
            parameters=[{'agent_id': 'agent1', 'movement_type': 'human'}]
        ),
        Node(
            package='star_cs_agent',
            executable='agent_simulator',
            name='agent2_simulator',
            output='screen',
            parameters=[{'agent_id': 'agent2', 'movement_type': 'human'}]
        ),
        Node(
            package='star_cs_agent',
            executable='agent_simulator',
            name='agent3_simulator',
            output='screen',
            parameters=[{'agent_id': 'agent3', 'movement_type': 'ugv'}]
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen'
        ),
    ])