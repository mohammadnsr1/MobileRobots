import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    package_share_directory = get_package_share_directory('ras598_assignment_2')
    map_yaml_path = os.path.join(package_share_directory, 'maps', 'map.yaml')
    scout_script_path = os.path.join(package_share_directory, 'grading_scout.py')

    return LaunchDescription([
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            parameters=[{
                'yaml_filename': map_yaml_path,
                'use_sim_time': True,
            }],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager',
            output='screen',
            parameters=[{
                'autostart': True,
                'node_names': ['map_server'],
            }],
        ),
        Node(
            package='ras598_assignment_2',
            executable='planner',
            name='planner',
            output='screen',
        ),
        ExecuteProcess(
            cmd=['python3', scout_script_path],
            output='screen',
        ),
    ])
