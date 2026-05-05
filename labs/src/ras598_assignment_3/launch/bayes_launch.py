from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    assignment_share = get_package_share_directory('ras598_assignment_3')
    stage_share = get_package_share_directory('stage_ros2')

    rviz_config = os.path.join(assignment_share, 'config', 'bayes.rviz')
    stage_launch = os.path.join(stage_share, 'launch', 'demo.launch.py')

    return LaunchDescription([
        SetEnvironmentVariable('QT_QPA_PLATFORM', 'wayland'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(stage_launch),
            launch_arguments={
                'world': 'cave',
                'use_stamped_velocity': 'false',
            }.items(),
        ),

        Node(
            package='ras598_assignment_3',
            executable='bayes_filter',
            name='bayes_filter_3d_node',
            output='screen',
            parameters=[{'use_sim_time': True}],
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': True}],
        ),
    ])
