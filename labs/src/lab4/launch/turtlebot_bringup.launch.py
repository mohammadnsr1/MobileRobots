from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    declare_safety_distance = DeclareLaunchArgument(
        'safety_distance',
        default_value='0.5',
        description='Minimum distance (m) to obstacle in front before triggering avoidance',
    )

    reactive_navigator_node = Node(
        package='lab4',
        executable='reactive_navigator_v2',
        name='reactive_navigator',
        parameters=[{'safety_distance': LaunchConfiguration('safety_distance')}],
    )

    return LaunchDescription([
        declare_safety_distance,
        reactive_navigator_node,
    ])
