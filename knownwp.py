#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ------------------------------------------------------------------
    # Launch arguments
    # ------------------------------------------------------------------
    default_map = PathJoinSubstitution([
        FindPackageShare('balance_mapping'),
        'maps',
        'balance_map_laab.yaml',
    ])

    map_arg = DeclareLaunchArgument(
        'map',
        default_value=default_map,
        description='Full path to the saved map YAML file.',
    )
    

    map_file = LaunchConfiguration('map')

    # ------------------------------------------------------------------
    # Locate existing project launch files
    # ------------------------------------------------------------------
    robot_base_launch_file = os.path.join(
        get_package_share_directory('balance_robot_base'),
        'launch',
        'balance_robot_base.launch.py',
    )

    lidar_launch_file = os.path.join(
        get_package_share_directory('lslidar_driver_n10p'),
        'launch',
        'lslidar_launch.py',
    )

    localization_launch_file = os.path.join(
        get_package_share_directory('balance_navigation'),
        'launch',
        'balance_amcl_localization.launch.py',
    )

    nav2_launch_file = os.path.join(
        get_package_share_directory('balance_navigation'),
        'launch',
        'balance_nav2_core.launch.py',
    )

    # ------------------------------------------------------------------
    # 0 seconds: robot base and static transforms
    # ------------------------------------------------------------------
    robot_base_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(robot_base_launch_file)
    )

    # ------------------------------------------------------------------
    # 2 seconds: LiDAR, publishing /scan
    # ------------------------------------------------------------------
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(lidar_launch_file)
    )

    delayed_lidar_launch = TimerAction(
        period=2.0,
        actions=[
            LogInfo(msg='[bringup] Starting N10P LiDAR...'),
            lidar_launch,
        ],
    )

    # ------------------------------------------------------------------
    # 5 seconds: map server and AMCL
    # ------------------------------------------------------------------
    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(localization_launch_file),
        launch_arguments={
            'map': map_file,
        }.items(),
    )

    delayed_localization_launch = TimerAction(
        period=5.0,
        actions=[
            LogInfo(msg='[bringup] Starting map server and AMCL...'),
            localization_launch,
        ],
    )

    # ------------------------------------------------------------------
    # 9 seconds: Nav2 planner, controller, BT navigator, behaviors,
    # waypoint follower, and lifecycle manager
    # ------------------------------------------------------------------
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_launch_file)
    )

    delayed_nav2_launch = TimerAction(
        period=9.0,
        actions=[
            LogInfo(msg='[bringup] Starting Nav2...'),
            nav2_launch,
        ],
    )    
    # 15 seconds: Start frontier coverage explorer
    coverage_node = Node(
        package='lidar_coverage',
        executable='coverage_explorer',
        name='lidar_coverage_explorer',
        output='screen',
        parameters=[{
            'map_file': map_file,
            'target_class': 'bottle',
        }],
    )

    delayed_coverage_launch = TimerAction(
        period=15.0,
        actions=[
            LogInfo(
                msg='[bringup] Starting frontier exploration'
            ),
            coverage_node,
        ],
    )


    return LaunchDescription([
        map_arg,
        robot_base_launch,
        delayed_lidar_launch,
        delayed_localization_launch,
        delayed_nav2_launch,
        delayed_coverage_launch,
    ])
