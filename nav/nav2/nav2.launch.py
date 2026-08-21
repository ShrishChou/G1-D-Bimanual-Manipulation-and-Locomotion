"""Nav2 bringup for the G1-D on the live slamware SLAM (localization + map + TF come from the robot).
Starts planner + controller + behaviors + bt_navigator + lifecycle. NO amcl/map_server.

cmd_vel is remapped to `cmd_vel_topic` (default /cmd_vel_test -> nothing drives the base, safe for bring-up).
To actually drive: cmd_vel_topic:=/cmd_vel_no_limit
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_launch_description():
    params = LaunchConfiguration("params_file")
    cmd_vel = LaunchConfiguration("cmd_vel_topic")
    lifecycle_nodes = ["controller_server", "planner_server", "behavior_server", "bt_navigator"]

    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=os.path.join(HERE, "nav2_params.yaml")),
        DeclareLaunchArgument("cmd_vel_topic", default_value="/cmd_vel_test"),

        Node(package="nav2_controller", executable="controller_server", output="screen",
             parameters=[params], remappings=[("cmd_vel", cmd_vel)]),
        Node(package="nav2_planner", executable="planner_server", output="screen",
             parameters=[params]),
        Node(package="nav2_behaviors", executable="behavior_server", output="screen",
             parameters=[params], remappings=[("cmd_vel", cmd_vel)]),
        Node(package="nav2_bt_navigator", executable="bt_navigator", output="screen",
             parameters=[params]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", output="screen",
             name="lifecycle_manager_navigation",
             parameters=[{"use_sim_time": False, "autostart": True, "node_names": lifecycle_nodes}]),
    ])
