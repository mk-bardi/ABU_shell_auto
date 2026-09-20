#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, GroupAction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

def generate_launch_description() -> LaunchDescription:
    package_name = 'shell_simulation' # Assuming your package name is shell_simulation
    shell_simulation_pkg_share_dir = get_package_share_directory(package_name)

    # Default path for waypoints.yaml within the package
    default_waypoints_yaml = os.path.join(shell_simulation_pkg_share_dir, "config", "waypoints.yaml")
    
    # --- Declare Launch Arguments ---
    declare_waypoints_yaml_arg = DeclareLaunchArgument(
        'waypoints_yaml',
        default_value=default_waypoints_yaml,
        description='Path to the waypoints YAML file for the planner node.'
    )
    declare_sampling_resolution_arg = DeclareLaunchArgument(
        'sampling_resolution',
        default_value='2.0',
        description='Sampling resolution for GlobalRoutePlanner in meters.'
    )
    declare_use_lidar_arg = DeclareLaunchArgument(
        'use_lidar',
        default_value='True', # Make sure this is a string 'True' or 'False'
        description='Whether to use LiDAR for perception.'
    )
    declare_obstacle_dist_thresh_arg = DeclareLaunchArgument(
        'obstacle_distance_threshold',
        default_value='3.0',
        description='Distance threshold for obstacle alert in meters.'
    )
    declare_target_speed_arg = DeclareLaunchArgument(
        'target_speed_mps',
        default_value='8.0', # Default cruise speed in m/s
        description='Cruise speed for the control node in m/s.'
    )
    # PID parameters for ControlNode
    declare_kp_arg = DeclareLaunchArgument('speed_pid_kp', default_value='0.8')
    declare_ki_arg = DeclareLaunchArgument('speed_pid_ki', default_value='0.1')
    declare_kd_arg = DeclareLaunchArgument('speed_pid_kd', default_value='0.0')
    declare_max_integral_arg = DeclareLaunchArgument('speed_pid_max_integral', default_value='5.0')

    # --- Node Definitions ---
    planning_node = Node(
        package=package_name,
        executable='planning_node', # Ensure this matches your setup.py entry point
        name='planning_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            {'waypoints_yaml': LaunchConfiguration('waypoints_yaml')},
            {'sampling_resolution': LaunchConfiguration('sampling_resolution')},
            # Add other planning_node specific parameters if any, e.g., carla_host, carla_port
            {'carla_host': 'localhost'}, # Example, can be launch args too
            {'carla_port': 2000},
            {'republish_target_period': 1.0}
        ]
    )

    perception_node = Node(
        package=package_name,
        executable='perception_node', # Ensure this matches your setup.py entry point
        name='perception_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            # PythonExpression is good for converting string 'True'/'False' to bool
            {'use_lidar': PythonExpression(["True if '", LaunchConfiguration('use_lidar'), "' == 'True' else False"])},
            {'obstacle_distance_threshold': LaunchConfiguration('obstacle_distance_threshold')},
            # Add other perception_node specific parameters from its __init__
            {'roi_x_start_ratio': 0.38},
            {'roi_x_end_ratio': 0.62},
            {'roi_y_start_ratio': 0.45},
            {'roi_y_end_ratio': 0.80},
            {'lidar_max_fwd_angle_deg': 30.0},
            {'alert_latch_sec': 0.5}
        ]
    )

    control_node = Node(
        package=package_name,
        executable='control_node', # Ensure this matches your setup.py entry point
        name='control_node',
        output='screen',
        emulate_tty=True,
        parameters=[
            {'target_speed_mps': LaunchConfiguration('target_speed_mps')},
            # PID parameters are typically nested under 'speed_pid' in the node
            {'speed_pid.k_p': LaunchConfiguration('speed_pid_kp')},
            {'speed_pid.k_i': LaunchConfiguration('speed_pid_ki')},
            {'speed_pid.k_d': LaunchConfiguration('speed_pid_kd')},
            {'speed_pid.max_integral': LaunchConfiguration('speed_pid_max_integral')},
            # Add other control_node specific parameters from its __init__
            {'wheel_base': 2.8},
            {'pp_lookahead_time': 0.5},
            {'pp_L_min': 2.0},
            {'pp_L_max': 15.0},
            {'throttle_jerk_max': 0.4},
            {'brake_jerk_max': 0.6},
            {'throttle_alpha': 0.2},
            {'steer_rate_max': 0.04}, # Note: param name in node is steer_rate_max_cycle after multiplication
            {'speed_limit_mps': 13.0},
            {'control_period': 0.05},
            {'obs_brake_full': 3.0},
            {'obs_brake_start': 6.0},
            {'obs_critical_speed_threshold': 0.5}
        ]
    )
    
    # Group for nodes if needed, or just return list
    # ros_nodes = GroupAction(actions=[planning_node, perception_node, control_node])

    return LaunchDescription([
        # Declare arguments
        declare_waypoints_yaml_arg,
        declare_sampling_resolution_arg,
        declare_use_lidar_arg,
        declare_obstacle_dist_thresh_arg,
        declare_target_speed_arg,
        declare_kp_arg,
        declare_ki_arg,
        declare_kd_arg,
        declare_max_integral_arg,

        # Log message from original file
        LogInfo(msg="Ensuring 'agents' module is available for GlobalRoutePlanner in planning_node."),
        
        # Nodes
        planning_node,
        perception_node,
        control_node
        # ros_nodes # If using GroupAction
    ])
