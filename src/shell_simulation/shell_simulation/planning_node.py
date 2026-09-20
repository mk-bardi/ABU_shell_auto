#!/usr/bin/env python3

from __future__ import annotations
import math
import os
import traceback
from pathlib import Path
from typing import List, Optional # Added Optional
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import Point, PoseStamped, PointStamped, Quaternion
from nav_msgs.msg import Odometry, Path as NavPath # Renamed to avoid conflict
from std_msgs.msg import Bool, Header # Added Header for explicit use if needed
import yaml

try:
    import carla
    from .agents.navigation.global_route_planner import GlobalRoutePlanner
except ImportError as e:
    # Log an error and exit if carla or agents are not found.
    # This helps in diagnosing setup issues early.
    rclpy.logging.get_logger("planning_node_imports").fatal(
        f"Failed to import carla or agents.navigation.global_route_planner: {e}. "
        "Ensure CARLA Python API is installed and agents module is in PYTHONPATH."
    )
    raise SystemExit(f"ImportError: {e}")


class PlannerNode(Node):
    def __init__(self) -> None:
        super().__init__('planning_node')

        self.declare_parameter('waypoints_yaml', 'config/waypoints.yaml')
        self.declare_parameter('waypoint_threshold', 3.0) # meters
        self.declare_parameter('sampling_resolution', 2.0) # meters for GRP
        # The simulator runs on a remote host in the competition CI; its address is
        # exported as CARLA_SERVER by the job script. Fall back to localhost so a
        # developer running CARLA on their own machine needs no extra configuration.
        self.declare_parameter('carla_host', os.environ.get('CARLA_SERVER', 'localhost'))
        self.declare_parameter('carla_port', 2000)
        self.declare_parameter('republish_target_period', 1.0) # seconds

        self.waypoint_threshold_sq: float = self.get_parameter('waypoint_threshold').value ** 2
        sampling_res: float = self.get_parameter('sampling_resolution').value
        carla_host: str = self.get_parameter('carla_host').value
        carla_port: int = self.get_parameter('carla_port').value
        way_yaml_path_str: str = self.get_parameter('waypoints_yaml').value
        self.way_yaml_file: Path = self._resolve_waypoints_path(way_yaml_path_str)
            
        republish_period: float = self.get_parameter('republish_target_period').value

        self.tsp_goals: List[Point] = self._load_yaml_waypoints(self.way_yaml_file)
        if not self.tsp_goals:
            self.get_logger().fatal("No waypoints loaded — shutting down.")
            raise SystemExit("No waypoints loaded from YAML.")

        try:
            self.get_logger().info(f"Attempting to connect to CARLA at {carla_host}:{carla_port}")
            client = carla.Client(carla_host, carla_port)
            client.set_timeout(10.0) # seconds
            world = client.get_world()
            carla_map = world.get_map()
            self.grp = GlobalRoutePlanner(carla_map, sampling_res)
            self.get_logger().info(f"GlobalRoutePlanner initialized (resolution = {sampling_res:.1f} m)")
        except RuntimeError as e: # More specific exception for CARLA connection issues
            self.get_logger().fatal(f"Failed to connect to CARLA or initialize GlobalRoutePlanner: {e}")
            raise SystemExit(f"CARLA/GRP initialization failed: {e}")
        except Exception as e: # Catch other potential exceptions
            self.get_logger().fatal(f"Unexpected error during CARLA/GRP initialization: {e}\n{traceback.format_exc()}")
            raise SystemExit(f"Unexpected CARLA/GRP init error: {e}")


        self.current_nav_path_wps: List[carla.Waypoint] = []
        self.needs_new_path_segment: bool = True

        latched_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.path_pub = self.create_publisher(NavPath, '/nav_path', latched_qos)
        self.waypoint_pub = self.create_publisher(PointStamped, '/current_target_waypoint', latched_qos)
        self.complete_pub = self.create_publisher(Bool, '/mission_complete', latched_qos)

        sensor_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        self.odom_sub = self.create_subscription(Odometry, '/carla/ego_vehicle/odometry', self.odom_cb, sensor_qos)

        self.current_tsp_goal_idx: int = 0
        self.all_goals_done: bool = False
        self.vehicle_odom: Optional[Odometry] = None

        if self.tsp_goals: # Initial publication of the first target
             self._publish_current_target_tsp_goal()

        self.target_republish_timer = self.create_timer(republish_period, self._republish_target_tsp_goal_timed_event)
        self.path_planning_timer = self.create_timer(0.1, self.plan_path_segment_timed_event) # Plan more frequently

        self.get_logger().info("Planner node ready. Waiting for odometry to plan first path segment.")

    def _resolve_waypoints_path(self, configured: str) -> Path:
        """Find the waypoints YAML, whoever launched us.

        The competition's own launch file starts this node without setting
        `waypoints_yaml`, and its working directory is the job root rather
        than the package, so a bare relative default resolves to nothing.
        Try the configured value first, then the copy installed into the
        package's share directory, which is always present at run time.
        """
        candidates = []
        if configured:
            candidates.append(Path(configured).expanduser())
        try:
            from ament_index_python.packages import get_package_share_directory
            share_dir = Path(get_package_share_directory('shell_simulation'))
            candidates.append(share_dir / 'config' / 'waypoints.yaml')
            if configured:
                # Allow a path given relative to the package share directory.
                candidates.append(share_dir / configured)
        except Exception as e:  # package not found / ament index unavailable
            self.get_logger().warn(f"Could not query package share directory: {e}")

        # An absolute `configured` makes share_dir / configured the same path again.
        candidates = list(dict.fromkeys(candidates))

        for candidate in candidates:
            if candidate.is_file():
                self.get_logger().info(f"Using waypoints file: {candidate}")
                return candidate.resolve()

        tried = ', '.join(str(c) for c in candidates) or '<nothing configured>'
        self.get_logger().fatal(f"Waypoints YAML not found. Tried: {tried}")
        raise SystemExit(f"Waypoints file not found (tried: {tried})")

    def _load_yaml_waypoints(self, yaml_file: Path) -> List[Point]:
        try:
            with yaml_file.open('r') as f:
                data = yaml.safe_load(f)
            if 'ordered_waypoints' not in data or not isinstance(data['ordered_waypoints'], list):
                self.get_logger().error("YAML file must contain 'ordered_waypoints' as a list.")
                return []
            
            wps: List[Point] = []
            for i, p_coords in enumerate(data['ordered_waypoints']):
                if isinstance(p_coords, list) and len(p_coords) >= 2:
                    try:
                        x = float(p_coords[0])
                        y = float(p_coords[1])
                        z = float(p_coords[2]) if len(p_coords) > 2 else 0.0 # Default Z to 0
                        wps.append(Point(x=x, y=y, z=z))
                    except ValueError:
                        self.get_logger().error(f"Invalid coordinate format for waypoint {i} in YAML: {p_coords}")
                        return [] # Or skip this waypoint
                else:
                    self.get_logger().error(f"Waypoint {i} in YAML is not a list of 2 or 3 coordinates: {p_coords}")
                    return [] # Or skip
            self.get_logger().info(f"Loaded {len(wps)} TSP-ordered goal points from {yaml_file.name}.")
            return wps
        except FileNotFoundError:
            self.get_logger().error(f"Waypoints YAML file not found at {yaml_file}.")
            return []
        except yaml.YAMLError as e:
            self.get_logger().error(f"Error parsing waypoints YAML file {yaml_file}: {e}")
            return []
        except Exception as e:
            self.get_logger().error(f"Unexpected error loading waypoints from {yaml_file}: {e}\n{traceback.format_exc()}")
            return []

    def _publish_current_target_tsp_goal(self) -> None:
        if self.all_goals_done or not self.tsp_goals or self.current_tsp_goal_idx >= len(self.tsp_goals):
            return

        try:
            current_goal_point = self.tsp_goals[self.current_tsp_goal_idx]
            
            target_point_msg = PointStamped()
            target_point_msg.header = Header() # Explicitly create Header object
            target_point_msg.header.stamp = self.get_clock().now().to_msg()
            target_point_msg.header.frame_id = "map"  # Ensure this frame_id is correct for your system
            target_point_msg.point = current_goal_point # Assign the geometry_msgs/Point

            self.waypoint_pub.publish(target_point_msg)
            # self.get_logger().debug(f"Published current TSP target waypoint: idx={self.current_tsp_goal_idx}, point=({current_goal_point.x:.2f}, {current_goal_point.y:.2f})")
        except IndexError: # Should be caught by the initial check, but as a safeguard
            self.get_logger().error(f"TSP goal index {self.current_tsp_goal_idx} out of bounds for {len(self.tsp_goals)} goals during publish.")
        except AttributeError as ae: # Catch the specific error from the trace
            self.get_logger().error(f"AttributeError while creating PointStamped: {ae}. This might indicate a ROS typesupport issue.\n{traceback.format_exc()}")
        except Exception as e:
            self.get_logger().error(f"Error in _publish_current_target_tsp_goal: {e}\n{traceback.format_exc()}")


    def _republish_target_tsp_goal_timed_event(self) -> None:
        if self.vehicle_odom is None: # Don't do anything if we don't have an odom yet
            return
        if self.all_goals_done:
            # self.get_logger().info("All TSP goals achieved. Not republishing target.")
            return
        self._publish_current_target_tsp_goal()

    def odom_cb(self, msg: Odometry) -> None:
        self.vehicle_odom = msg
        # No immediate need to plan path here, let the timer handle it
        # to decouple odom updates from planning frequency.

    def plan_path_segment_timed_event(self) -> None:
        if self.vehicle_odom is None or self.all_goals_done:
            # self.get_logger().info("Skipping path planning: no odom or mission complete.")
            if self.all_goals_done and not self.current_nav_path_wps: # Publish empty path if done and no path
                 self._publish_nav_path_from_wps([])
            return

        current_position = self.vehicle_odom.pose.pose.position
        current_cl_wp = carla.Location(current_position.x, current_position.y, current_position.z)

        if self.needs_new_path_segment:
            if self.current_tsp_goal_idx >= len(self.tsp_goals):
                self.get_logger().info("All TSP goals have been processed for path planning.")
                self.all_goals_done = True
                self.complete_pub.publish(Bool(data=True))
                self._publish_nav_path_from_wps([]) # Publish empty path
                return

            target_tsp_goal = self.tsp_goals[self.current_tsp_goal_idx]
            target_cl_wp = carla.Location(target_tsp_goal.x, target_tsp_goal.y, target_tsp_goal.z)
            self.get_logger().info(f"Planning new path segment from current odom to TSP goal {self.current_tsp_goal_idx} at ({target_cl_wp.x:.1f}, {target_cl_wp.y:.1f})")

            try:
                # GRP trace_route returns a list of (carla.Waypoint, RoadOption) tuples
                path_segment_tuples = self.grp.trace_route(current_cl_wp, target_cl_wp)
                if path_segment_tuples:
                    self.current_nav_path_wps = [wp_tuple[0] for wp_tuple in path_segment_tuples]
                    self.get_logger().info(f"Successfully planned path segment with {len(self.current_nav_path_wps)} waypoints to TSP Goal {self.current_tsp_goal_idx}.")
                    self._publish_nav_path_from_wps(self.current_nav_path_wps)
                    self.needs_new_path_segment = False # Path found, wait until near end
                else:
                    self.get_logger().warn(f"GRP trace_route returned empty path for TSP Goal {self.current_tsp_goal_idx}. Retrying on next cycle.")
                    self.current_nav_path_wps = []
                    self._publish_nav_path_from_wps([]) # Publish empty path

            except RuntimeError as e: # GRP can raise RuntimeError
                 self.get_logger().error(f"RuntimeError during GRP trace_route for TSP Goal {self.current_tsp_goal_idx}: {e}. Will retry.")
                 self.current_nav_path_wps = []
                 self._publish_nav_path_from_wps([])
            except Exception as e: # Catch any other GRP errors
                self.get_logger().error(f"Unexpected error during GRP trace_route for TSP Goal {self.current_tsp_goal_idx}: {e}. Will retry.\n{traceback.format_exc()}")
                self.current_nav_path_wps = []
                self._publish_nav_path_from_wps([])
        
        # Check if current TSP goal is reached or if end of current path segment is near
        if self.current_nav_path_wps:
            # Check distance to the *actual current TSP goal*, not just end of path segment
            dist_to_tsp_goal_sq = (current_position.x - self.tsp_goals[self.current_tsp_goal_idx].x)**2 + \
                                  (current_position.y - self.tsp_goals[self.current_tsp_goal_idx].y)**2
            
            if dist_to_tsp_goal_sq < self.waypoint_threshold_sq:
                self.get_logger().info(f"Reached TSP Goal {self.current_tsp_goal_idx}.")
                self.current_tsp_goal_idx += 1
                self.needs_new_path_segment = True # Need to plan to the next TSP goal
                if self.current_tsp_goal_idx >= len(self.tsp_goals):
                    self.get_logger().info("All TSP goals achieved!")
                    self.all_goals_done = True
                    self.complete_pub.publish(Bool(data=True))
                    self.current_nav_path_wps = [] # Clear path
                    self._publish_nav_path_from_wps([]) # Publish empty path
                else:
                    self.get_logger().info(f"Moving to next TSP Goal {self.current_tsp_goal_idx}.")
                    self._publish_current_target_tsp_goal() # Update target visualization
            # Also, if near the end of the current *segment* but not yet at the TSP goal, might replan
            # This logic can be refined based on how GRP segments are used.
            # For now, we only replan when a TSP goal is reached or if planning failed.

    def _publish_nav_path_from_wps(self, carla_wps: List[carla.Waypoint]) -> None:
        path_msg = NavPath()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map" # Or your odometry frame

        for wp in carla_wps:
            pose = PoseStamped()
            pose.header.stamp = path_msg.header.stamp 
            pose.header.frame_id = path_msg.header.frame_id
            pose.pose.position.x = wp.transform.location.x
            pose.pose.position.y = wp.transform.location.y
            pose.pose.position.z = wp.transform.location.z #CARLA waypoints have z
            # Orientation can be derived from waypoint transform if needed, or path direction
            # For now, keeping orientation as default (0,0,0,1)
            q = carla_rotation_to_ros_quaternion(wp.transform.rotation)
            pose.pose.orientation.x = q.x
            pose.pose.orientation.y = q.y
            pose.pose.orientation.z = q.z
            pose.pose.orientation.w = q.w
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)
        # self.get_logger().debug(f"Published navigation path with {len(path_msg.poses)} poses.")

def carla_rotation_to_ros_quaternion(carla_rotation: carla.Rotation) -> Quaternion:
    """Converts a CARLA rotation to a ROS quaternion."""
    # CARLA rotation: pitch (Y), yaw (Z), roll (X) in degrees
    # ROS quaternion: x, y, z, w
    # tf.transformations.quaternion_from_euler can be used if available and preferred.
    # Manual conversion:
    roll = math.radians(carla_rotation.roll)
    pitch = math.radians(carla_rotation.pitch)
    yaw = math.radians(carla_rotation.yaw)

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q

def main(args=None):
    rclpy.init(args=args)
    node: Optional[PlannerNode] = None
    try:
        node = PlannerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info(f"Node '{node.get_name()}' shutting down due to KeyboardInterrupt.")
        else:
            rclpy.logging.get_logger("planning_node_main").info("Node shutting down during initialization due to KeyboardInterrupt.")
    except SystemExit as e: # To catch SystemExit from __init__
        # Node might not be fully initialized if SystemExit came from __init__
        logger_name = node.get_name() if node else "planning_node_main"
        rclpy.logging.get_logger(logger_name).fatal(f"Node exited with SystemExit: {e}")
    except Exception as e:
        logger = rclpy.logging.get_logger("planning_node_main")
        if node: # If node was initialized before exception
            logger = node.get_logger()
        logger.fatal(f"Unhandled exception in PlannerNode: {e}\n{traceback.format_exc()}")
    finally:
        if node and rclpy.ok():
            node.get_logger().info(f"Destroying node '{node.get_name()}'.")
            node.destroy_node()
        if rclpy.ok():
            rclpy.logging.get_logger("planning_node_main").info("Shutting down rclpy.")
            rclpy.shutdown()

if __name__ == '__main__':
    main()
