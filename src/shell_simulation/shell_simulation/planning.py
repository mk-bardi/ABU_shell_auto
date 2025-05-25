#!/usr/bin/env python3
import rclpy 
from rclpy.node import Node 
from rclpy.qos import QoSProfile, ReliabilityPolicy
import math
import heapq
import time
import atexit
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String

class PlanningNode(Node):  
    def __init__(self):
        super().__init__("planning")
        
        # Declare parameters - flatten waypoints list for ROS parameter compatibility
        self.declare_parameter('distance_threshold', 2.0)
        self.declare_parameter('log_filename', 'vehicle_path.log')
        self.declare_parameter('start_position', [280.363739, -129.306351, 0.101746])
        
        # Enhanced lane detection parameters
        self.declare_parameter('require_lanes', True)  # Strict lane requirement
        self.declare_parameter('lane_check_distance', 5.0)  # Distance to check ahead for lanes
        self.declare_parameter('max_no_lane_waypoints', 2)  # Max consecutive waypoints without lanes
        
        # Flatten the waypoints into a single list (each waypoint is 3 consecutive values: x, y, z)
        flattened_waypoints = [
            334.949799, -161.106171, 0.001736,
            339.100037, -258.568939, 0.001679,
            396.295319, -183.195740, 0.001678,
            267.657074, -1.983160, 0.001678,
            153.868896, -26.115866, 0.001678,
            290.515564, -56.175072, 0.001677,
            92.325722, -86.063644, 0.001677,
            88.384346, -287.468567, 0.001728,
            177.594101, -326.386902, 0.001677,
            -1.646942, -197.501282, 0.001555,
            59.701321, -1.970804, 0.001467,
            122.100121, -55.142044, 0.001596,
            161.030975, -129.313187, 0.001679,
            184.758713, -199.424271, 0.001680
        ]
        self.declare_parameter('waypoints_flat', flattened_waypoints)

        # Get parameter values
        self.start_point = self.get_parameter('start_position').value
        waypoints_flat = self.get_parameter('waypoints_flat').value
        
        # Reshape flattened waypoints back to list of [x, y, z] coordinates
        self.waypoints = []
        for i in range(0, len(waypoints_flat), 3):
            self.waypoints.append([waypoints_flat[i], waypoints_flat[i+1], waypoints_flat[i+2]])
        
        # Get enhanced parameters
        self.require_lanes = self.get_parameter('require_lanes').value
        self.lane_check_distance = self.get_parameter('lane_check_distance').value
        self.max_no_lane_waypoints = self.get_parameter('max_no_lane_waypoints').value
        
        self.distance_threshold = self.get_parameter('distance_threshold').value
        log_filename = self.get_parameter('log_filename').value

        # Initialize waypoints and distance matrix
        self.points = [self.start_point] + self.waypoints
        self.distance_matrix = self.precompute_distances()
        
        # Solve TSP
        self.optimal_path = self.solve_tsp()
        self.current_waypoint_idx = 0
        self.obstacle_detected = False
        self.lane_detected = True
        self.traffic_light_state = "green"
        self.current_pose = None
        
        # State monitoring with lane tracking
        self.planner_active = True
        self.consecutive_no_lane_count = 0
        self.waypoint_lane_status = {}  # Track lane availability per waypoint
        self.skip_waypoints = set()  # Waypoints to skip due to no lanes

        # QoS profile for simulation compatibility
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        
        # ROS Publishers
        self.path_pub = self.create_publisher(Path, '/planning/path', qos_profile)
        self.next_wp_pub = self.create_publisher(PoseStamped, '/planning/next_waypoint', qos_profile)

        # ROS Subscribers
        self.obstacle_sub = self.create_subscription(
            Bool, '/perception/obstacles', self.obstacle_callback, qos_profile)
        self.lane_sub = self.create_subscription(
            Bool, '/perception/lane_detected', self.lane_callback, qos_profile)
        self.traffic_light_sub = self.create_subscription(
            String, '/perception/traffic_light', self.traffic_light_callback, qos_profile)
        self.odom_sub = self.create_subscription(
            Odometry, '/carla/ego_vehicle/odometry', self.odom_callback, qos_profile)

        # Logging setup
        self.log_file = open(log_filename, "w")
        self.log_file.write("timestamp,x,y,z\n")
        atexit.register(self.close_log_file)
        self.log_timer = self.create_timer(0.5, self.log_position)

        # Initial publishing
        if self.optimal_path:
            self.publish_path()
            self.publish_next_waypoint()
            self.get_logger().info(f"Generated optimal path with {len(self.optimal_path)} waypoints")
        else:
            self.get_logger().error("Failed to generate optimal path!")

    def close_log_file(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def destroy_node(self):
        self.close_log_file()
        super().destroy_node()

    def precompute_distances(self):
        n = len(self.points)
        return [[math.dist(p1, p2) for p2 in self.points] for p1 in self.points]

    def solve_tsp(self):
        initial_mask = 0
        heap = []
        heuristic = self.calculate_heuristic(0, initial_mask)
        heapq.heappush(heap, (heuristic, 0, 0, initial_mask, [0]))
        
        visited = {}
        
        while heap:
            priority, cost, current, mask, path = heapq.heappop(heap)
            
            if mask == (1 << len(self.waypoints)) - 1:
                # Return path including return to start point for complete TSP
                return [self.points[i] for i in path] + [self.points[0]]
            
            state_key = (current, mask)
            if state_key in visited and visited[state_key] <= cost:
                continue
            visited[state_key] = cost
            
            for next_wp in range(len(self.waypoints)):
                if not (mask & (1 << next_wp)):
                    next_index = next_wp + 1
                    new_cost = cost + self.distance_matrix[current][next_index]
                    new_mask = mask | (1 << next_wp)
                    new_path = path + [next_index]
                    new_heuristic = self.calculate_heuristic(next_index, new_mask)
                    total_priority = new_cost + new_heuristic
                    
                    heapq.heappush(heap, (total_priority, new_cost, next_index, new_mask, new_path))
        
        return None

    def calculate_heuristic(self, current, mask):
        unvisited = [i+1 for i in range(len(self.waypoints)) if not (mask & (1 << i))]
        return self.mst_cost(unvisited) if unvisited else 0

    def mst_cost(self, nodes):
        if len(nodes) <= 1:
            return 0
            
        edges = []
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                edges.append((self.distance_matrix[nodes[i]][nodes[j]], i, j))
        edges.sort()
        
        parent = list(range(len(nodes)))
        total = 0
        edge_count = 0
        
        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]
        
        for cost, u, v in edges:
            if edge_count == len(nodes) - 1:
                break
            pu, pv = find(u), find(v)
            if pu != pv:
                parent[pu] = pv
                total += cost
                edge_count += 1
                
        return total

    def publish_path(self):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        
        for point in self.optimal_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = point[2]
            pose.pose.orientation.w = 1.0  # Identity orientation
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)
        self.get_logger().info(f"Published path with {len(self.optimal_path)} waypoints")

    def publish_next_waypoint(self):
        if self.current_waypoint_idx < len(self.optimal_path):
            wp = self.optimal_path[self.current_waypoint_idx]
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "map"
            pose.pose.position.x = wp[0]
            pose.pose.position.y = wp[1]
            pose.pose.position.z = wp[2]
            
            # Calculate orientation to next waypoint if available
            if self.current_waypoint_idx < len(self.optimal_path) - 1:
                next_wp = self.optimal_path[self.current_waypoint_idx + 1]
                dx = next_wp[0] - wp[0]
                dy = next_wp[1] - wp[1]
                yaw = math.atan2(dy, dx)
                
                # Convert yaw to quaternion
                pose.pose.orientation.x = 0.0
                pose.pose.orientation.y = 0.0
                pose.pose.orientation.z = math.sin(yaw / 2.0)
                pose.pose.orientation.w = math.cos(yaw / 2.0)
            else:
                pose.pose.orientation.w = 1.0  # Identity orientation for last waypoint
                
            self.next_wp_pub.publish(pose)
            self.get_logger().info(f"Published waypoint {self.current_waypoint_idx} at ({wp[0]:.2f}, {wp[1]:.2f})")

    def obstacle_callback(self, msg):
        self.obstacle_detected = msg.data
        if self.obstacle_detected:
            self.planner_active = False
            self.get_logger().warn("Obstacle detected! Planner paused.")
        else:
            self.planner_active = True
            self.get_logger().info("Obstacle cleared. Planner resumed.")

    def lane_callback(self, msg):
        self.lane_detected = msg.data
        
        # Track lane status for current waypoint area
        if self.current_pose and self.optimal_path:
            self.waypoint_lane_status[self.current_waypoint_idx] = self.lane_detected
        
        if not self.lane_detected:
            self.consecutive_no_lane_count += 1
            self.planner_active = False
            self.get_logger().warn(f"Lane tracking lost! Count: {self.consecutive_no_lane_count}")
            
            # If too many consecutive waypoints without lanes, try to skip ahead
            if self.consecutive_no_lane_count >= self.max_no_lane_waypoints:
                self.handle_no_lane_situation()
        else:
            self.consecutive_no_lane_count = 0
            if self.require_lanes:
                self.planner_active = True
                self.get_logger().info("Lane tracking restored. Planner resumed.")
            else:
                self.planner_active = True  # Always allow if lanes not required

    def traffic_light_callback(self, msg):
        new_state = msg.data.lower()
        if new_state != self.traffic_light_state:
            self.get_logger().info(f"Traffic light changed to {new_state}")
            self.traffic_light_state = new_state
            
            if new_state == "red":
                self.planner_active = False
                self.get_logger().warn("Red light detected! Planner paused.")
            elif new_state == "green":
                self.planner_active = True
                self.get_logger().info("Green light! Planner resumed.")

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        
        # Check if we have a valid path to follow
        if not self.optimal_path:
            return
            
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        target = self.optimal_path[self.current_waypoint_idx]
        distance = math.hypot(target[0] - current_x, target[1] - current_y)
        
        # Enhanced waypoint advancement with lane validation
        if (distance < self.distance_threshold and 
            self.current_waypoint_idx < len(self.optimal_path) - 1):
            
            # Check if current waypoint should be skipped due to no lanes
            if self.current_waypoint_idx in self.skip_waypoints:
                self.get_logger().info(f"Skipping waypoint {self.current_waypoint_idx} (no lanes detected)")
                self.current_waypoint_idx += 1
                self.publish_next_waypoint()
                return
            
            # Standard safety checks + enhanced lane validation
            if (not self.obstacle_detected and 
                self.traffic_light_state == "green" and
                self.is_lane_safe_for_advancement()):
                
                self.current_waypoint_idx += 1
                self.publish_next_waypoint()
                self.get_logger().info(f"Advanced to waypoint {self.current_waypoint_idx}")
                
        elif not self.planner_active:
            # Enhanced logging for why we're not moving
            reasons = []
            if self.obstacle_detected:
                reasons.append("obstacle detected")
            if not self.lane_detected and self.require_lanes:
                reasons.append("no lane markings")
            if self.traffic_light_state == "red":
                reasons.append("red traffic light")
            
            if reasons:
                self.get_logger().debug(f"Planner paused: {', '.join(reasons)}")
    
    def is_lane_safe_for_advancement(self):
        """Enhanced lane safety check for waypoint advancement"""
        if not self.require_lanes:
            return True  # Skip lane checks if not required
            
        # Current lane must be detected
        if not self.lane_detected:
            return False
            
        # Check if next waypoint area has known lane issues
        next_wp_idx = self.current_waypoint_idx + 1
        if next_wp_idx in self.waypoint_lane_status:
            if not self.waypoint_lane_status[next_wp_idx]:
                self.get_logger().warn(f"Next waypoint {next_wp_idx} has no lanes detected previously")
                return False
                
        return True
    
    def handle_no_lane_situation(self):
        """Handle situations where lanes are consistently not detected"""
        self.get_logger().warn(f"No lanes detected for {self.consecutive_no_lane_count} consecutive checks")
        
        if not self.require_lanes:
            # If lanes not strictly required, continue with caution
            self.planner_active = True
            self.get_logger().info("Continuing without lane detection (not required)")
            return
            
        # Try to find next waypoint with lanes
        next_valid_waypoint = self.find_next_lane_waypoint()
        
        if next_valid_waypoint is not None:
            # Mark intermediate waypoints as skip
            for i in range(self.current_waypoint_idx, next_valid_waypoint):
                self.skip_waypoints.add(i)
                
            self.get_logger().info(f"Rerouting: skipping to waypoint {next_valid_waypoint} with lanes")
            self.consecutive_no_lane_count = 0
            self.planner_active = True
        else:
            self.get_logger().error("No waypoints with lanes found ahead! Manual intervention required.")
            
    def find_next_lane_waypoint(self):
        """Find the next waypoint where lanes were previously detected"""
        for i in range(self.current_waypoint_idx + 1, len(self.optimal_path)):
            if i in self.waypoint_lane_status and self.waypoint_lane_status[i]:
                return i
        return None

    def log_position(self):
        if self.current_pose and self.log_file:
            x = self.current_pose.position.x
            y = self.current_pose.position.y
            z = self.current_pose.position.z
            self.log_file.write(f"{time.time()},{x},{y},{z}\n")
            self.log_file.flush()  # Ensure real-time logging
    
    def replan_path(self):
        """Method to trigger replanning if needed for dynamic environments"""
        self.get_logger().info("Replanning path...")
        new_path = self.solve_tsp()
        if new_path:
            self.optimal_path = new_path
            self.current_waypoint_idx = 0
            self.publish_path()
            self.publish_next_waypoint()
            self.get_logger().info("Path replanned successfully")
        else:
            self.get_logger().error("Failed to replan path!")

def main(args=None):
    rclpy.init(args=args)
    node = PlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down planning node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()