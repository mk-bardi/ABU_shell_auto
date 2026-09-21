#!/usr/bin/env python3

from __future__ import annotations
import math
import traceback
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from std_msgs.msg import Float32, Float64, Bool, String
from nav_msgs.msg import Odometry, Path as NavPath # Renamed to avoid conflict with pathlib.Path
from geometry_msgs.msg import Point

# The vehicle interface accepts only these two gear values.
GEAR_FORWARD = "forward"
GEAR_REVERSE = "reverse"


def _dist2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx, dy = a[0] - b[0], a[1] - b[1]
    return dx * dx + dy * dy

class PIDController:
    def __init__(self, k_p: float, k_i: float, k_d: float, max_integral: float, period: float):
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.max_integral = max_integral
        self.period = period
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, error: float) -> float:
        self.integral += error * self.period
        self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
        derivative = (error - self.prev_error) / self.period
        output = self.k_p * error + self.k_i * self.integral + self.k_d * derivative
        self.prev_error = error
        return output

class ControlNode(Node):
    def __init__(self) -> None:
        super().__init__('control_node')

        self.declare_parameter('wheel_base', 2.8)
        self.declare_parameter('pp_lookahead_time', 0.5)
        self.declare_parameter('pp_L_min', 2.0)
        self.declare_parameter('pp_L_max', 15.0)
        # speed_pid parameters are now fetched with a prefix
        self.declare_parameter('speed_pid.k_p', 0.8)
        self.declare_parameter('speed_pid.k_i', 0.1)
        self.declare_parameter('speed_pid.k_d', 0.0)
        self.declare_parameter('speed_pid.max_integral', 5.0)
        self.declare_parameter('throttle_jerk_max', 0.4) # Max change in throttle per second
        self.declare_parameter('brake_jerk_max', 0.6)    # Max change in brake per second
        self.declare_parameter('throttle_alpha', 0.2) # Smoothing factor for throttle/brake
        self.declare_parameter('steer_rate_max', 0.04) # Max change in steering command per control cycle
        self.declare_parameter('speed_limit_mps', 13.0) # Absolute speed cap
        self.declare_parameter('target_speed_mps', 8.0) # Desired cruise speed
        self.declare_parameter('control_period', 0.05) # Control loop period in seconds

        self.declare_parameter('obs_brake_full', 3.0) # Distance for full brake
        self.declare_parameter('obs_brake_start', 6.0) # Distance to start braking for obstacle
        self.declare_parameter('obs_critical_speed_threshold', 0.5) # Speed below which obstacle braking is more aggressive

        p = self.get_parameter
        self.Lf = p('wheel_base').value
        self.tau = p('control_period').value
        self.L_time = p('pp_lookahead_time').value
        self.L_min = p('pp_L_min').value
        self.L_max = p('pp_L_max').value

        pid_kp = p('speed_pid.k_p').value
        pid_ki = p('speed_pid.k_i').value
        pid_kd = p('speed_pid.k_d').value
        pid_max_integral = p('speed_pid.max_integral').value
        self.speed_pid = PIDController(pid_kp, pid_ki, pid_kd, pid_max_integral, self.tau)

        self.t_jerk_limit = p('throttle_jerk_max').value * self.tau # Max change per control cycle
        self.b_jerk_limit = p('brake_jerk_max').value * self.tau    # Max change per control cycle
        self.alpha = p('throttle_alpha').value # For LPF on throttle/brake commands
        self.steer_rate_max_cycle = p('steer_rate_max').value # Renamed for clarity

        self.speed_cap = p('speed_limit_mps').value
        self.speed_cruise = p('target_speed_mps').value # Target speed from params

        self.obs_full_brake_dist = p('obs_brake_full').value
        self.obs_start_brake_dist = p('obs_brake_start').value
        self.obs_critical_speed = p('obs_critical_speed_threshold').value


        self.pub_throttle = self.create_publisher(Float64, '/throttle_command', 10)
        self.pub_brake = self.create_publisher(Float64, '/brake_command', 10)
        self.pub_steer = self.create_publisher(Float64, '/steering_command', 10)
        # Latched: these are state, not a stream. A bridge that subscribes after
        # we start still needs the gear, or the car sits in neutral.
        drivetrain_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.pub_gear = self.create_publisher(String, '/gear_command', drivetrain_qos)
        self.pub_handbrake = self.create_publisher(Bool, '/handbrake_command', drivetrain_qos)


        sensor_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        # The planner and perception nodes publish these latched (transient local):
        # they send a message only when something changes. A volatile subscription
        # that comes up after that publication never receives it, which would leave
        # the controller without a path and the car parked. Match their QoS so a
        # late-joining subscriber still gets the last value.
        latched_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.create_subscription(NavPath, '/nav_path', self.cb_path, qos_profile=latched_qos)
        self.create_subscription(Odometry, '/carla/ego_vehicle/odometry', self.cb_odom, qos_profile=sensor_qos)
        self.create_subscription(Float32, '/carla/ego_vehicle/speedometer', self.cb_speed, qos_profile=sensor_qos)
        self.create_subscription(Float32, '/nearest_obstacle_distance', self.cb_obs, qos_profile=latched_qos)
        self.create_subscription(Bool, '/mission_complete', self.cb_mission, qos_profile=latched_qos)

        self.path_xy: List[Tuple[float, float]] = []
        self.path_s: List[float] = []
        self.path_yaw: List[float] = [] # If path provides yaw

        self.vehicle_odom: Optional[Odometry] = None
        self.speed_mps: float = 0.0
        self.speed_mps_raw: float = 0.0 # From speedometer
        self.current_pos: Optional[Tuple[float, float]] = None
        self.current_yaw: float = 0.0

        self.obs_dist: float = float('inf')
        self.mission_done: bool = False

        self.pid_error_sum = 0.0
        self.pid_error_prev = 0.0
        
        self.prev_throttle: float = 0.0
        self.prev_brake: float = 0.0
        self.prev_steer: float = 0.0

        # The vehicle starts in neutral: throttle does nothing until a gear is
        # selected, which is why full throttle left the car stationary. The
        # interface accepts only "forward" or "reverse" -- "drive" is not valid.
        self._publish_drivetrain_state()

        self.control_timer = self.create_timer(self.tau, self.control_timer_callback)
        self.get_logger().info("Control node ready (Pure-Pursuit + PID).")

    def _publish_drivetrain_state(self) -> None:
        """Keep the car in forward gear with the handbrake off.

        Published every control cycle rather than once at start-up: publishing
        once races the bridge's subscriber, and a bridge restart would
        otherwise leave the vehicle in neutral with no way to recover.
        """
        self.pub_gear.publish(String(data=GEAR_FORWARD))
        self.pub_handbrake.publish(Bool(data=False))

    def cb_path(self, msg: NavPath) -> None:
        if not msg.poses:
            self.path_xy = []
            self.path_s = []
            self.path_yaw = []
            return

        new_path_xy = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        
        # Prevent path change if it's too similar to avoid oscillations (optional)
        # if self.path_xy and len(new_path_xy) > 0 and _dist2(self.path_xy[0], new_path_xy[0]) < 0.1:
        #     return

        self.path_xy = new_path_xy
        self.path_s = [0.0] * len(self.path_xy)
        self.path_yaw = [] # Recalculate or use from path if available

        for i in range(1, len(self.path_xy)):
            dist = math.sqrt(_dist2(self.path_xy[i], self.path_xy[i-1]))
            self.path_s[i] = self.path_s[i-1] + dist
        
        for i in range(len(msg.poses) -1):
            p1 = msg.poses[i].pose.position
            p2 = msg.poses[i+1].pose.position
            self.path_yaw.append(math.atan2(p2.y - p1.y, p2.x - p1.x))
        if msg.poses: # Add yaw for the last point (can be same as second to last)
            self.path_yaw.append(self.path_yaw[-1] if self.path_yaw else 0.0)


    def cb_odom(self, msg: Odometry) -> None:
        self.vehicle_odom = msg
        self.current_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        
        # Orientation to Yaw
        q = msg.pose.pose.orientation
        # Using simple yaw calculation, ensure it matches your vehicle's coordinate system
        # For ENU, yaw from quaternion: math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

        # Speed from odometry (more reliable than speedometer for control usually)
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.speed_mps = math.sqrt(vx*vx + vy*vy)
        # self.get_logger().info(f"Odom: pos=({self.current_pos[0]:.1f}, {self.current_pos[1]:.1f}), yaw={math.degrees(self.current_yaw):.1f}, speed={self.speed_mps:.1f} m/s")


    def cb_speed(self, msg: Float32) -> None:
        self.speed_mps_raw = msg.data # m/s

    def cb_obs(self, msg: Float32) -> None:
        self.obs_dist = msg.data

    def cb_mission(self, msg: Bool) -> None:
        self.mission_done = msg.data
        if self.mission_done:
            self.get_logger().info("Mission complete flag received.")

    def control_timer_callback(self) -> None:
        if self.current_pos is None or not self.path_xy:
            # self.get_logger().info("No odom or path yet, skipping control loop.")
            self._publish_drivetrain_state()
            self.pub_throttle.publish(Float64(data=0.0))
            self.pub_brake.publish(Float64(data=0.0)) # Publish zero brake if no path
            self.pub_steer.publish(Float64(data=0.0))
            return

        target_speed = self.speed_cruise
        
        # --- Pure Pursuit Steering Control ---
        lookahead_dist = self.L_time * self.speed_mps + self.L_min # Dynamic lookahead
        lookahead_dist = max(self.L_min, min(self.L_max, lookahead_dist))

        target_idx = len(self.path_xy) - 1 # Default to last point
        for i in range(len(self.path_xy) -1, -1, -1): # Search backwards
            dist_to_path_point = math.sqrt(_dist2(self.current_pos, self.path_xy[i]))
            if dist_to_path_point < lookahead_dist:
                # Check if this point or the next one is better if segment is crossed
                if i < len(self.path_xy) - 1:
                    # Simple check: if lookahead_dist is between this point and next
                    # More robust intersection logic might be needed
                    target_idx = i + 1 # Aim for the point just beyond lookahead distance
                else:
                    target_idx = i
                break
            target_idx = i # Keep track of closest point if no intersection found within lookahead

        if target_idx >= len(self.path_xy): # Should not happen with corrected logic
            target_idx = len(self.path_xy) - 1

        target_wp_x, target_wp_y = self.path_xy[target_idx]

        alpha_pp = math.atan2(target_wp_y - self.current_pos[1], target_wp_x - self.current_pos[0]) - self.current_yaw
        steer_cmd_raw = math.atan2(2.0 * self.Lf * math.sin(alpha_pp), lookahead_dist)
        steer_cmd = max(-1.0, min(1.0, steer_cmd_raw)) # Clamp to [-1, 1] range for normalized steering

        # Steering rate limit
        steer_diff = steer_cmd - self.prev_steer
        if abs(steer_diff) > self.steer_rate_max_cycle:
            steer_cmd = self.prev_steer + math.copysign(self.steer_rate_max_cycle, steer_diff)
        steer_cmd = max(-1.0, min(1.0, steer_cmd))


        # --- Speed Control (PID) ---
        speed_error = target_speed - self.speed_mps
        pid_output = self.speed_pid.step(speed_error)
        
        # Convert PID output to throttle/brake
        # This is a common way: positive PID output -> throttle, negative -> brake
        # Needs tuning.
        if pid_output > 0:
            throttle_cmd = max(0.0, min(1.0, pid_output))
            brake_cmd = 0.0
        else:
            throttle_cmd = 0.0
            brake_cmd = max(0.0, min(1.0, -pid_output * 0.5)) # Scale brake response

        # Smooth throttle and brake
        final_throttle = (1 - self.alpha) * self.prev_throttle + self.alpha * throttle_cmd
        final_brake = (1 - self.alpha) * self.prev_brake + self.alpha * brake_cmd

        # Jerk limits for throttle and brake
        throttle_jerk = final_throttle - self.prev_throttle
        if abs(throttle_jerk) > self.t_jerk_limit:
            final_throttle = self.prev_throttle + math.copysign(self.t_jerk_limit, throttle_jerk)

        brake_jerk = final_brake - self.prev_brake
        if abs(brake_jerk) > self.b_jerk_limit:
            final_brake = self.prev_brake + math.copysign(self.b_jerk_limit, brake_jerk)
            
        final_throttle = max(0.0, min(1.0, final_throttle))
        final_brake = max(0.0, min(1.0, final_brake))


        # --- Obstacle Avoidance ---
        obs_brake_factor = 0.0
        if self.obs_dist < self.obs_start_brake_dist:
            if self.obs_dist <= self.obs_full_brake_dist:
                obs_brake_factor = 1.0
            else:
                obs_brake_factor = (self.obs_start_brake_dist - self.obs_dist) / \
                                   (self.obs_start_brake_dist - self.obs_full_brake_dist)
            
            final_brake = max(final_brake, obs_brake_factor)

            # Hold the car only for an obstacle genuinely inside the full-brake
            # distance. The previous rule fired at 1.5x that distance whenever
            # speed was low, which is self-sustaining: the car braked because it
            # was stopped and stayed stopped because it was braking, so a single
            # spurious reading parked it for good. Anything between the full-brake
            # and start-brake distances is left to the proportional factor above,
            # which still slows the car without pinning it at zero.
            if self.obs_dist <= self.obs_full_brake_dist and self.speed_mps < self.obs_critical_speed:
                 self.get_logger().warning(
                    f"Obstacle within {self.obs_full_brake_dist:.1f}m ({self.obs_dist:.1f}m) at "
                    f"{self.speed_mps:.1f}m/s. Holding.",
                    throttle_duration_sec=2.0,
                )
                 final_throttle = 0.0
                 final_brake = max(final_brake, 0.8) # Stronger brake

        # Ensure throttle and brake are not applied simultaneously
        if final_brake > 0.05:
            final_throttle = 0.0
        if final_throttle > 0.05: # if throttle is applied, no brake
            final_brake = 0.0
        
        # Speed cap
        if self.speed_mps > self.speed_cap:
            final_throttle = 0.0
            final_brake = max(final_brake, 0.3) # Apply some brake if over speed cap

        # Mission completion handling
        if self.mission_done:
            final_throttle = 0.0
            if self.speed_mps < 0.3: # If almost stopped
                final_brake = 1.0 # Full stop
            else:
                final_brake = max(final_brake, 0.5) # Gentle stop

        final_throttle = max(0.0, min(1.0, final_throttle))
        final_brake = max(0.0, min(1.0, final_brake))

        self._publish_drivetrain_state()
        self.pub_steer.publish(Float64(data=steer_cmd))
        self.pub_throttle.publish(Float64(data=final_throttle))
        self.pub_brake.publish(Float64(data=final_brake))

        self.prev_throttle, self.prev_brake, self.prev_steer = \
            final_throttle, final_brake, steer_cmd


def main(args=None):
    rclpy.init(args=args)
    node: Optional[ControlNode] = None
    try:
        node = ControlNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info(f"Node '{node.get_name()}' shutting down due to KeyboardInterrupt.")
        else:
            rclpy.logging.get_logger("control_node_main").info("Node shutting down during initialization due to KeyboardInterrupt.")
    except Exception as e:
        logger = rclpy.logging.get_logger("control_node_main")
        if node:
            logger = node.get_logger()
        logger.fatal(f"Unhandled exception in ControlNode: {e}\n{traceback.format_exc()}")
    finally:
        if node and rclpy.ok():
            node.get_logger().info(f"Destroying node '{node.get_name()}'.")
            node.destroy_node()
        if rclpy.ok(): # Ensure shutdown is called if rclpy was initialized
            rclpy.logging.get_logger("control_node_main").info("Shutting down rclpy.")
            rclpy.shutdown()

if __name__ == '__main__':
    main()
