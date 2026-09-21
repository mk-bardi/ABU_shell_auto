#!/usr/bin/env python3

import math
from typing import Optional, Tuple
import traceback

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import struct

class PerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__('perception_node')

        self.declare_parameter('obstacle_distance_threshold', 3.0)
        self.declare_parameter('roi_x_start_ratio', 0.38) # Region of Interest X start (percentage)
        self.declare_parameter('roi_x_end_ratio', 0.62)   # Region of Interest X end (percentage)
        self.declare_parameter('roi_y_start_ratio', 0.45) # Region of Interest Y start (percentage)
        self.declare_parameter('roi_y_end_ratio', 0.80)   # Region of Interest Y end (percentage)
        self.declare_parameter('use_lidar', True)
        self.declare_parameter('lidar_max_fwd_angle_deg', 30.0) # Max forward angle for LiDAR points
        # Height band for LiDAR returns, in the sensor's own frame. Without this
        # the downward beams strike the road a few metres ahead and are reported
        # as an obstacle, which brakes the car to a permanent standstill.
        self.declare_parameter('lidar_mount_height', 2.4)     # metres above the road
        self.declare_parameter('obstacle_min_height', 0.30)   # ignore returns this close to the road
        self.declare_parameter('obstacle_max_height', 3.00)   # ignore overhead signs and bridges
        self.declare_parameter('alert_latch_sec', 0.5) # How long an alert persists

        self.threshold: float = self.get_parameter('obstacle_distance_threshold').value
        self.roi_x_start_ratio: float = self.get_parameter('roi_x_start_ratio').value
        self.roi_x_end_ratio: float = self.get_parameter('roi_x_end_ratio').value
        self.roi_y_start_ratio: float = self.get_parameter('roi_y_start_ratio').value
        self.roi_y_end_ratio: float = self.get_parameter('roi_y_end_ratio').value
        self.use_lidar: bool = self.get_parameter('use_lidar').value
        self.lidar_fwd_rad: float = math.radians(self.get_parameter('lidar_max_fwd_angle_deg').value)
        mount_height: float = self.get_parameter('lidar_mount_height').value
        # Sensor frame: z grows upwards, so the road sits at -mount_height.
        self.lidar_min_z: float = -mount_height + self.get_parameter('obstacle_min_height').value
        self.lidar_max_z: float = -mount_height + self.get_parameter('obstacle_max_height').value
        self._lidar_lowest_z_seen: float = float('inf')
        self.alert_latch_sec: float = self.get_parameter('alert_latch_sec').value

        latched_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.alert_pub = self.create_publisher(Bool, '/obstacle_alert', latched_qos)
        self.dist_pub = self.create_publisher(Float32, '/nearest_obstacle_distance', latched_qos)

        sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1, # Process latest sensor data
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        self.bridge = CvBridge()
        self._img_shape: Optional[Tuple[int, int]] = None # (height, width)
        self._roi_px: Optional[Tuple[int, int, int, int]] = None # (x_start, x_end, y_start, y_end)
        self._last_alert_time = self.get_clock().now()
        self._lidar_message_received_recently = False
        self._last_lidar_msg_time = self.get_clock().now()

        if self.use_lidar:
            self.lidar_sub = self.create_subscription(
                PointCloud2,
                '/carla/ego_vehicle/vlp16_1', # Make sure this topic name is correct
                self.lidar_callback,
                sensor_qos
            )
            self.get_logger().info("LiDAR mode enabled. Depth camera will be secondary if LiDAR messages are frequent.")
        else:
            self.lidar_sub = None
            self.get_logger().info("Depth-camera only mode enabled.")

        # Depth camera subscription is always active as a fallback or primary
        self.depth_sub = self.create_subscription(
            Image,
            '/carla/ego_vehicle/depth_middle/image', # Make sure this topic name is correct
            self.depth_callback,
            sensor_qos
        )
        self.get_logger().info("Perception node ready.")


    def depth_callback(self, msg: Image) -> None:
        # If using LiDAR and messages are recent, skip depth processing
        if self.use_lidar and (self.get_clock().now() - self._last_lidar_msg_time).nanoseconds * 1e-9 < 1.0 : # 1 second timeout
            if not self._lidar_message_received_recently: # Log once when switching
                self.get_logger().info("LiDAR active, skipping depth processing.")
                self._lidar_message_received_recently = True
            return 
        
        if self._lidar_message_received_recently and self.use_lidar: # Log once when switching back
            self.get_logger().info("No recent LiDAR data, using depth camera.")
            self._lidar_message_received_recently = False


        try:
            # CV bridge expects meters, in float32. CARLA depth is often logarithmic or in other units.
            # Assuming CARLA depth camera gives depth in meters directly (might need conversion)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough') # or '32FC1'
            if msg.encoding == 'bgra8': # This indicates raw depth data, needs interpretation
                 # Example: scale to meters if it's raw uint16, not directly applicable to bgra8 without more info
                pass # cv_image = (cv_image[:, :, 0] + cv_image[:, :, 1]*256 + cv_image[:, :, 2]*256*256) * (1.0/ (256**3 -1)) * MaxRange
            elif msg.encoding != '32FC1': # If not float meters directly
                self.get_logger().warn_once(f"Depth image encoding is {msg.encoding}. Assuming it's in meters. If not, conversion is needed.")

        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        if self._img_shape is None or self._img_shape != (msg.height, msg.width):
            self._img_shape = (msg.height, msg.width)
            h, w = self._img_shape
            self._roi_px = (
                int(self.roi_x_start_ratio * w),
                int(self.roi_x_end_ratio * w),
                int(self.roi_y_start_ratio * h),
                int(self.roi_y_end_ratio * h)
            )
            self.get_logger().info(f"Depth ROI initialized to: x=[{self._roi_px[0]}-{self._roi_px[1]}], y=[{self._roi_px[2]}-{self._roi_px[3]}]")

        if self._roi_px is None: return

        roi_x_start, roi_x_end, roi_y_start, roi_y_end = self.roi_px
        
        # Ensure ROI is valid
        if roi_x_start >= roi_x_end or roi_y_start >= roi_y_end:
            self.get_logger().error_once("Invalid ROI dimensions, check ratios.")
            return

        depth_roi = cv_image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        if depth_roi.size == 0:
            # self.get_logger().warn_once("Depth ROI is empty.")
            self._publish_obstacle(False, float('inf'))
            return

        min_dist = np.min(depth_roi[depth_roi > 0.01]) if np.any(depth_roi > 0.01) else float('inf')

        self._publish_obstacle(min_dist < self.threshold, min_dist)


    def lidar_callback(self, msg: PointCloud2) -> None:
        if not self.use_lidar:
            return
        
        self._last_lidar_msg_time = self.get_clock().now()
        if not self._lidar_message_received_recently: # Log once
            self.get_logger().info("Processing LiDAR data.")
            self._lidar_message_received_recently = True


        nearest = float('inf')
        cos_fov_half_angle = math.cos(self.lidar_fwd_rad / 2.0)
        
        # Assuming 'x', 'y', 'z' fields are present and are float32
        # Check msg.fields to confirm offsets and datatypes
        # Example: field 'x' might be at offset 0, 'y' at 4, 'z' at 8
        # This part needs to be robust to the actual PointCloud2 structure
        
        # A more robust way to parse PointCloud2:
        # from sensor_msgs_py import point_cloud2
        # for point in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        #     px, py, pz = point[0], point[1], point[2]
        # This requires sensor_msgs_py. For now, using struct as in original if offsets are known.

        # Assuming standard VLP16 structure where x,y,z are the first three float32 fields
        point_step = msg.point_step
        for i in range(0, len(msg.data), point_step):
            # Adjust offsets based on your PointCloud2 fields
            # Example: if x, y, z are the first 3 float32 fields:
            try:
                px, py, pz = struct.unpack_from('fff', msg.data, offset=i) # Check field offsets from msg.fields!
            except struct.error:
                self.get_logger().warn_once("Struct unpack error in LiDAR. Check PointCloud2 fields and offsets.")
                break # Stop processing this message

            if px <= 0.1:  # Ignore points behind or too close to the sensor origin (forward axis)
                continue

            # Drop road returns and anything overhead. The beams angled downwards
            # hit the tarmac well within braking distance, and counting those as
            # obstacles means the vehicle never moves.
            if pz < self._lidar_lowest_z_seen:
                self._lidar_lowest_z_seen = pz
            if pz < self.lidar_min_z or pz > self.lidar_max_z:
                continue

            dist_sq_3d = px*px + py*py + pz*pz
            if dist_sq_3d < 0.01: # Ignore points extremely close to origin
                 continue
            
            # Check if point is within forward FOV (simplified cone check)
            # Assumes LiDAR 'x' is forward. If not, transform points or adjust logic.
            if px / math.sqrt(dist_sq_3d) < cos_fov_half_angle: # px / ||point|| is cos(angle_to_fwd_axis)
                continue
            
            current_dist = math.sqrt(dist_sq_3d)

            if current_dist < nearest:
                nearest = current_dist
                if nearest < 0.5: # Optimization: if something is super close, no need to check further
                    break
        
        self.get_logger().info(
            f"LiDAR height filter: keeping {self.lidar_min_z:.2f} m to "
            f"{self.lidar_max_z:.2f} m in sensor frame; lowest return seen "
            f"{self._lidar_lowest_z_seen:.2f} m. If the lowest return is far below "
            f"the filter floor it is the road, as expected; if obstacles are being "
            f"missed, lower lidar_mount_height.",
            throttle_duration_sec=10.0,
        )

        if nearest == float('inf'):
            self._publish_obstacle(False, float('inf'))
        else:
            self._publish_obstacle(nearest < self.threshold, nearest)


    def _publish_obstacle(self, detected: bool, distance: float) -> None:
        now = self.get_clock().now()
        current_alert_state = detected

        if detected:
            self._last_alert_time = now
        else:
            # Latch the alert for a short duration
            if (now - self._last_alert_time).nanoseconds * 1e-9 < self.alert_latch_sec:
                current_alert_state = True # Keep alert active during latch period

        self.alert_pub.publish(Bool(data=current_alert_state))
        self.dist_pub.publish(Float32(data=float(distance)))


def main(args=None):
    rclpy.init(args=args)
    node: Optional[PerceptionNode] = None
    try:
        node = PerceptionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info(f"Node '{node.get_name()}' shutting down due to KeyboardInterrupt.")
        else:
            rclpy.logging.get_logger("perception_node_main").info("Node shutting down during initialization due to KeyboardInterrupt.")
    except Exception as e:
        logger = rclpy.logging.get_logger("perception_node_main")
        if node:
            logger = node.get_logger()
        logger.fatal(f"Unhandled exception in PerceptionNode: {e}\n{traceback.format_exc()}")
    finally:
        if node and rclpy.ok():
            node.get_logger().info(f"Destroying node '{node.get_name()}'.")
            node.destroy_node()
        if rclpy.ok():
            rclpy.logging.get_logger("perception_node_main").info("Shutting down rclpy.")
            rclpy.shutdown()

if __name__ == '__main__':
    main()
