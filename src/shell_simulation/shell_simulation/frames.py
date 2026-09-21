#!/usr/bin/env python3

"""Conversion between the ROS and CARLA coordinate frames.

CARLA is left-handed with y to the right; ROS follows REP-103, right-handed
with y to the left. The carla_ros_bridge negates y when it publishes, so
anything read from a ROS topic must be converted before it is handed to the
CARLA API, and anything read from CARLA must be converted before publishing.

Measured against Town01 with the vehicle at ROS (281.12, -135.70):

    y as-is    -> nearest road 133.67 m away
    y negated  -> nearest road   2.20 m away

Skipping the conversion does not fail loudly. It silently plans routes from
a mirrored position, which is how every route in this package came to start
133 m from the car.

Kept free of ROS and CARLA imports so it can be tested directly.
"""


def ros_to_carla_xyz(x: float, y: float, z: float):
    """ROS position -> CARLA position."""
    return x, -y, z


def carla_to_ros_xyz(x: float, y: float, z: float):
    """CARLA position -> ROS position. The transform is its own inverse."""
    return x, -y, z


def carla_to_ros_rpy(roll: float, pitch: float, yaw: float):
    """CARLA rotation -> ROS rotation, in whatever unit is passed in.

    Mirroring y reverses the sense of rotation about the other two axes, so
    pitch and yaw change sign while roll does not.
    """
    return roll, -pitch, -yaw


def ros_to_carla_rpy(roll: float, pitch: float, yaw: float):
    """ROS rotation -> CARLA rotation. Also its own inverse."""
    return roll, -pitch, -yaw
