#!/usr/bin/env python3

"""Tests for the ROS/CARLA frame conversion.

Getting this wrong is silent: routes are planned from a mirrored position
and simply come back wrong, with no error anywhere. These cases pin the
convention against real measurements taken from Town01.
"""

import pytest

from shell_simulation.frames import (
    carla_to_ros_rpy,
    carla_to_ros_xyz,
    ros_to_carla_rpy,
    ros_to_carla_xyz,
)


def test_y_is_negated():
    assert ros_to_carla_xyz(281.12, -135.70, 0.5) == (281.12, 135.70, 0.5)
    assert carla_to_ros_xyz(281.12, 135.70, 0.5) == (281.12, -135.70, 0.5)


def test_x_and_z_are_untouched():
    x, y, z = ros_to_carla_xyz(1.5, 2.5, 3.5)
    assert (x, z) == (1.5, 3.5)


def test_position_round_trips():
    original = (281.12, -135.70, 0.5)
    assert carla_to_ros_xyz(*ros_to_carla_xyz(*original)) == original


def test_pitch_and_yaw_are_negated_roll_is_not():
    assert carla_to_ros_rpy(10.0, 20.0, 30.0) == (10.0, -20.0, -30.0)
    assert ros_to_carla_rpy(10.0, 20.0, 30.0) == (10.0, -20.0, -30.0)


def test_rotation_round_trips():
    original = (10.0, 20.0, 30.0)
    assert ros_to_carla_rpy(*carla_to_ros_rpy(*original)) == original


@pytest.mark.parametrize('ros_xy, carla_xy', [
    # Measured on Town01: converting y puts each point within a couple of
    # metres of a road, while leaving it alone lands 130-260 m away.
    ((281.12, -135.70), (281.12, 135.70)),   # vehicle at spawn-ish
    ((339.10, -258.60), (339.10, 258.60)),   # official goal 2
    ((334.949799, -161.106171), (334.949799, 161.106171)),   # official goal 1
    ((-1.646942, -197.501282), (-1.646942, 197.501282)),     # official goal 10
])
def test_measured_competition_points(ros_xy, carla_xy):
    assert ros_to_carla_xyz(ros_xy[0], ros_xy[1], 0.0)[:2] == carla_xy
    assert carla_to_ros_xyz(carla_xy[0], carla_xy[1], 0.0)[:2] == ros_xy


def test_origin_is_a_fixed_point():
    assert ros_to_carla_xyz(0.0, 0.0, 0.0) == (0.0, 0.0, 0.0)
