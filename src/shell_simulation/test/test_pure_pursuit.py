#!/usr/bin/env python3

"""Tests for the pure-pursuit steering law.

Against a live simulator the car sat at full throttle with the steering
pinned at -1.0, scrubbing its tyres and creeping at 1.3 cm/s. Three faults
combined: the law's output (a road-wheel angle in radians) was used directly
as a normalised command, the lookahead distance was shorter than the
wheelbase, and a target behind the vehicle produced almost no steering.
"""

import math

import pytest

from shell_simulation.steering import pure_pursuit_steer

WHEEL_BASE = 2.8
MAX_STEER = 1.22
LOOKAHEAD = 5.0


def steer(deg, lookahead=LOOKAHEAD):
    return pure_pursuit_steer(math.radians(deg), lookahead, WHEEL_BASE, MAX_STEER)


def test_straight_ahead_needs_no_steering():
    assert steer(0.0) == pytest.approx(0.0)


def test_output_is_always_a_valid_command():
    for deg in range(-360, 361, 5):
        assert -1.0 <= steer(deg) <= 1.0


def test_sign_follows_the_target():
    assert steer(30) > 0      # target left -> steer left
    assert steer(-30) < 0     # target right -> steer right


def test_response_is_monotonic_before_saturation():
    values = [steer(d) for d in (5, 10, 20, 30, 40)]
    assert values == sorted(values)


def test_modest_heading_error_does_not_saturate():
    """The original bug: anything past ~21 degrees pinned the wheels."""
    assert abs(steer(21)) < 0.5
    assert abs(steer(30)) < 0.7


def test_target_behind_commands_near_full_lock():
    """sin() collapses near +-pi, so this case needs explicit handling."""
    assert steer(170) > 0.9
    assert steer(-170) < -0.9
    # Dead astern: full lock, but which way round is arbitrary and decided by
    # the floating-point sign of the wrapped angle.
    assert abs(steer(180)) == pytest.approx(1.0)
    assert abs(steer(-180)) == pytest.approx(1.0)


def test_response_is_continuous_across_the_ninety_degree_boundary():
    """A step here would fight the controller's steering rate limit."""
    just_under, just_over = steer(89.999), steer(90.001)
    assert abs(just_over - just_under) < 0.01


def test_magnitude_keeps_growing_behind_the_vehicle():
    values = [steer(d) for d in (90, 110, 130, 150, 170, 180)]
    assert values == sorted(values)


def test_angle_wrapping():
    """A heading error is the same turn whichever way it is expressed."""
    for deg in (-150, -90, -30, 0, 30, 90, 150):
        assert steer(deg) == pytest.approx(steer(deg + 360), abs=1e-9)
        assert steer(deg) == pytest.approx(steer(deg - 360), abs=1e-9)


def test_longer_lookahead_softens_steering():
    """Why pp_L_min must exceed the wheelbase."""
    assert abs(steer(30, lookahead=2.0)) > abs(steer(30, lookahead=5.0))
    assert abs(steer(30, lookahead=5.0)) > abs(steer(30, lookahead=15.0))


def test_lookahead_shorter_than_wheelbase_saturates():
    """Documents the geometry that caused the tyre scrubbing."""
    assert abs(steer(30, lookahead=2.0)) > 0.75


# -- vehicle command convention ------------------------------------------

from shell_simulation.steering import steering_command  # noqa: E402


def command(deg, lookahead=LOOKAHEAD):
    return steering_command(math.radians(deg), lookahead, WHEEL_BASE, MAX_STEER)


def test_command_is_negated_relative_to_the_pure_pursuit_law():
    """/steering_command is -1.0 full LEFT, +1.0 full RIGHT.

    alpha follows the maths convention, positive to the left, so the two are
    opposed. Publishing the law's output unnegated steers away from the path
    and the error grows until the car leaves the road.
    """
    for deg in (-120, -60, -30, -5, 0, 5, 30, 60, 120):
        assert command(deg) == pytest.approx(-steer(deg))


def test_target_on_the_left_commands_left():
    assert command(30) < 0     # negative is left


def test_target_on_the_right_commands_right():
    assert command(-30) > 0    # positive is right


def test_straight_ahead_commands_neutral():
    assert command(0.0) == pytest.approx(0.0)


def test_command_stays_in_range():
    for deg in range(-360, 361, 5):
        assert -1.0 <= command(deg) <= 1.0
