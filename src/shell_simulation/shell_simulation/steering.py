#!/usr/bin/env python3

"""Steering geometry for the controller.

Kept free of ROS and CARLA imports so it can be exercised directly, which
matters: the faults this module fixes were invisible in every static check
and only showed up as a car scrubbing its tyres against the kerb.
"""

import math


def pure_pursuit_steer(alpha: float, lookahead: float, wheel_base: float,
                       max_steer_angle: float) -> float:
    """Normalised steering command in [-1, 1] for a heading error of `alpha`.

    `alpha` is the angle from the vehicle's heading to the target point, in
    radians, positive to the left. The pure-pursuit law yields a road-wheel
    angle, which is divided by the full-lock angle to become a command; using
    the raw angle saturates the command for any target more than about 21
    degrees off-axis.

    A target behind the vehicle is handled separately, because sin() collapses
    towards zero near +-pi and the law would ask for almost no steering exactly
    when the tightest available turn is needed.

    `lookahead` should exceed `wheel_base`, or the geometry demands turns the
    vehicle cannot execute.
    """
    alpha = math.atan2(math.sin(alpha), math.cos(alpha))  # wrap to [-pi, pi]

    if abs(alpha) <= math.pi / 2.0:
        steer_angle = math.atan2(2.0 * wheel_base * math.sin(alpha), lookahead)
        return max(-1.0, min(1.0, steer_angle / max_steer_angle))

    # Behind the vehicle: ramp from whatever the law gives at 90 degrees up to
    # full lock at 180, so the command stays continuous across the boundary. A
    # step there would fight the controller's steering rate limit.
    at_ninety = min(1.0, math.atan2(2.0 * wheel_base, lookahead) / max_steer_angle)
    beyond = (abs(alpha) - math.pi / 2.0) / (math.pi / 2.0)  # 0 at 90, 1 at 180
    magnitude = at_ninety + (1.0 - at_ninety) * beyond
    return math.copysign(min(1.0, magnitude), alpha)
