"""
tests/test_oob_collision_order.py — Issue #1017 edge case.

A fast fleet whose swept path passes through a planet but whose endpoint lands
out of bounds must be CAPTURED by the planet (added to combat), not destroyed by
the out-of-bounds check. The CWM mirrors orbit_wars_original.py, which runs the
planet-collision check BEFORE the OOB check (see interpreter() fleet movement).

Reference: https://github.com/Kaggle/kaggle-environments/issues/1017
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cwm.state import State, BOARD_SIZE
from cwm.interpreter import cwm_apply_joint_action


def _edge_state() -> State:
    # Defender planet near the top-right corner, on the y = x diagonal.
    target = [0, 1, 98.6, 98.6, 1.0, 5, 0]      # owner 1, garrison 5, prod 0
    # Player-0 home planet, far away and static (no rotation, no interference).
    home = [1, 0, 10.0, 10.0, 1.0, 5, 0]

    planets = [target, home]

    # A 1000-ship fleet just short of the target, heading at 45 degrees.
    # speed == ship_speed (6.0) at 1000 ships, so:
    #   start (96, 96) -> end (96 + 6*cos45, 96 + 6*sin45) ~= (100.24, 100.24)
    # The endpoint is OOB (> 100) but the segment passes through the target
    # (which lies on the y = x line, distance 0 < radius 1).
    fleet = [10, 0, 96.0, 96.0, math.pi / 4.0, 1, 1000]

    return State(
        planets=[list(p) for p in planets],
        fleets=[list(fleet)],
        initial_planets=[list(p) for p in planets],
        comets=[],
        comet_planet_ids=[],
        step=10,
        next_fleet_id=11,
        angular_velocity=0.03,
        num_players=2,
        episode_steps=500,
        ship_speed=6.0,
        comet_speed=4.0,
    )


def test_fleet_through_planet_into_oob_is_captured_not_destroyed():
    state = _edge_state()

    # Sanity: the fleet endpoint really is out of bounds (so OOB-first ordering
    # would have destroyed it).
    speed = 6.0
    end_x = 96.0 + math.cos(math.pi / 4.0) * speed
    end_y = 96.0 + math.sin(math.pi / 4.0) * speed
    assert end_x > BOARD_SIZE and end_y > BOARD_SIZE

    nxt = cwm_apply_joint_action(state, [[], []])

    # The attacking fleet must have been consumed (captured into combat), not
    # left floating.
    assert all(f[6] != 1000 for f in nxt.fleets)

    # Combat resolved: 1000 attackers vs garrison 5 -> planet 0 conquered by
    # player 0. If OOB had run first the fleet would vanish and the planet would
    # still belong to player 1 with garrison 5.
    target = next(p for p in nxt.planets if p[0] == 0)
    assert target[1] == 0, "planet should be conquered by player 0"
    assert target[5] == 995, "garrison should be 1000 - 5 = 995 after conquest"
