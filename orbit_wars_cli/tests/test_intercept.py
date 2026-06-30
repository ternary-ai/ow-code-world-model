"""
tests/test_intercept.py — Tests for cwm/intercept.py (Module 2).

Coverage:
  - test_static_target_matches_direct_angle: static planet → angle == atan2(dy, dx)
  - test_orbiting_target_leads_correctly: angle leads planet to arrival position
  - test_larger_fleet_changes_solved_angle: speed-dependent angle for orbiting target
  - test_returns_none_on_non_convergence: max_iters=0 → None
"""

from __future__ import annotations

import math
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cwm.intercept import solve_intercept
from cwm.state import State, CENTER, SUN_RADIUS, ROTATION_RADIUS_LIMIT
from cwm.geometry import fleet_speed


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_two_planet_state(
    src_x: float, src_y: float,
    tgt_x: float, tgt_y: float,
    tgt_owner: int = -1,
    angular_velocity: float = 0.025,
    step: int = 0,
    ship_speed: float = 6.0,
) -> State:
    """Build a minimal 2-planet State.

    Planet 0 (source): owned by player 0, at (src_x, src_y), no ships.
    Planet 1 (target): owned by tgt_owner, at (tgt_x, tgt_y).
    """
    planets = [
        [0, 0, src_x, src_y, 1.0, 20, 1],
        [1, tgt_owner, tgt_x, tgt_y, 1.0, 5, 1],
    ]
    initial_planets = [list(p) for p in planets]
    return State(
        planets=planets,
        fleets=[],
        initial_planets=initial_planets,
        comets=[],
        comet_planet_ids=[],
        step=step,
        next_fleet_id=0,
        angular_velocity=angular_velocity,
        num_players=2,
        episode_steps=500,
        ship_speed=ship_speed,
    )


def _static_state(ship_speed: float = 6.0) -> State:
    """Source at (15, 15), static target at (85, 15) — same y, no orbital motion.

    The target's orbital_radius = sqrt((85-50)^2 + (15-50)^2) = sqrt(35^2+35^2)
    ≈ 49.5 + radius 1.0 = 50.5 > ROTATION_RADIUS_LIMIT(50) → static.
    """
    return _make_two_planet_state(15.0, 15.0, 85.0, 15.0, ship_speed=ship_speed)


def _orbiting_state(angular_velocity: float = 0.025, ship_speed: float = 6.0) -> State:
    """Source at (15, 15) (static corner), orbiting target near center.

    Target at (50+20, 50) = (70, 50):
    orbital_radius = 20, radius = 1.0 → 21 < 50 → ORBITING.
    """
    return _make_two_planet_state(15.0, 15.0, 70.0, 50.0,
                                  angular_velocity=angular_velocity,
                                  ship_speed=ship_speed)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestSolveIntercept:

    def test_static_target_matches_direct_angle(self):
        """For a static (outer) target, solved angle == atan2(dy, dx)."""
        state = _static_state()
        source_pos = (15.0, 15.0)
        target_id = 1

        angle = solve_intercept(source_pos, target_id, fleet_size=10,
                                depart_turn=0, state=state)
        assert angle is not None

        # Direct angle from source to (static) target
        tgt = state.planets[1]
        expected = math.atan2(tgt[3] - source_pos[1], tgt[2] - source_pos[0])
        expected = expected % (2 * math.pi)

        # Allow a small tolerance since we iterate
        diff = abs(math.atan2(math.sin(angle - expected), math.cos(angle - expected)))
        assert diff < 1e-3, f"angle={angle:.6f} expected={expected:.6f} diff={diff:.6f}"

    def test_orbiting_target_leads_correctly(self):
        """Solved angle for orbiting target: the aim point is the planet's future position.

        We verify by computing the target's predicted position at the estimated
        arrival turn and checking it matches the aim direction within tolerance.
        The aim is derived from the game's own orbital parameters (not the function
        under test), so this is an independent ground-truth check.
        """
        av = 0.025
        state = _orbiting_state(angular_velocity=av)
        source_pos = (15.0, 15.0)
        fleet_sz = 10
        depart_turn = 0

        angle = solve_intercept(source_pos, target_planet_id=1,
                                fleet_size=fleet_sz, depart_turn=depart_turn,
                                state=state)
        assert angle is not None

        # Estimate arrival: travel distance / speed
        speed = fleet_speed(fleet_sz, state.ship_speed)
        tgt = state.planets[1]
        aim_x = source_pos[0] + math.cos(angle) * 1.0  # direction unit vector
        aim_y = source_pos[1] + math.sin(angle) * 1.0

        # Target's initial position
        init_tgt = state.initial_planets[1]
        dx = init_tgt[2] - CENTER
        dy = init_tgt[3] - CENTER
        r = math.sqrt(dx * dx + dy * dy)
        init_angle = math.atan2(dy, dx)

        # Find the arrival turn: iterate once to get a good estimate
        # (mirrors the fixed-point logic in solve_intercept)
        cur_tgt_pos = (tgt[2], tgt[3])
        for _ in range(20):
            dist = math.hypot(cur_tgt_pos[0] - source_pos[0],
                              cur_tgt_pos[1] - source_pos[1])
            eta = dist / speed
            future_angle = init_angle + av * (depart_turn + eta)
            cur_tgt_pos = (CENTER + r * math.cos(future_angle),
                           CENTER + r * math.sin(future_angle))

        # The angle returned by solve_intercept should point toward cur_tgt_pos
        expected_angle = math.atan2(
            cur_tgt_pos[1] - source_pos[1],
            cur_tgt_pos[0] - source_pos[0],
        ) % (2 * math.pi)

        diff = abs(math.atan2(math.sin(angle - expected_angle),
                              math.cos(angle - expected_angle)))
        assert diff < 1e-3, (
            f"angle={angle:.6f} expected={expected_angle:.6f} diff={diff:.6f}"
        )

    def test_larger_fleet_changes_solved_angle(self):
        """Different fleet_size values produce different angles for an orbiting target.

        Larger fleets travel faster → shorter travel time → target is at a
        different (closer) future position → different intercept angle.
        """
        state = _orbiting_state()
        source_pos = (15.0, 15.0)

        angle_small = solve_intercept(source_pos, target_planet_id=1,
                                      fleet_size=1, depart_turn=0, state=state)
        angle_large = solve_intercept(source_pos, target_planet_id=1,
                                      fleet_size=500, depart_turn=0, state=state)

        assert angle_small is not None
        assert angle_large is not None
        # Angles should differ because speeds are very different
        diff = abs(math.atan2(math.sin(angle_small - angle_large),
                              math.cos(angle_small - angle_large)))
        assert diff > 1e-6, (
            f"Expected angles to differ; small={angle_small:.6f} large={angle_large:.6f}"
        )

    def test_returns_none_on_non_convergence(self):
        """max_iters=0 forces the solver to return None immediately."""
        state = _orbiting_state()
        result = solve_intercept(
            source_pos=(15.0, 15.0),
            target_planet_id=1,
            fleet_size=10,
            depart_turn=0,
            state=state,
            max_iters=0,
        )
        assert result is None

    def test_unknown_target_returns_none(self):
        """Nonexistent target planet id returns None."""
        state = _static_state()
        result = solve_intercept((15.0, 15.0), target_planet_id=999,
                                 fleet_size=5, depart_turn=0, state=state)
        assert result is None

    def test_angle_in_range_zero_to_two_pi(self):
        """Returned angle is always in [0, 2π)."""
        state = _static_state()
        angle = solve_intercept((85.0, 15.0), target_planet_id=0,
                                fleet_size=5, depart_turn=0, state=state)
        assert angle is not None
        assert 0.0 <= angle < 2 * math.pi


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
