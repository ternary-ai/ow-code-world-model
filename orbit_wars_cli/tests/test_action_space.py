"""
tests/test_action_space.py — Tests for cwm/action_space.py (Module 3).

Coverage:
  - test_candidate_count_matches_targets_times_tiers_minus_blocked
  - test_excludes_insufficient_garrison
  - test_each_candidate_angle_matches_solve_intercept
  - test_excludes_sun_blocked_targets
"""

from __future__ import annotations

import math
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cwm.action_space import CandidateAction, generate_candidates
from cwm.intercept import solve_intercept
from cwm.masking import sun_blocks_path
from cwm.state import State, CENTER, SUN_RADIUS


# ── State helpers ──────────────────────────────────────────────────────────────

def _make_state(planets: list, step: int = 0, ship_speed: float = 6.0) -> State:
    initial = [list(p) for p in planets]
    return State(
        planets=[list(p) for p in planets],
        fleets=[],
        initial_planets=initial,
        comets=[],
        comet_planet_ids=[],
        step=step,
        next_fleet_id=0,
        angular_velocity=0.025,
        num_players=2,
        episode_steps=500,
        ship_speed=ship_speed,
    )


def _planet(pid, owner, x, y, ships, production=2, radius=1.5):
    return [pid, owner, x, y, radius, ships, production]


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestGenerateCandidates:

    def test_candidate_count_matches_targets_times_tiers_minus_blocked(self):
        """N reachable targets × T tiers == candidate count (minus sun-blocked pairs)."""
        # Place source and 3 targets all on the same side (top-right quadrant),
        # none blocked by the sun.
        state = _make_state([
            _planet(0, 0, 20.0, 20.0, ships=30),   # source: player 0
            _planet(1, -1, 80.0, 20.0, ships=5),   # target 1
            _planet(2, -1, 80.0, 80.0, ships=5),   # target 2 (may be blocked)
            _planet(3, -1, 20.0, 80.0, ships=5),   # target 3
        ])

        tiers = [5, 10, 20]
        candidates = generate_candidates(state, source_planet_id=0,
                                         ship_count_tiers=tiers)

        # Count how many (target, tier) pairs are blocked
        src_pos = (20.0, 20.0)
        blocked = 0
        for tid in [1, 2, 3]:
            tgt = next(p for p in state.planets if p[0] == tid)
            if sun_blocks_path(src_pos, (tgt[2], tgt[3]), (CENTER, CENTER), SUN_RADIUS):
                blocked += len(tiers)

        expected = 3 * len(tiers) - blocked
        assert len(candidates) == expected, (
            f"Expected {expected} candidates, got {len(candidates)}"
        )

    def test_excludes_insufficient_garrison(self):
        """Tiers that exceed the source garrison are excluded."""
        state = _make_state([
            _planet(0, 0, 20.0, 20.0, ships=8),    # garrison = 8
            _planet(1, -1, 80.0, 20.0, ships=3),
        ])

        tiers = [3, 8, 10, 20]   # 10 and 20 exceed garrison 8
        candidates = generate_candidates(state, source_planet_id=0,
                                         ship_count_tiers=tiers)

        # Only tiers [3, 8] are valid; both target planet 1
        ship_counts = {c.ships for c in candidates}
        assert 10 not in ship_counts
        assert 20 not in ship_counts

    def test_each_candidate_angle_matches_solve_intercept(self):
        """Every returned candidate's angle matches a direct call to solve_intercept."""
        state = _make_state([
            _planet(0, 0, 20.0, 20.0, ships=20),
            _planet(1, -1, 80.0, 20.0, ships=3),
            _planet(2, -1, 20.0, 80.0, ships=3),
        ])

        tiers = [5, 15]
        candidates = generate_candidates(state, source_planet_id=0,
                                         ship_count_tiers=tiers)
        assert len(candidates) > 0

        for c in candidates:
            expected_angle = solve_intercept(
                source_pos=(state.planets[0][2], state.planets[0][3]),
                target_planet_id=c.target_planet_id,
                fleet_size=c.ships,
                depart_turn=state.step,
                state=state,
            )
            assert expected_angle is not None, (
                f"solve_intercept returned None for candidate {c}"
            )
            diff = abs(math.atan2(math.sin(c.angle - expected_angle),
                                  math.cos(c.angle - expected_angle)))
            assert diff < 1e-6, (
                f"Angle mismatch for {c}: got {c.angle:.6f}, expected {expected_angle:.6f}"
            )

    def test_excludes_sun_blocked_targets(self):
        """A target whose path from source crosses the sun is not included."""
        # Source below sun, target above sun — path goes through center
        state = _make_state([
            _planet(0, 0, 50.0, 10.0, ships=20),   # below sun
            _planet(1, -1, 50.0, 90.0, ships=3),   # above sun — path through center
            _planet(2, -1, 80.0, 20.0, ships=3),   # safe target
        ])

        tiers = [5, 10]
        candidates = generate_candidates(state, source_planet_id=0,
                                         ship_count_tiers=tiers)

        target_ids = {c.target_planet_id for c in candidates}
        assert 1 not in target_ids, "Sun-blocked target (id=1) should be excluded"
        assert 2 in target_ids, "Reachable target (id=2) should be included"

    def test_source_planet_excluded_from_targets(self):
        """The source planet is never a candidate target."""
        state = _make_state([
            _planet(0, 0, 20.0, 20.0, ships=15),
            _planet(1, -1, 80.0, 20.0, ships=3),
        ])
        candidates = generate_candidates(state, source_planet_id=0,
                                         ship_count_tiers=[5])
        for c in candidates:
            assert c.source_planet_id != c.target_planet_id

    def test_returns_dataclass_instances(self):
        """Each candidate is a CandidateAction instance."""
        state = _make_state([
            _planet(0, 0, 20.0, 20.0, ships=10),
            _planet(1, -1, 80.0, 20.0, ships=3),
        ])
        candidates = generate_candidates(state, source_planet_id=0,
                                         ship_count_tiers=[5])
        for c in candidates:
            assert isinstance(c, CandidateAction)
            assert isinstance(c.angle, float)
            assert isinstance(c.predicted_arrival_turn, int)


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
