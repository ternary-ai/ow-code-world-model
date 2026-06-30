"""
tests/test_symmetry.py — Tests for cwm/symmetry.py (Module 6).

Coverage:
  - test_apply_transform_identity_is_noop
  - test_apply_transform_is_involution
  - test_canonical_transform_produces_quadrant_invariant
  - test_augment_batch_length
  - test_transformed_state_is_physically_valid
"""

from __future__ import annotations

import math
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cwm.symmetry import (
    canonical_transform_id,
    apply_transform,
    augment_batch,
)
from cwm.state import State, CENTER
from cwm.interpreter import cwm_apply_joint_action


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_state(
    player0_home_pos: tuple = (20.0, 20.0),
    include_fleet: bool = False,
) -> State:
    """Build a two-planet state with player 0's home at player0_home_pos."""
    planets = [
        [0, 0, player0_home_pos[0], player0_home_pos[1], 1.5, 20, 2],  # player 0 home
        [1, 1, 80.0, 80.0, 1.5, 15, 2],                                  # player 1 home
    ]
    fleets = []
    if include_fleet:
        # Fleet from planet 0 heading toward planet 1
        angle = math.atan2(80.0 - player0_home_pos[1], 80.0 - player0_home_pos[0])
        fleets = [[0, 0, player0_home_pos[0] + 2.0, player0_home_pos[1], angle, 0, 5]]

    return State(
        planets=[list(p) for p in planets],
        fleets=[list(f) for f in fleets],
        initial_planets=[list(p) for p in planets],
        comets=[],
        comet_planet_ids=[],
        step=5,
        next_fleet_id=len(fleets),
        angular_velocity=0.025,
        num_players=2,
        episode_steps=500,
        ship_speed=6.0,
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestApplyTransform:

    def test_apply_transform_identity_is_noop(self):
        """Transform 0 (identity) returns an unchanged state."""
        state = _make_state((30.0, 40.0), include_fleet=True)
        result = apply_transform(state, 0)

        assert result.planets[0][2] == state.planets[0][2]
        assert result.planets[0][3] == state.planets[0][3]
        assert result.planets[1][2] == state.planets[1][2]
        assert result.planets[1][3] == state.planets[1][3]
        if result.fleets:
            assert abs(result.fleets[0][4] - state.fleets[0][4]) < 1e-9

    def test_apply_transform_mirror_x_flips_x(self):
        """Transform 1 (mirror-x): x → 100-x, y unchanged."""
        state = _make_state((30.0, 40.0))
        result = apply_transform(state, 1)

        assert abs(result.planets[0][2] - (100.0 - 30.0)) < 1e-9
        assert abs(result.planets[0][3] - 40.0) < 1e-9

    def test_apply_transform_mirror_y_flips_y(self):
        """Transform 2 (mirror-y): y → 100-y, x unchanged."""
        state = _make_state((30.0, 40.0))
        result = apply_transform(state, 2)

        assert abs(result.planets[0][2] - 30.0) < 1e-9
        assert abs(result.planets[0][3] - (100.0 - 40.0)) < 1e-9

    def test_apply_transform_mirror_both(self):
        """Transform 3 (mirror-both): (x,y) → (100-x, 100-y)."""
        state = _make_state((30.0, 40.0))
        result = apply_transform(state, 3)

        assert abs(result.planets[0][2] - (100.0 - 30.0)) < 1e-9
        assert abs(result.planets[0][3] - (100.0 - 40.0)) < 1e-9

    def test_apply_transform_is_involution(self):
        """Applying the same non-identity transform twice returns the original state."""
        state = _make_state((30.0, 40.0), include_fleet=True)

        for tid in (1, 2, 3):
            twice = apply_transform(apply_transform(state, tid), tid)

            for i, (orig, back) in enumerate(zip(state.planets, twice.planets)):
                assert abs(orig[2] - back[2]) < 1e-9, f"transform {tid} planet {i} x"
                assert abs(orig[3] - back[3]) < 1e-9, f"transform {tid} planet {i} y"

            for i, (fo, fb) in enumerate(zip(state.fleets, twice.fleets)):
                assert abs(fo[4] - fb[4]) < 1e-9, f"transform {tid} fleet {i} angle"

    def test_fleet_angle_transformed_consistently(self):
        """Fleet angles are updated to match the coordinate transform."""
        state = _make_state((20.0, 20.0), include_fleet=True)
        orig_angle = state.fleets[0][4]

        # Mirror-x: x → 100-x. Fleet moving right (cos>0) now moves left.
        result = apply_transform(state, 1)
        new_angle = result.fleets[0][4]
        # In mirror-x, cos(angle) → -cos(angle), sin(angle) unchanged
        # So new_angle = π - orig_angle (mod 2π)
        expected = (math.pi - orig_angle) % (2 * math.pi)
        diff = abs(math.atan2(math.sin(new_angle - expected),
                              math.cos(new_angle - expected)))
        assert diff < 1e-9, f"Mirror-x angle: got {new_angle:.4f}, expected {expected:.4f}"


class TestCanonicalTransform:

    def test_canonical_transform_produces_quadrant_invariant(self):
        """After applying canonical_transform_id, the home planet satisfies x<=50, y<=50."""
        # Test all 4 starting quadrants
        home_positions = [
            (20.0, 20.0),   # already canonical (Q3 in screen coords, x<=50, y<=50)
            (80.0, 20.0),   # Q1: x>50, y<=50 → mirror-x
            (20.0, 80.0),   # Q2: x<=50, y>50 → mirror-y
            (80.0, 80.0),   # Q4: x>50, y>50 → mirror-both
        ]
        for pos in home_positions:
            state = _make_state(pos)
            tid = canonical_transform_id(state)
            transformed = apply_transform(state, tid)
            home = transformed.planets[0]
            assert home[2] <= 50.0 + 1e-9 and home[3] <= 50.0 + 1e-9, (
                f"Home {pos} → transform {tid} → ({home[2]:.2f}, {home[3]:.2f})"
                " not in canonical quadrant x<=50, y<=50"
            )

    def test_already_canonical_uses_identity(self):
        """A state already in the canonical quadrant gets transform_id=0."""
        state = _make_state((20.0, 20.0))   # x<=50, y<=50
        assert canonical_transform_id(state) == 0


class TestAugmentBatch:

    def test_augment_batch_length(self):
        """len(augment_batch(states)) == 4 * len(states)."""
        states = [_make_state((20.0, 20.0)), _make_state((80.0, 20.0))]
        augmented = augment_batch(states)
        assert len(augmented) == 4 * len(states)

    def test_augment_batch_empty(self):
        """augment_batch([]) returns []."""
        assert augment_batch([]) == []

    def test_transformed_state_is_physically_valid(self):
        """Transformed state has sun at (50,50) and planets in [0,100]."""
        state = _make_state((30.0, 40.0))
        for tid in range(4):
            transformed = apply_transform(state, tid)
            for p in transformed.planets:
                assert 0.0 <= p[2] <= 100.0, f"transform {tid}: x={p[2]} out of bounds"
                assert 0.0 <= p[3] <= 100.0, f"transform {tid}: y={p[3]} out of bounds"
            # Sun is at CENTER=(50,50) — not stored but transforms preserve it
            # Verify by checking that simulating a step doesn't error
            action = [[], []]
            next_s = cwm_apply_joint_action(transformed, action)
            assert next_s.step == transformed.step + 1


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
