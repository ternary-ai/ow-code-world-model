"""
tests/test_masking.py — Tests for cwm/masking.py (Module 4: sun-occlusion legality).

Coverage:
  - sun_blocks_path: direct/blocked/tangent paths
  - legal_pair_mask: symmetry, diagonal, blocked pairs
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cwm.masking import sun_blocks_path, legal_pair_mask
from cwm.state import State, CENTER, SUN_RADIUS

SUN_POS = (CENTER, CENTER)  # (50.0, 50.0)


def _make_state(planet_positions: list[tuple[float, float]]) -> State:
    """Build a minimal State with planets at the given (x, y) positions."""
    planets = [
        [i, -1, x, y, 1.0, 0, 1]
        for i, (x, y) in enumerate(planet_positions)
    ]
    return State(
        planets=planets,
        fleets=[],
        initial_planets=[list(p) for p in planets],
        comets=[],
        comet_planet_ids=[],
        step=0,
        next_fleet_id=0,
        angular_velocity=0.025,
        num_players=2,
    )


# ── sun_blocks_path ────────────────────────────────────────────────────────────

class TestSunBlocksPath:

    def test_direct_path_not_blocked(self):
        """Two planets on the same side of the sun, clearly unobstructed."""
        # Both at y=80 (well above sun at y=50), so no path through sun
        source = (30.0, 80.0)
        target = (70.0, 80.0)
        assert not sun_blocks_path(source, target, SUN_POS, SUN_RADIUS)

    def test_diametrically_opposite_blocked(self):
        """Two planets on opposite sides, straight path passes through center."""
        source = (50.0, 10.0)  # below sun
        target = (50.0, 90.0)  # above sun
        # Segment (50,10)→(50,90) passes through center (50,50) — dist=0 < 10
        assert sun_blocks_path(source, target, SUN_POS, SUN_RADIUS)

    def test_tangent_path_not_blocked(self):
        """Path that passes exactly at sun_radius from center returns False (strict <)."""
        # Horizontal segment at y = 50 - SUN_RADIUS = 40.0
        # Closest point to center (50,50) is (50,40), dist = 10.0 exactly
        # Since we use strict < sun_radius, this should NOT be blocked
        source = (10.0, 40.0)
        target = (90.0, 40.0)
        assert not sun_blocks_path(source, target, SUN_POS, SUN_RADIUS)

    def test_path_just_inside_radius_blocked(self):
        """Path passing just inside sun_radius is blocked."""
        # Horizontal segment at y = 40.01 → closest dist = 9.99 < 10 → blocked
        source = (10.0, 40.01)
        target = (90.0, 40.01)
        assert sun_blocks_path(source, target, SUN_POS, SUN_RADIUS)

    def test_path_through_center_blocked(self):
        """Segment going straight through the sun center is always blocked."""
        assert sun_blocks_path((0.0, 50.0), (100.0, 50.0), SUN_POS, SUN_RADIUS)

    def test_both_endpoints_outside_sun_far_away(self):
        """Segment with endpoints far from sun and path not crossing."""
        # Short segment in a corner, nowhere near the sun
        assert not sun_blocks_path((5.0, 5.0), (15.0, 5.0), SUN_POS, SUN_RADIUS)

    def test_different_sun_position_and_radius(self):
        """Works with arbitrary sun position and radius."""
        custom_sun = (20.0, 20.0)
        radius = 5.0
        # Segment through custom sun center: blocked
        assert sun_blocks_path((20.0, 10.0), (20.0, 30.0), custom_sun, radius)
        # Segment far from custom sun: not blocked
        assert not sun_blocks_path((60.0, 60.0), (80.0, 60.0), custom_sun, radius)


# ── legal_pair_mask ────────────────────────────────────────────────────────────

class TestLegalPairMask:

    def test_diagonal_is_false(self):
        """A planet cannot target itself: mask[i][i] is always False."""
        # Two planets on the same side
        state = _make_state([(20.0, 80.0), (80.0, 80.0)])
        mask = legal_pair_mask(state)
        assert mask.shape == (2, 2)
        assert not mask[0, 0]
        assert not mask[1, 1]

    def test_unblocked_pair(self):
        """Two planets on the same side of the sun: mask[i][j] and mask[j][i] are True."""
        state = _make_state([(20.0, 80.0), (80.0, 80.0)])
        mask = legal_pair_mask(state)
        assert mask[0, 1]
        assert mask[1, 0]

    def test_blocked_pair(self):
        """Two planets on opposite sides through the center: both directions False."""
        # Planet 0 below sun, planet 1 above sun — straight path goes through center
        state = _make_state([(50.0, 10.0), (50.0, 90.0)])
        mask = legal_pair_mask(state)
        assert not mask[0, 1]
        assert not mask[1, 0]

    def test_mask_is_symmetric(self):
        """legal_pair_mask[i][j] == legal_pair_mask[j][i] for all i, j."""
        positions = [
            (20.0, 80.0),  # upper-left quadrant
            (80.0, 80.0),  # upper-right
            (50.0, 10.0),  # below sun (will be blocked to/from above)
            (50.0, 90.0),  # above sun
        ]
        state = _make_state(positions)
        mask = legal_pair_mask(state)
        n = len(positions)
        for i in range(n):
            for j in range(n):
                assert mask[i, j] == mask[j, i], (
                    f"Asymmetry at ({i},{j}): {mask[i,j]} vs {mask[j,i]}"
                )

    def test_shape_matches_planet_count(self):
        """Mask shape is (n_planets, n_planets)."""
        state = _make_state([(10.0, 10.0), (90.0, 10.0), (10.0, 90.0)])
        mask = legal_pair_mask(state)
        assert mask.shape == (3, 3)

    def test_returns_numpy_bool_array(self):
        """Return type is np.ndarray with dtype bool."""
        state = _make_state([(20.0, 20.0), (80.0, 20.0)])
        mask = legal_pair_mask(state)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
