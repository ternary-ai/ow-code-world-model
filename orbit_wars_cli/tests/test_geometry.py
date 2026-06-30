"""
tests/test_geometry.py — Tests for cwm/geometry.py

Coverage:
  - fleet_speed: README anchors (1 ship, ~500, ~1000) + edge cases
  - segment_circle_collision: hit, miss, tangent, segment entirely inside
  - is_orbiting: just inside / just outside / exactly on threshold
  - rotate_point: full-circle recovery, zero radius, negative angle
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cwm.geometry import (
    fleet_speed,
    segment_circle_collision,
    swept_pair_hit,
    is_orbiting,
    rotate_point,
)

CENTER = (50.0, 50.0)
ROTATION_RADIUS_LIMIT = 50.0


# ── fleet_speed ────────────────────────────────────────────────────────────────

class TestFleetSpeed:

    def test_one_ship_is_min_speed(self):
        """1 ship → speed 1.0 (log(1)=0, so formula gives 1.0 exactly)."""
        assert fleet_speed(1) == 1.0

    def test_one_ship_explicit_max_speed(self):
        assert fleet_speed(1, max_speed=6.0) == 1.0

    def test_zero_ships_clamped_to_one(self):
        """0 or negative ships are clamped to 1 → speed == 1.0."""
        assert fleet_speed(0) == 1.0
        assert fleet_speed(-5) == 1.0

    def test_thousand_ships_is_max_speed(self):
        """1000 ships → log(1000)/log(1000) = 1 → speed = max_speed."""
        assert abs(fleet_speed(1000) - 6.0) < 1e-10

    def test_thousand_ships_custom_max(self):
        """Works with non-default max_speed."""
        result = fleet_speed(1000, max_speed=8.0)
        assert abs(result - 8.0) < 1e-10

    def test_five_hundred_ships_approx_five(self):
        """~500 ships → speed in [5.0, 5.6] per README 'approximately 5'."""
        # Exact: 1 + 5*(log500/log1000)^1.5 ≈ 5.27 with max_speed=6
        speed = fleet_speed(500)
        assert 5.0 <= speed <= 5.6, f"Expected ~5.0-5.6 for 500 ships, got {speed:.4f}"

    def test_speed_monotonically_increases_with_ships(self):
        """Larger fleets are never slower than smaller fleets."""
        prev = fleet_speed(1)
        for n in [2, 5, 10, 50, 100, 500, 999, 1000]:
            cur = fleet_speed(n)
            assert cur >= prev, f"Non-monotone at n={n}: {cur} < {prev}"
            prev = cur

    def test_speed_never_exceeds_max(self):
        """Speed is always clamped to max_speed."""
        for n in [1000, 2000, 9999]:
            assert fleet_speed(n) <= 6.0

    def test_large_fleet_capped_at_max_speed(self):
        """Very large fleets are capped, not extrapolated past max."""
        assert fleet_speed(10_000) == 6.0


# ── segment_circle_collision ───────────────────────────────────────────────────

class TestSegmentCircleCollision:

    # The sun: center=(50,50), radius=10

    def test_clear_hit_passes_through_circle(self):
        """Segment going straight through the center clearly hits."""
        # Horizontal segment through center (y=50, x: 0→100)
        assert segment_circle_collision((0.0, 50.0), (100.0, 50.0), CENTER, 10.0)

    def test_clear_miss_far_from_circle(self):
        """Segment far above the circle doesn't collide."""
        # Horizontal segment at y=0 (center y=50, radius=10 → min dist=40)
        assert not segment_circle_collision((0.0, 0.0), (100.0, 0.0), CENTER, 10.0)

    def test_segment_entirely_inside_circle(self):
        """Segment with both endpoints inside the circle → True (dist = 0 < r)."""
        # Both endpoints at (50, 50) and (50, 51) inside sun
        assert segment_circle_collision((50.0, 50.0), (50.0, 51.0), CENTER, 10.0)

    def test_endpoint_on_circle_boundary(self):
        """Endpoint exactly on boundary: dist == radius → False (strict <)."""
        # Point (50, 40) is exactly 10 units from center (50,50)
        assert not segment_circle_collision((50.0, 40.0), (0.0, 40.0), CENTER, 10.0)

    def test_tangent_graze(self):
        """Segment grazing just inside radius → True."""
        # Horizontal line at y=40.0 from (0,40) to (100,40).
        # Closest point to center(50,50) = (50,40), dist=10.0 → not strictly less.
        # Move to y=39.9: dist=10.1 → miss
        assert not segment_circle_collision((0.0, 39.9), (100.0, 39.9), CENTER, 10.0)
        # y=40.01: dist=9.99 < 10 → hit
        assert segment_circle_collision((0.0, 40.01), (100.0, 40.01), CENTER, 10.0)

    def test_segment_misses_due_to_extent(self):
        """Segment endpoint is near circle but the segment doesn't reach it."""
        # Segment from (90,90) to (95,90) — far from center (50,50)
        assert not segment_circle_collision((90.0, 90.0), (95.0, 90.0), CENTER, 10.0)

    def test_vertical_hit(self):
        """Vertical segment through sun center."""
        assert segment_circle_collision((50.0, 0.0), (50.0, 100.0), CENTER, 10.0)

    def test_sun_radius_on_fleet_path(self):
        """Simulate a fleet flying from (10,50) to (45,50): path passes within 5 of sun."""
        # Gets within 5 of sun at closest approach x=45 → still outside radius 10? No.
        # dist from center(50,50) to segment(10,50)→(45,50): closest point is (45,50),
        # dist = sqrt((50-45)^2 + 0) = 5 < 10 → HIT
        assert segment_circle_collision((10.0, 50.0), (45.0, 50.0), CENTER, 10.0)

    def test_fleet_stops_before_sun(self):
        """Fleet stopping well before the sun doesn't trigger sun collision."""
        # Segment (10,50) → (39,50): closest pt=(39,50), dist=11 > 10 → miss
        assert not segment_circle_collision((10.0, 50.0), (39.0, 50.0), CENTER, 10.0)

    def test_small_circle(self):
        """Works with small radii (planet collision scenario)."""
        # Planet at (30,30) with radius 2, fleet passes at (28,30)→(32,30): hit
        assert segment_circle_collision((28.0, 30.0), (32.0, 30.0), (30.0, 30.0), 2.0)

    def test_zero_length_segment(self):
        """Zero-length segment (p0 == p1): distance = dist(p0, center)."""
        # Point at center: inside any positive radius
        assert segment_circle_collision(CENTER, CENTER, CENTER, 5.0)
        # Point far away: miss
        assert not segment_circle_collision((0.0, 0.0), (0.0, 0.0), CENTER, 5.0)


# ── is_orbiting ────────────────────────────────────────────────────────────────

class TestIsOrbiting:
    """Condition: orbital_radius + planet_radius < 50.0"""

    def _make_planet(self, cx, cy, radius):
        """Build a minimal planet list at position (cx, cy) with given radius."""
        # planet = [id, owner, x, y, radius, ships, production]
        return [0, -1, cx, cy, radius, 10, 1]

    def test_clearly_orbiting(self):
        """Planet well inside the orbit threshold."""
        # orbital_radius = sqrt((40-50)^2+(50-50)^2) = 10, radius=2 → 12 < 50
        p = self._make_planet(40.0, 50.0, 2.0)
        assert is_orbiting(p, CENTER, ROTATION_RADIUS_LIMIT)

    def test_clearly_static(self):
        """Planet far from center: orbital + radius >> 50."""
        # orbital_radius = sqrt((90-50)^2+(50-50)^2)=40, radius=2 → 42 < 50 still orbiting
        # Let's use orbital=46, radius=5 → 51 > 50 → static
        p = self._make_planet(96.0, 50.0, 5.0)  # orbital=46, r=5 → 51 > 50
        assert not is_orbiting(p, CENTER, ROTATION_RADIUS_LIMIT)

    def test_just_inside_threshold(self):
        """orbital_radius + radius = 49.99 → orbiting (strictly < 50)."""
        # orbital=47.99, radius=2.0 → 49.99
        p = self._make_planet(50.0 + 47.99, 50.0, 2.0)
        assert is_orbiting(p, CENTER, ROTATION_RADIUS_LIMIT)

    def test_just_outside_threshold(self):
        """orbital_radius + radius = 50.01 → static (not < 50)."""
        # orbital=48.01, radius=2.0 → 50.01
        p = self._make_planet(50.0 + 48.01, 50.0, 2.0)
        assert not is_orbiting(p, CENTER, ROTATION_RADIUS_LIMIT)

    def test_exactly_on_threshold(self):
        """orbital_radius + radius == 50.0 → static (strict <, not <=)."""
        # orbital=48.0, radius=2.0 → 50.0 exactly
        p = self._make_planet(50.0 + 48.0, 50.0, 2.0)
        assert not is_orbiting(p, CENTER, ROTATION_RADIUS_LIMIT)

    def test_at_center(self):
        """Planet exactly at center: orbital=0 + any positive radius < 50 → orbiting."""
        p = self._make_planet(50.0, 50.0, 1.0)
        assert is_orbiting(p, CENTER, ROTATION_RADIUS_LIMIT)

    def test_minimum_production_radius(self):
        """Production=1 → radius=1+ln(1)=1.0. Check threshold accordingly."""
        # orbital = 48.0, radius = 1.0 → sum = 49.0 < 50 → orbiting
        p = self._make_planet(50.0 + 48.0, 50.0, 1.0)
        assert is_orbiting(p, CENTER, ROTATION_RADIUS_LIMIT)

    def test_custom_threshold(self):
        """Custom rotation_radius_limit parameter is respected."""
        p = self._make_planet(50.0 + 5.0, 50.0, 1.0)  # orbital=5, r=1 → 6
        assert is_orbiting(p, CENTER, 10.0)   # 6 < 10 → orbiting
        assert not is_orbiting(p, CENTER, 5.0)  # 6 >= 5 → static


# ── rotate_point ───────────────────────────────────────────────────────────────

class TestRotatePoint:

    def test_full_circle_returns_to_start(self):
        """After 2π rotation, point should be back at original position."""
        point = (60.0, 50.0)  # 10 units right of center
        result = rotate_point(point, CENTER, 2 * math.pi)
        assert abs(result[0] - point[0]) < 1e-9
        assert abs(result[1] - point[1]) < 1e-9

    def test_quarter_turn_cw(self):
        """pi/2 rotation of (60,50) around (50,50) → should go to (50,60)."""
        # (60,50) is at angle 0 (right). +pi/2 → angle pi/2 → (50+0, 50+10) = (50,60)
        result = rotate_point((60.0, 50.0), CENTER, math.pi / 2)
        assert abs(result[0] - 50.0) < 1e-9
        assert abs(result[1] - 60.0) < 1e-9

    def test_half_turn(self):
        """π rotation of (60,50) around (50,50) → (40,50)."""
        result = rotate_point((60.0, 50.0), CENTER, math.pi)
        assert abs(result[0] - 40.0) < 1e-9
        assert abs(result[1] - 50.0) < 1e-9

    def test_preserves_radius(self):
        """Rotation preserves distance from center."""
        point = (65.0, 55.0)
        r_before = math.sqrt((point[0] - CENTER[0])**2 + (point[1] - CENTER[1])**2)
        result = rotate_point(point, CENTER, 0.1)
        r_after = math.sqrt((result[0] - CENTER[0])**2 + (result[1] - CENTER[1])**2)
        assert abs(r_before - r_after) < 1e-9

    def test_zero_angular_velocity(self):
        """Zero rotation returns the same point."""
        point = (70.0, 40.0)
        result = rotate_point(point, CENTER, 0.0)
        assert abs(result[0] - point[0]) < 1e-9
        assert abs(result[1] - point[1]) < 1e-9

    def test_negative_angle_reverses_positive(self):
        """Rotating by -θ then +θ should return to start."""
        point = (62.0, 47.0)
        mid = rotate_point(point, CENTER, 0.3)
        back = rotate_point(mid, CENTER, -0.3)
        assert abs(back[0] - point[0]) < 1e-9
        assert abs(back[1] - point[1]) < 1e-9

    def test_point_at_center_stays_at_center(self):
        """A point exactly at the center of rotation stays put."""
        result = rotate_point(CENTER, CENTER, 1.0)
        assert abs(result[0] - CENTER[0]) < 1e-9
        assert abs(result[1] - CENTER[1]) < 1e-9

    def test_small_angular_velocity_matches_game_range(self):
        """Angular velocity in [0.025, 0.05] (game range) produces sensible results."""
        point = (58.0, 50.0)  # 8 units right of center
        for av in [0.025, 0.035, 0.05]:
            result = rotate_point(point, CENTER, av)
            # After small rotation, point should be very close to original
            dist_moved = math.sqrt((result[0] - point[0])**2 + (result[1] - point[1])**2)
            # Arc length = r * theta = 8 * av ≈ 0.2 to 0.4 — small but positive
            assert 0.0 < dist_moved < 1.0, f"av={av}: moved {dist_moved:.4f}"


# ── swept_pair_hit ─────────────────────────────────────────────────────────────

class TestSweptPairHit:

    def test_stationary_planet_same_as_segment_circle(self):
        """With P0==P1 (stationary planet), swept_pair_hit agrees with segment_circle_collision
        for interior hits and clear misses. Boundary behavior differs by design:
        swept_pair_hit uses c<=0 (touching = hit), segment_circle_collision uses strict < radius
        (touching = miss), matching the source's sun-check vs planet-check convention."""
        # Clear hit: segment through center
        assert swept_pair_hit((0.0, 50.0), (100.0, 50.0), CENTER, CENTER, 10.0) is True
        # Clear miss: far from circle
        assert swept_pair_hit((0.0, 0.0), (100.0, 0.0), CENTER, CENTER, 10.0) is False
        # Boundary: starting point exactly ON circle surface → swept_pair_hit returns True
        # (c = d²-r² = 100-100 = 0, and c<=0 branch returns True)
        # This is intentional: fleet starting on a planet's surface is captured.
        assert swept_pair_hit((50.0, 40.0), (0.0, 40.0), CENTER, CENTER, 10.0) is True

    def test_moving_planet_hit(self):
        """Fleet and planet moving toward each other should collide."""
        # Fleet moves right: (30,50)→(70,50)
        # Planet moves left: (80,50)→(40,50), radius=5
        # They clearly overlap during the tick
        assert swept_pair_hit((30.0, 50.0), (70.0, 50.0),
                               (80.0, 50.0), (40.0, 50.0), 5.0)

    def test_moving_planet_miss_passes_behind(self):
        """Fleet and planet move parallel, never within r of each other."""
        # Fleet: (0,50)→(10,50); Planet: (0,60)→(10,60), radius=2
        # Constant distance = 10 > 2
        assert not swept_pair_hit((0.0, 50.0), (10.0, 50.0),
                                   (0.0, 60.0), (10.0, 60.0), 2.0)


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
