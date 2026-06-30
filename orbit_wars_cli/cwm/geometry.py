"""
cwm/geometry.py — Pure geometry functions for the Orbit Wars CWM.

No State dependency. All functions operate on raw numbers and tuples.
Coordinates use the simulation's native convention: index 0 is the
cos-axis (right = 0 rad), index 1 is the sin-axis (down = pi/2 rad).
See cwm/state.py §X/Y COORDINATE NOTE for full explanation.
"""

from __future__ import annotations

import math


# ── Fleet speed ────────────────────────────────────────────────────────────────

def fleet_speed(num_ships: int, max_speed: float = 6.0) -> float:
    """Compute fleet travel speed from ship count.

    Formula (from orbit_wars_original.py fleet-movement section):
        speed = 1.0 + (max_speed - 1.0) * (log(ships) / log(1000)) ** 1.5

    Anchors (max_speed=6.0):
        1 ship   → 1.0  (log(1) = 0)
        ~500 ships → ~5.3
        1000 ships → 6.0 (log(1000)/log(1000) = 1, **1.5 = 1)

    Clamped: ships < 1 is treated as 1 (log undefined at 0).
    Result is also clamped to max_speed (matches source's `speed = min(speed, max_speed)`).
    """
    ships = max(1, num_ships)
    speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
    return min(speed, max_speed)


# ── Segment–circle collision (stationary circle) ───────────────────────────────

def _point_to_segment_distance(p: tuple, v: tuple, w: tuple) -> float:
    """Minimum distance from point p to line segment v-w.

    Matches the source function point_to_segment_distance() exactly.
    """
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0.0:
        return math.sqrt((p[0] - v[0]) ** 2 + (p[1] - v[1]) ** 2)
    t = max(0.0, min(1.0,
        ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2
    ))
    proj = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
    return math.sqrt((p[0] - proj[0]) ** 2 + (p[1] - proj[1]) ** 2)


def segment_circle_collision(
    p0: tuple, p1: tuple, center: tuple, radius: float
) -> bool:
    """True iff segment p0→p1 passes strictly within `radius` of `center`.

    Used for:
      - Fleet vs. sun:    segment_circle_collision(old, new, (CENTER, CENTER), SUN_RADIUS)
      - Fleet vs. static planet (fallback; see swept_pair_hit for moving planets)

    Implements point_to_segment_distance(center, p0, p1) < radius (strict, matching
    source's sun-crossing check).
    """
    return _point_to_segment_distance(center, p0, p1) < radius


# ── Swept-pair hit (moving fleet vs. moving planet) ────────────────────────────

def swept_pair_hit(
    A: tuple, B: tuple, P0: tuple, P1: tuple, r: float
) -> bool:
    """True iff a fleet moving A→B and a planet moving P0→P1 come within r
    of each other for some t ∈ [0, 1].

    Linearises both motions over the tick (matches source swept_pair_hit exactly).
    Used in interpreter for all fleet-vs-planet collision detection.

    For a stationary planet (P0 == P1) this reduces to segment_circle_collision.
    """
    d0x = A[0] - P0[0]
    d0y = A[1] - P0[1]
    dvx = (B[0] - A[0]) - (P1[0] - P0[0])
    dvy = (B[1] - A[1]) - (P1[1] - P0[1])
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return t2 >= 0.0 and t1 <= 1.0


# ── Orbital status ─────────────────────────────────────────────────────────────

def is_orbiting(planet: list, center: tuple, rotation_radius_limit: float) -> bool:
    """True iff orbital_radius + planet_radius < rotation_radius_limit.

    Source: orbit_wars_original.py uses ROTATION_RADIUS_LIMIT = 50.0.
    Planets satisfying this condition rotate around the sun each tick.

    Parameters
    ----------
    planet : list
        Planet list [id, owner, x, y, radius, ships, production] OR any sequence
        where [2]=x, [3]=y, [4]=radius.
    center : tuple
        (cx, cy) — normally (CENTER, CENTER) = (50.0, 50.0).
    rotation_radius_limit : float
        50.0 in the current game config.
    """
    orbital_radius = math.sqrt(
        (planet[2] - center[0]) ** 2 + (planet[3] - center[1]) ** 2
    )
    return orbital_radius + planet[4] < rotation_radius_limit


# ── Single-step rotation ───────────────────────────────────────────────────────

def rotate_point(point: tuple, center: tuple, angular_velocity: float) -> tuple:
    """Rotate `point` one step around `center` by `angular_velocity` radians.

    Returns the new (x, y) after one tick of rotation. For exact position at
    tick N from the initial angle, prefer the absolute formula in state.py's
    planet_current_pos() to avoid floating-point drift accumulation.
    """
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    r = math.sqrt(dx * dx + dy * dy)
    if r == 0.0:
        return point
    angle = math.atan2(dy, dx) + angular_velocity
    return (center[0] + r * math.cos(angle), center[1] + r * math.sin(angle))
