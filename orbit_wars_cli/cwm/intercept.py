"""
cwm/intercept.py — Lead-targeting solver (Module 2).

Given a source position, target planet, fleet size, and departure turn,
computes the firing angle so the fleet arrives exactly at the target's
predicted position, accounting for orbital motion and fleet speed.

Algorithm: fixed-point iteration.
  1. Start with target's position at depart_turn.
  2. Compute travel time from source to that position at the fleet's speed.
  3. Compute target's predicted position at (depart_turn + travel_time).
  4. Derive the new firing angle from source to that predicted position.
  5. Repeat until the angle changes by less than `tol`, or max_iters is reached.

For a static target the first iteration is already exact.  For orbiting
targets convergence typically occurs in 2–4 iterations.

Returns None when no convergent solution is found within max_iters, or
when the target planet does not exist in the state.
"""

from __future__ import annotations

import math

from cwm.state import State, CENTER, ROTATION_RADIUS_LIMIT
from cwm.geometry import fleet_speed


def _predict_pos(
    planet: list,
    initial_planet: list | None,
    av: float,
    future_step: float,
) -> tuple[float, float]:
    """Predict planet position at absolute game step `future_step`.

    Orbiting planets (orbital_radius + radius < ROTATION_RADIUS_LIMIT) rotate
    around the sun using the absolute formula from state.py.  Static planets
    return their current stored position unchanged.
    """
    if initial_planet is None:
        return (planet[2], planet[3])
    dx = initial_planet[2] - CENTER
    dy = initial_planet[3] - CENTER
    r = math.sqrt(dx * dx + dy * dy)
    if r + planet[4] < ROTATION_RADIUS_LIMIT:
        init_angle = math.atan2(dy, dx)
        ang = init_angle + av * future_step
        return (CENTER + r * math.cos(ang), CENTER + r * math.sin(ang))
    return (planet[2], planet[3])


def solve_intercept(
    source_pos: tuple[float, float],
    target_planet_id: int,
    fleet_size: int,
    depart_turn: int,
    state: State,
    max_iters: int = 20,
    tol: float = 1e-4,
) -> float | None:
    """Compute the firing angle to intercept target_planet_id at its future position.

    Returns the angle in radians in [0, 2π), or None if:
      - target_planet_id is not found in state.planets
      - the solver does not converge within max_iters iterations
    """
    # Locate target in state
    target = next((p for p in state.planets if p[0] == target_planet_id), None)
    if target is None:
        return None

    # Locate initial planet entry (for orbital prediction)
    initial_target = next(
        (p for p in state.initial_planets if p[0] == target_planet_id), None
    )

    speed = fleet_speed(fleet_size, state.ship_speed)
    av = state.angular_velocity

    # Start: target's position at departure turn
    aim = _predict_pos(target, initial_target, av, float(depart_turn))

    prev_angle: float | None = None

    for _ in range(max_iters):
        dist = math.hypot(aim[0] - source_pos[0], aim[1] - source_pos[1])
        eta = dist / speed if speed > 0.0 else 0.0
        new_aim = _predict_pos(target, initial_target, av, depart_turn + eta)

        new_angle = math.atan2(new_aim[1] - source_pos[1],
                               new_aim[0] - source_pos[0])

        if prev_angle is not None:
            # Wrap-aware angle difference
            diff = abs(math.atan2(math.sin(new_angle - prev_angle),
                                  math.cos(new_angle - prev_angle)))
            if diff < tol:
                return new_angle % (2 * math.pi)

        prev_angle = new_angle
        aim = new_aim

    # Did not converge within max_iters
    return None
