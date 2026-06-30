"""
cwm/symmetry.py — 4-fold map symmetry canonicalization and augmentation (Module 6).

Exploits the game board's 4-fold mirror symmetry:
  0: identity          (x, y) → (x, y)
  1: mirror-x          (x, y) → (100 - x, y)
  2: mirror-y          (x, y) → (x, 100 - y)
  3: mirror-both       (x, y) → (100 - x, 100 - y)

The sun at (50, 50) is invariant under all 4 transforms.

Fleet angles transform consistently with the coordinate flip:
  transform 0: angle → angle              (identity)
  transform 1: angle → π - angle         (flip cos, keep sin)
  transform 2: angle → -angle (mod 2π)   (keep cos, flip sin)
  transform 3: angle → π + angle (mod 2π)(flip both)
"""

from __future__ import annotations

import math

from cwm.state import State

BOARD_SIZE = 100.0
_CENTER = 50.0


def _transform_coord(x: float, y: float, tid: int) -> tuple[float, float]:
    """Apply coordinate transform `tid` to (x, y)."""
    nx = BOARD_SIZE - x if tid in (1, 3) else x
    ny = BOARD_SIZE - y if tid in (2, 3) else y
    return (nx, ny)


def _transform_angle(angle: float, tid: int) -> float:
    """Transform a direction angle consistently with coordinate transform `tid`."""
    if tid == 0:
        return angle
    if tid == 1:
        # mirror-x: cos → -cos, sin unchanged → new_angle = π - angle
        return (math.pi - angle) % (2 * math.pi)
    if tid == 2:
        # mirror-y: cos unchanged, sin → -sin → new_angle = -angle
        return (-angle) % (2 * math.pi)
    # tid == 3: mirror-both: cos → -cos, sin → -sin → new_angle = π + angle
    return (math.pi + angle) % (2 * math.pi)


def apply_transform(state: State, transform_id: int) -> State:
    """Return a new State with all coordinates mapped under transform_id.

    Planet and fleet (x, y) positions are remapped; fleet angles are updated
    to remain consistent with the coordinate flip.  Non-spatial fields
    (owners, ships, production, step, etc.) are unchanged.
    """
    tid = transform_id

    new_planets = []
    for p in state.planets:
        np_ = list(p)
        np_[2], np_[3] = _transform_coord(p[2], p[3], tid)
        new_planets.append(np_)

    new_initial = []
    for p in state.initial_planets:
        np_ = list(p)
        np_[2], np_[3] = _transform_coord(p[2], p[3], tid)
        new_initial.append(np_)

    new_fleets = []
    for f in state.fleets:
        nf = list(f)
        nf[2], nf[3] = _transform_coord(f[2], f[3], tid)
        nf[4] = _transform_angle(f[4], tid)
        new_fleets.append(nf)

    # Comet paths also need coordinate transformation
    new_comets = []
    for group in state.comets:
        new_paths = []
        for path in group["paths"]:
            new_path = []
            for pt in path:
                nx, ny = _transform_coord(pt[0], pt[1], tid)
                new_path.append([nx, ny])
            new_paths.append(new_path)
        new_comets.append({
            "planet_ids": list(group["planet_ids"]),
            "paths": new_paths,
            "path_index": group["path_index"],
        })

    return State(
        planets=new_planets,
        fleets=new_fleets,
        initial_planets=new_initial,
        comets=new_comets,
        comet_planet_ids=list(state.comet_planet_ids),
        step=state.step,
        next_fleet_id=state.next_fleet_id,
        angular_velocity=state.angular_velocity,
        num_players=state.num_players,
        episode_steps=state.episode_steps,
        ship_speed=state.ship_speed,
        comet_speed=state.comet_speed,
    )


def canonical_transform_id(state: State, observing_player: int = 0) -> int:
    """Return which of the 4 transforms maps this state into the canonical quadrant.

    The canonical quadrant is defined as: the observing player's home planet
    satisfies x <= 50 and y <= 50.

    The home planet is taken as the first planet owned by observing_player.
    If no such planet exists, returns 0 (identity).
    """
    home = next((p for p in state.planets if p[1] == observing_player), None)
    if home is None:
        return 0

    x, y = home[2], home[3]
    need_flip_x = x > _CENTER
    need_flip_y = y > _CENTER

    if not need_flip_x and not need_flip_y:
        return 0   # identity
    if need_flip_x and not need_flip_y:
        return 1   # mirror-x
    if not need_flip_x and need_flip_y:
        return 2   # mirror-y
    return 3       # mirror-both


def augment_batch(states: list[State]) -> list[State]:
    """Return a list 4x the length of the input.

    Each input state appears under all 4 symmetry transforms
    (transforms 0, 1, 2, 3), in that order, grouped by original state.
    """
    result = []
    for state in states:
        for tid in range(4):
            result.append(apply_transform(state, tid))
    return result
