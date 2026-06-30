"""
cwm/comets.py — Comet spawn, advance, and expiry logic for the Orbit Wars CWM.

Source: orbit_wars_original.py — `generate_comet_paths`, `interpreter` comet sections.

Two distinct expiry points (matching source exactly):
  1. Pre-launch expiry (`expire_comets`): removes comets where path_index is already
     >= len(path) at the START of the tick — i.e., they expired during the previous
     tick's advance phase. Happens BEFORE fleet launch.
  2. Mid-tick expiry: during `advance_comet_positions`, when path_index is incremented
     to >= len(path) this tick. These comets stay in place for collision detection but
     are removed AFTER fleet movement and BEFORE combat. The "black hole" effect
     (issue_1047 Item 1) happens here: fleets that hit such comets have ships vanish.
     The interpreter handles this case; `advance_comet_positions` returns expired PIDs.
"""

from __future__ import annotations

import copy
import math

from cwm.state import (
    State,
    BOARD_SIZE,
    CENTER,
    SUN_RADIUS,
    ROTATION_RADIUS_LIMIT,
    COMET_RADIUS,
    COMET_PRODUCTION,
    COMET_SPAWN_STEPS,
)


# ── Internal helpers (mirrored from reference, no cross-import) ────────────────

def _distance(p1: tuple, p2: tuple) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# ── generate_comet_paths ───────────────────────────────────────────────────────

def generate_comet_paths(
    initial_planets: list,
    angular_velocity: float,
    spawn_step: int,
    comet_planet_ids: list | None = None,
    comet_speed: float = 4.0,
    rng=None,
) -> list | None:
    """Generate 4 symmetric elliptical orbit paths for extra-solar objects.

    Re-implemented faithfully from orbit_wars_original.py (lines 187-345).
    Cannot import from reference because the reference module loads orbit_wars.json
    at import time (which doesn't exist outside the kaggle_environments package).

    Returns list of 4 paths (one per quadrant symmetry), each path a list of
    [x, y] positions at comet_speed units/turn. Returns None on failure (no valid
    path found in 300 attempts).
    """
    import random as _random
    if rng is None:
        rng = _random
    if comet_planet_ids is None:
        comet_planet_ids = set()
    else:
        comet_planet_ids = set(comet_planet_ids)

    for _ in range(300):
        # Highly eccentric ellipse with sun at one focus
        e = rng.uniform(0.75, 0.93)
        a = rng.uniform(60, 150)
        perihelion = a * (1 - e)
        if perihelion < SUN_RADIUS + COMET_RADIUS:
            continue

        b = a * math.sqrt(1 - e ** 2)
        c_val = a * e
        # Orientation: perihelion direction from sun (keep in Q4 quadrant)
        phi = rng.uniform(math.pi / 6, math.pi / 3)

        # Dense sample around perihelion half of orbit
        dense = []
        num = 5000
        for i in range(num):
            t = 0.3 * math.pi + 1.4 * math.pi * i / (num - 1)
            # Ellipse with focus at origin
            ex = c_val + a * math.cos(t)
            ey = b * math.sin(t)
            # Rotate and translate to board
            x = CENTER + ex * math.cos(phi) - ey * math.sin(phi)
            y = CENTER + ex * math.sin(phi) + ey * math.cos(phi)
            dense.append((x, y))

        # Re-sample at constant comet_speed arc-length intervals
        path = [dense[0]]
        cum = 0.0
        target = comet_speed
        for i in range(1, len(dense)):
            cum += _distance(dense[i], dense[i - 1])
            if cum >= target:
                path.append(dense[i])
                target += comet_speed

        # Extract contiguous on-board segment
        board_start = None
        board_end = None
        for i, (x, y) in enumerate(path):
            if 0 <= x <= BOARD_SIZE and 0 <= y <= BOARD_SIZE:
                if board_start is None:
                    board_start = i
                board_end = i

        if board_start is None:
            continue
        visible = path[board_start : board_end + 1]
        if not (5 <= len(visible) <= 40):
            continue

        # Build 4 rotationally symmetric paths (4-fold rotation about center).
        # Q1 and Q3 copies are reflected across the y=x diagonal so all 4
        # copies are 90° rotations of each other.
        paths = [
            [[y, x] for x, y in visible],
            [[BOARD_SIZE - x, y] for x, y in visible],
            [[x, BOARD_SIZE - y] for x, y in visible],
            [[BOARD_SIZE - y, BOARD_SIZE - x] for x, y in visible],
        ]

        # Separate planets into static and orbiting (exclude other comets)
        static_planets = []
        orbiting_planets = []
        for planet in initial_planets:
            if planet[0] in comet_planet_ids:
                continue
            pr = _distance((planet[2], planet[3]), (CENTER, CENTER))
            if pr + planet[4] < ROTATION_RADIUS_LIMIT:
                orbiting_planets.append(planet)
            else:
                static_planets.append(planet)

        valid = True
        buf = COMET_RADIUS + 0.5
        for k, (cx, cy) in enumerate(visible):
            # Check sun clearance
            if _distance((cx, cy), (CENTER, CENTER)) < SUN_RADIUS + COMET_RADIUS:
                valid = False
                break

            # The 4-fold symmetric positions for this path step
            sym_pts = [
                (cy, cx),
                (BOARD_SIZE - cx, cy),
                (cx, BOARD_SIZE - cy),
                (BOARD_SIZE - cy, BOARD_SIZE - cx),
            ]
            # Check against static planets
            for planet in static_planets:
                for sp in sym_pts:
                    if _distance(sp, (planet[2], planet[3])) < planet[4] + buf:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

            # Check against orbiting planets at their actual positions
            game_step = spawn_step - 1 + k
            for planet in orbiting_planets:
                dx = planet[2] - CENTER
                dy = planet[3] - CENTER
                orb_r = math.sqrt(dx ** 2 + dy ** 2)
                init_angle = math.atan2(dy, dx)
                cur_angle = init_angle + angular_velocity * game_step
                px = CENTER + orb_r * math.cos(cur_angle)
                py = CENTER + orb_r * math.sin(cur_angle)
                for sp in sym_pts:
                    if _distance(sp, (px, py)) < planet[4] + COMET_RADIUS:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

        if valid:
            return paths

    return None


# ── expire_comets ──────────────────────────────────────────────────────────────

def expire_comets(state: State) -> State:
    """Remove comets whose path has already ended (path_index >= path length).

    This is the PRE-LAUNCH expiry. It removes comets that were advanced past
    the end of their path during the PREVIOUS tick's advance phase.

    Per source: checked BEFORE fleet launch so agents can't act on expiring comets.
    Removes comet planets from: state.planets, state.initial_planets,
    state.comet_planet_ids. Prunes empty comet groups from state.comets.
    Any garrisoned ships on the comet are lost with it.

    Source reference: orbit_wars_original.py interpreter() lines ~391-412.
    """
    expired_pids: set[int] = set()
    for group in state.comets:
        idx = group["path_index"]
        for i, pid in enumerate(group["planet_ids"]):
            if idx >= len(group["paths"][i]):
                expired_pids.add(pid)

    if not expired_pids:
        return state

    new_state = copy.copy(state)
    new_state.planets          = [p for p in state.planets          if p[0] not in expired_pids]
    new_state.initial_planets  = [p for p in state.initial_planets  if p[0] not in expired_pids]
    new_state.comet_planet_ids = [pid for pid in state.comet_planet_ids if pid not in expired_pids]

    new_comets = []
    for group in state.comets:
        new_pids   = [pid for pid in group["planet_ids"]  if pid not in expired_pids]
        new_paths  = [path for pid, path in zip(group["planet_ids"], group["paths"]) if pid not in expired_pids]
        if new_pids:
            new_comets.append({
                "planet_ids": new_pids,
                "paths":      new_paths,
                "path_index": group["path_index"],
            })
    new_state.comets = new_comets

    return new_state


# ── spawn_comet_group ──────────────────────────────────────────────────────────

def spawn_comet_group(state: State, rng) -> State:
    """Attempt to spawn one group of 4 comets with 4-fold symmetric paths.

    Ship count = min of 4 draws from randint(1, 99) — all 4 comets share the
    same count (per source lines ~430-437).
    Starting position: (-99, -99) — off-board placeholder; first advance places
    them at path[0].
    path_index initialised to -1 so the first advance sets it to 0 (path[0]).

    Returns state UNCHANGED if generate_comet_paths() returns None (no valid path
    found in 300 attempts).

    Called by interpreter ONLY at steps in COMET_SPAWN_STEPS [50,150,250,350,450].
    The step check is in the interpreter; this function always attempts a spawn.

    Source reference: orbit_wars_original.py interpreter() lines ~415-453.
    """
    comet_paths = generate_comet_paths(
        state.initial_planets,
        state.angular_velocity,
        state.step + 1,          # spawn_step = current_step + 1 (matches source)
        state.comet_planet_ids,
        state.comet_speed,
        rng=rng,
    )
    if comet_paths is None:
        return state

    comet_ships = min(
        rng.randint(1, 99),
        rng.randint(1, 99),
        rng.randint(1, 99),
        rng.randint(1, 99),
    )

    next_id = max(p[0] for p in state.planets) + 1 if state.planets else 0

    new_planets        = [list(p) for p in state.planets]
    new_initial        = [list(p) for p in state.initial_planets]
    new_comet_ids      = list(state.comet_planet_ids)
    new_comets         = [
        {
            "planet_ids": list(g["planet_ids"]),
            "paths":      g["paths"],
            "path_index": g["path_index"],
        }
        for g in state.comets
    ]

    group = {"planet_ids": [], "paths": comet_paths, "path_index": -1}
    for i in range(4):
        pid = next_id + i
        group["planet_ids"].append(pid)
        new_comet_ids.append(pid)
        planet = [pid, -1, -99.0, -99.0, COMET_RADIUS, comet_ships, COMET_PRODUCTION]
        new_planets.append(planet)
        new_initial.append(list(planet))
    new_comets.append(group)

    new_state = copy.copy(state)
    new_state.planets          = new_planets
    new_state.initial_planets  = new_initial
    new_state.comet_planet_ids = new_comet_ids
    new_state.comets           = new_comets
    return new_state


# ── advance_comet_positions ────────────────────────────────────────────────────

def advance_comet_positions(state: State) -> tuple[State, list[int]]:
    """Increment path_index for all comets and move each to its new path position.

    For each comet:
      - Increments group["path_index"] by 1.
      - If new index < len(path): moves comet planet to path[new_index].
      - If new index >= len(path): comet has expired THIS TICK. It stays at its
        current position (for swept-collision detection this tick). The returned
        `expired_this_tick` list contains its planet ID.

    Returns
    -------
    (new_state, expired_this_tick_pids)
        new_state              : State with updated planet positions and path_indices.
        expired_this_tick_pids : PIDs of comets that just expired (path exhausted).
                                 The interpreter removes these AFTER fleet movement /
                                 collision detection but BEFORE combat resolution.
                                 Fleets that hit these comets have ships silently
                                 deleted — the "black hole" effect (issue_1047 Item 1).

    Source reference: orbit_wars_original.py interpreter() lines ~535-570.
    """
    # Build a pid -> planet-list mapping for O(1) lookup
    planet_map: dict[int, list] = {p[0]: p for p in state.planets}

    new_comets: list[dict] = []
    expired_this_tick: list[int] = []

    for group in state.comets:
        new_idx = group["path_index"] + 1
        new_group = {
            "planet_ids": list(group["planet_ids"]),
            "paths":      group["paths"],       # shared read-only reference
            "path_index": new_idx,
        }
        for i, pid in enumerate(group["planet_ids"]):
            planet = planet_map.get(pid)
            if planet is None:
                continue
            p_path = group["paths"][i]
            if new_idx >= len(p_path):
                # Expired this tick: comet stays put (old position unchanged)
                expired_this_tick.append(pid)
            else:
                planet[2] = p_path[new_idx][0]
                planet[3] = p_path[new_idx][1]
        new_comets.append(new_group)

    new_state = copy.copy(state)
    # Rebuild planets list with updated positions (planet_map entries were mutated)
    new_state.planets = list(planet_map.values())
    new_state.comets  = new_comets
    return new_state, expired_this_tick


def remove_expired_comets(state: State, expired_pids: list[int]) -> State:
    """Remove the comets that expired THIS tick (mid-tick expiry).

    Called by the interpreter AFTER fleet movement / collision detection but
    BEFORE combat resolution. This is the second expiry point — distinct from
    `expire_comets` (pre-launch expiry).

    Matches source: orbit_wars_original.py interpreter() lines ~597-614.
    """
    if not expired_pids:
        return state
    expired_set = set(expired_pids)
    new_state = copy.copy(state)
    new_state.planets          = [p for p in state.planets          if p[0] not in expired_set]
    new_state.initial_planets  = [p for p in state.initial_planets  if p[0] not in expired_set]
    new_state.comet_planet_ids = [pid for pid in state.comet_planet_ids if pid not in expired_set]

    new_comets = []
    for group in state.comets:
        new_pids  = [pid for pid in group["planet_ids"]  if pid not in expired_set]
        new_paths = [path for pid, path in zip(group["planet_ids"], group["paths"]) if pid not in expired_set]
        if new_pids:
            new_comets.append({
                "planet_ids": new_pids,
                "paths":      new_paths,
                "path_index": group["path_index"],
            })
    new_state.comets = new_comets
    return new_state
