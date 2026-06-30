"""
cwm/interpreter.py — Code World Model transition function for Orbit Wars.

Implements cwm_apply_joint_action, cwm_is_terminal, cwm_get_rewards.

Turn order (matches orbit_wars_original.py interpreter() exactly):
  1. Pre-launch comet expiry  (comets whose path ended in the PREVIOUS tick)
  2. Comet spawning           (at steps 49,149,249,349,449 → spawns at 50,150,...)
  3. Fleet launch             (process joint_action; drop illegal moves silently)
  4. Production               (owned planets += production)
  5. Compute planet paths     (orbiting planet new positions; comet path advance)
  6. Fleet movement           (swept-pair collision; OOB; sun-crossing)
  7. Apply planet positions   (write new_pos; remove mid-tick expired comets;
                               remove dead fleets)
  8. Combat resolution        (resolve combat_queue)
  step += 1

Termination (Issue #1047 Item 6, still present in source):
  step >= episodeSteps - 2  (not episodeSteps; game is 499 turns not 500)
  OR <= 1 player remains with any planets/fleets (elimination).

Source reference: orbit_wars_original.py, interpreter() function.
"""

from __future__ import annotations

import copy
import math
import random as _random

from cwm.state import (
    State,
    BOARD_SIZE,
    CENTER,
    SUN_RADIUS,
    ROTATION_RADIUS_LIMIT,
    COMET_SPAWN_STEPS,
    total_ships,
)
from cwm.geometry import fleet_speed, swept_pair_hit
from cwm.comets import expire_comets, spawn_comet_group
from cwm.combat import resolve_combat


# ── State deep-copy helper ─────────────────────────────────────────────────────

def _copy_state(state: State) -> State:
    """Return a deep copy of State (mutable lists fully copied)."""
    return State(
        planets=[list(p) for p in state.planets],
        fleets=[list(f) for f in state.fleets],
        initial_planets=[list(p) for p in state.initial_planets],
        comets=[
            {
                "planet_ids": list(g["planet_ids"]),
                "paths": g["paths"],   # read-only paths shared; not mutated
                "path_index": g["path_index"],
            }
            for g in state.comets
        ],
        comet_planet_ids=list(state.comet_planet_ids),
        step=state.step,
        next_fleet_id=state.next_fleet_id,
        angular_velocity=state.angular_velocity,
        num_players=state.num_players,
        episode_steps=state.episode_steps,
        ship_speed=state.ship_speed,
        comet_speed=state.comet_speed,
    )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _point_to_segment_distance(p: tuple, v: tuple, w: tuple) -> float:
    """Minimum distance from point p to segment v-w. Matches source exactly."""
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0.0:
        return math.sqrt((p[0] - v[0]) ** 2 + (p[1] - v[1]) ** 2)
    t = max(0.0, min(1.0,
        ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2
    ))
    proj = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
    return math.sqrt((p[0] - proj[0]) ** 2 + (p[1] - proj[1]) ** 2)


# ── Main transition function ───────────────────────────────────────────────────

def cwm_apply_joint_action(
    state: State,
    joint_action: list,
    config=None,
    spawn_rng=None,
) -> State:
    """Apply one full game tick and return the successor state.

    Parameters
    ----------
    state : State
        Current game state. Not mutated.
    joint_action : list
        Length == state.num_players. Each element is a player's action:
        a list of [from_planet_id, angle, num_ships] moves (or [] for no-op).
    config : optional
        Unused (config constants are stored in State). Accepted for signature
        compatibility with tests that pass config.
    spawn_rng : optional random.Random
        RNG for comet path generation. If None, a fresh random.Random() is used.
        For trajectory validation, pass the seeded RNG derived from the episode
        seed: random.Random(f"orbit_wars-comet-{seed}-{step+1}").

    Returns
    -------
    State
        Successor state with step incremented by 1.
    """
    s = _copy_state(state)

    # ── Step 1: Pre-launch comet expiry ───────────────────────────────────────
    # Remove comets whose path_index >= path length at the START of this tick.
    # These were advanced to expiry during the PREVIOUS tick's position step.
    # Source: interpreter() lines ~391-412.
    s = expire_comets(s)

    # ── Step 2: Comet spawning ────────────────────────────────────────────────
    # Check: (current_step + 1) in COMET_SPAWN_STEPS.
    # Source uses `step = get(obs0, "step", 0)` for this check.
    # At step=49: (49+1)=50 in [50,150,...] → spawn.
    # Source: interpreter() lines ~415-453.
    if (s.step + 1) in COMET_SPAWN_STEPS:
        if spawn_rng is None:
            spawn_rng = _random.Random()
        s = spawn_comet_group(s, spawn_rng)

    # ── Step 3: Fleet launch ──────────────────────────────────────────────────
    # Process all player actions. Illegal moves silently dropped.
    # Source: interpreter() "0. Fleet Launch", lines ~455-490.
    planet_map: dict[int, list] = {p[0]: p for p in s.planets}

    for player_id, action in enumerate(joint_action):
        if not action or not isinstance(action, list):
            continue
        for move in action:
            if len(move) != 3:
                continue
            from_id, angle, num_ships = move
            num_ships = int(num_ships)  # sanitize to integer (matches source)

            from_planet = planet_map.get(from_id)
            if from_planet is None:
                continue  # unowned or nonexistent planet
            if from_planet[1] != player_id:
                continue  # not owned by this player
            if num_ships <= 0:
                continue  # zero or negative ships
            if from_planet[5] < num_ships:
                continue  # insufficient garrison

            # Deduct from garrison
            from_planet[5] -= num_ships

            # Start fleet just outside the planet radius
            start_x = from_planet[2] + math.cos(angle) * (from_planet[4] + 0.1)
            start_y = from_planet[3] + math.sin(angle) * (from_planet[4] + 0.1)
            s.fleets.append([
                s.next_fleet_id,
                player_id,
                start_x,
                start_y,
                angle,
                from_id,
                num_ships,
            ])
            s.next_fleet_id += 1

    # ── Step 4: Production ────────────────────────────────────────────────────
    # All owned planets (including comets) generate ships.
    # Source: interpreter() "1. Production", lines ~492-494.
    for planet in s.planets:
        if planet[1] != -1:
            planet[5] += planet[6]

    # ── Step 5: Compute planet end-of-tick positions ──────────────────────────
    # For orbiting planets: compute new position from initial angle + av * step.
    # For comets: increment path_index, get new position.
    # Source: interpreter() "2. Compute each planet's end-of-tick position",
    #         lines ~496-571.
    #
    # NOTE: source uses `step = get(obs0, "step", 1)` (default 1, not 0).
    # In our CWM, state.step is always provided.
    # planet_paths: {pid: (old_pos, new_pos, check_collision)}
    #   check_collision=False means the planet appears mid-tick (first comet
    #   placement) and should not be checked against fleets this tick.

    step = s.step   # current step (used with default=1 in source)
    av = s.angular_velocity
    comet_pid_set = set(s.comet_planet_ids)
    initial_by_id = {p[0]: p for p in s.initial_planets}

    planet_paths: dict[int, tuple] = {}
    expired_this_tick: list[int] = []

    # Regular (non-comet) planets
    for planet in s.planets:
        if planet[0] in comet_pid_set:
            continue
        old_pos = (planet[2], planet[3])
        new_pos = old_pos
        initial_p = initial_by_id.get(planet[0])
        if initial_p is not None:
            dx = initial_p[2] - CENTER
            dy = initial_p[3] - CENTER
            r = math.sqrt(dx * dx + dy * dy)
            if r + planet[4] < ROTATION_RADIUS_LIMIT:
                init_angle = math.atan2(dy, dx)
                cur_angle = init_angle + av * step
                new_pos = (
                    CENTER + r * math.cos(cur_angle),
                    CENTER + r * math.sin(cur_angle),
                )
        planet_paths[planet[0]] = (old_pos, new_pos, True)

    # Comets: increment path_index and record new positions
    for group in s.comets:
        group["path_index"] += 1
        idx = group["path_index"]
        for i, pid in enumerate(group["planet_ids"]):
            planet = planet_map.get(pid)
            if planet is None:
                continue
            p_path = group["paths"][i]
            old_pos = (planet[2], planet[3])
            if idx >= len(p_path):
                # Comet path exhausted this tick: stays put, will be removed
                # after fleet movement (mid-tick expiry / black hole, Item 1).
                expired_this_tick.append(pid)
                planet_paths[pid] = (old_pos, old_pos, True)
            else:
                new_pos = (p_path[idx][0], p_path[idx][1])
                # First placement: old_pos is off-board (-99,-99), skip collision
                check = old_pos[0] >= 0
                planet_paths[pid] = (old_pos, new_pos, check)

    # ── Step 6: Fleet movement ────────────────────────────────────────────────
    # Move fleets, detect collisions. Source: "3. Fleet Movement" ~lines 573-603.
    #
    # Collision priority (Issue #1047 Item 2): first planet in s.planets list wins.
    # Out-of-bounds and sun-crossing checked only if no planet collision.

    max_speed = s.ship_speed
    fleets_to_remove: list = []
    combat_queue: dict[int, list] = {p[0]: [] for p in s.planets}

    for fleet in s.fleets:
        angle = fleet[4]
        ships = fleet[6]
        speed = 1.0 + (max_speed - 1.0) * (math.log(max(1, ships)) / math.log(1000)) ** 1.5
        speed = min(speed, max_speed)
        old_pos = (fleet[2], fleet[3])
        fleet[2] += math.cos(angle) * speed
        fleet[3] += math.sin(angle) * speed
        new_pos = (fleet[2], fleet[3])

        # Planet collision check (first-match, list order — Issue #1047 Item 2)
        hit_planet = False
        for planet in s.planets:
            path = planet_paths.get(planet[0])
            if path is None or not path[2]:
                continue
            p_old, p_new, _ = path
            if swept_pair_hit(old_pos, new_pos, p_old, p_new, planet[4]):
                combat_queue[planet[0]].append((fleet[1], fleet[6]))
                fleets_to_remove.append(id(fleet))
                hit_planet = True
                break
        if hit_planet:
            continue

        # Out-of-bounds check
        if not (0 <= fleet[2] <= BOARD_SIZE and 0 <= fleet[3] <= BOARD_SIZE):
            fleets_to_remove.append(id(fleet))
            continue

        # Sun-crossing check (strict distance < SUN_RADIUS — matches source)
        if _point_to_segment_distance((CENTER, CENTER), old_pos, new_pos) < SUN_RADIUS:
            fleets_to_remove.append(id(fleet))
            continue

    # ── Step 7: Apply planet positions + remove mid-tick expired comets ───────
    # Source: "4. Apply planet movement" + mid-tick expiry removal + fleet cleanup.

    # Apply new positions to all planets
    for planet in s.planets:
        path = planet_paths.get(planet[0])
        if path is not None:
            planet[2], planet[3] = path[1]

    # Remove mid-tick expired comets (BEFORE combat — the black hole effect).
    if expired_this_tick:
        expired_set = set(expired_this_tick)
        s.planets = [p for p in s.planets if p[0] not in expired_set]
        s.initial_planets = [p for p in s.initial_planets if p[0] not in expired_set]
        s.comet_planet_ids = [pid for pid in s.comet_planet_ids if pid not in expired_set]
        new_comets = []
        for group in s.comets:
            new_pids = [pid for pid in group["planet_ids"] if pid not in expired_set]
            new_paths = [
                path for pid, path in zip(group["planet_ids"], group["paths"])
                if pid not in expired_set
            ]
            if new_pids:
                new_comets.append({
                    "planet_ids": new_pids,
                    "paths": new_paths,
                    "path_index": group["path_index"],
                })
        s.comets = new_comets
        # Remove combat_queue entries for expired comets — but DON'T: source
        # leaves them in, and combat resolution skips them (planet not found).
        # This produces the black hole effect. Do NOT remove from combat_queue.

    # Remove dead fleets (OOB, sun, planet-hit)
    fleets_to_remove_set = set(fleets_to_remove)
    s.fleets = [f for f in s.fleets if id(f) not in fleets_to_remove_set]

    # ── Step 8: Combat resolution ─────────────────────────────────────────────
    s = resolve_combat(s, combat_queue)

    # ── Advance step ──────────────────────────────────────────────────────────
    s.step += 1

    return s


# ── Terminal / reward functions ────────────────────────────────────────────────

def cwm_is_terminal(state: State, config=None) -> bool:
    """True iff the game has ended.

    Termination conditions (source-verified):
      1. step >= episodeSteps - 2
         (Issue #1047 Item 6, still present: 500-step game ends at step 498,
          effectively 499 turns. Replicated exactly from source line ~645.)
      2. <= 1 player remains with any planets or fleets (elimination).
         Relevant in 4p; degenerate in 2p (draw is possible but rare).

    Source: orbit_wars_original.py interpreter() lines ~644-658.
    """
    # Use config.episodeSteps if provided and differs from state default
    if config is not None:
        ep_steps = getattr(config, "episodeSteps", None) or state.episode_steps
    else:
        ep_steps = state.episode_steps

    # Condition 1: step limit (off-by-one matches source: episodeSteps - 2)
    if state.step >= ep_steps - 2:
        return True

    # Condition 2: elimination
    alive: set[int] = set()
    for p in state.planets:
        if p[1] != -1:
            alive.add(p[1])
    for f in state.fleets:
        alive.add(f[1])

    if len(alive) <= 1:
        return True

    return False


def cwm_get_rewards(state: State) -> list[float]:
    """Return total ship counts for each player.

    Matches the source's scoring: sum of ships on owned planets + ships in
    owned fleets. Used both as the terminal score and as a value signal.

    Returns a list of length state.num_players.
    Source: orbit_wars_original.py interpreter() lines ~659-670.
    """
    scores = [0.0] * state.num_players
    for p in state.planets:
        if 0 <= p[1] < state.num_players:
            scores[p[1]] += p[5]
    for f in state.fleets:
        if 0 <= f[1] < state.num_players:
            scores[f[1]] += f[6]
    return scores
