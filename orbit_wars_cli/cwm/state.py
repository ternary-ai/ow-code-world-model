"""
cwm/state.py — Code World Model state representation for Orbit Wars.

=============================================================================
OBS SCHEMA  (all arrays are plain Python lists; no numpy in obs)
=============================================================================

Field                   Type          Description
──────────────────────  ────────────  ─────────────────────────────────────────
planets                 list[list]    All planets INCLUDING comets.
                                      Each entry: [id, owner, x, y, radius,
                                      ships, production]
                                        id         int   unique planet id
                                        owner      int   player 0-3, or -1 (neutral)
                                        x          float coord-axis 0 (see note §X/Y)
                                        y          float coord-axis 1
                                        radius     float 1 + ln(production)
                                        ships      int   current garrison
                                        production int   ships/turn when owned (1-5)

fleets                  list[list]    Active fleets.
                                      Each entry: [id, owner, x, y, angle,
                                      from_planet_id, ships]
                                        id             int
                                        owner          int   player 0-3
                                        x              float current position
                                        y              float current position
                                        angle          float direction (radians)
                                        from_planet_id int
                                        ships          int   fixed at launch

player                  int           This agent's player id (0-3).

angular_velocity        float         Rotation speed of orbiting planets
                                      (radians/turn, 0.025–0.05, fixed per game).

initial_planets         list[list]    Planet positions at game start
                                      (same schema as planets). Used to compute
                                      current orbiting-planet positions:
                                        angle_t = atan2(ip[3]-50, ip[2]-50)
                                                  + angular_velocity * step
                                        x_t = 50 + r * cos(angle_t)
                                        y_t = 50 + r * sin(angle_t)
                                      where r = sqrt((ip[2]-50)²+(ip[3]-50)²).

comets                  list[dict]    Active comet group data.
                                      Each entry: {
                                        planet_ids: [int, ...]  (4 entries/group)
                                        paths:      [[x,y], ...]  per-comet path
                                        path_index: int  current step into path
                                      }

comet_planet_ids        list[int]     Planet IDs that are comets. Check membership
                                      here to distinguish comets from normal planets.

next_fleet_id           int           Monotonically increasing fleet ID counter.

step                    int           Current turn (0-indexed).

remainingOverageTime    float         Shared overage time bank (seconds).

=============================================================================
ACTION FORMAT
=============================================================================

Return a list of moves: [[from_planet_id, direction_angle, num_ships], ...]
  from_planet_id  int    planet id you own with ships > 0
  direction_angle float  radians (0 = +x direction, pi/2 = +y direction)
  num_ships       int    1 <= num_ships <= garrison

Empty list [] is a valid no-op. Invalid moves are silently dropped by the
interpreter's process_moves().

=============================================================================
NAMEDTUPLE FIELD ORDERS  (from orbit_wars_original.py lines 10-15)
=============================================================================

Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])
Fleet  = namedtuple("Fleet",  ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"])

=============================================================================
CONSTANTS  (from orbit_wars_original.py lines 17-28)
=============================================================================

BOARD_SIZE            = 100.0
CENTER                = 50.0          (BOARD_SIZE / 2.0)
SUN_RADIUS            = 10.0
ROTATION_RADIUS_LIMIT = 50.0          (orbital_radius + planet_radius < 50 → orbiting)
COMET_RADIUS          = 1.0
COMET_PRODUCTION      = 1
PLANET_CLEARANCE      = 7
COMET_SPAWN_STEPS     = [50, 150, 250, 350, 450]

=============================================================================
§X/Y COORDINATE NOTE  (see issue_1047_status.md Item 7)
=============================================================================

generate_planets() stores coordinates as [id, owner, y_val, x_val, ...] for the
Q1 planet in each symmetric group — the sine component at index 2 (labeled .x)
and the cosine component at index 3 (labeled .y). This is intentional for 4-fold
rotational symmetry. The entire interpreter uses planet[2] / planet[3] (not names)
consistently. Fleet movement applies cos(angle) to [2] and sin(angle) to [3].

CWM RULE: always use planet[2] / fleet[2] (or .x) as "coord-0" and
planet[3] / fleet[3] (or .y) as "coord-1". Never introduce a coord swap.
Direction to target: atan2(t[3] - s[3], t[2] - s[2]).

=============================================================================
NUM_PLAYERS DETECTION  (Phase 0 item 4 — derivable from obs/config alone)
=============================================================================

Primary method (preferred): config.agentCount
  - The kaggle_environments framework passes config when the agent accepts
    two parameters: agent(obs, config).
  - config.agentCount == 2 for 2-player, == 4 for 4-player.

Fallback (no config): at step 0, count distinct non-(-1) owners in obs.planets:
  num_players = len({p[1] for p in obs["planets"] if p[1] != -1})
  → returns 2 (2p) or 4 (4p).

IMPORTANT: cache the result at game start (step 0). Later in the game, players
may be eliminated, making the owner count unreliable as a player-count signal.

=============================================================================
TURN ORDER  (source-verified; see interpreter() in orbit_wars_original.py)
=============================================================================

Per tick, in source order:
  1. Comet expiration   — remove comets where current path_index >= path length
                          (i.e., expired in a prior tick and not yet cleaned)
  2. Comet spawning     — at (step+1) in COMET_SPAWN_STEPS; 4 new comets added
  3. Fleet launch       — process player actions; deduct garrison; invalid moves
                          silently dropped
  4. Production         — all owned planets (including comets) += production
  5. Compute positions  — advance path_index for all comets; compute new planet
                          positions (orbiting planets rotate); mark comets that
                          now exceed path length as "expire-this-tick"
  6. Fleet movement     — move fleets; swept-pair collision vs. (old, new) planet
                          positions; out-of-bounds and sun-crossing removal
  7. Apply positions    — write new planet coordinates; remove expire-this-tick
                          comets (BEFORE combat — ships hitting expiring comets
                          vanish; see issue_1047_status.md Item 1)
  8. Combat resolution  — resolve all combat_lists entries

=============================================================================
COMBAT RESOLUTION MODEL  (source-verified; issue_1047_status.md Items 1 & 4)
=============================================================================

For each planet with arriving fleets:
  Step A — Fleet battle (fleet ships only; garrison NOT included):
    player_ships = {owner: sum(fleet.ships) for each arriving fleet}
    sorted descending by ships.
    top-2 fight: survivor = sorted[0].ships - sorted[1].ships
    sorted[2+] owners' ships VANISH (Item 4, still present in source).
    Tie (sorted[0] == sorted[1]): survivor = 0, no fleet winner.

  Step B — Fleet winner vs. garrison:
    if survivor > 0 and survivor_owner == planet.owner:
        garrison += survivor                    (friendly reinforcement)
    elif survivor > 0 and survivor_owner != planet.owner:
        garrison -= survivor
        if garrison < 0:
            planet.owner = survivor_owner
            garrison = abs(garrison)            (conquest)
    # Tie in Step A → garrison untouched, owner unchanged.

=============================================================================
ISSUE #1047 SUMMARY  (see reference/issue_1047_status.md for full detail)
=============================================================================

Item 1 (Black Hole)        : STILL PRESENT — ships hitting expiring comets vanish.
Item 2 (Tunneling)         : STILL PRESENT — first-in-list planet wins collision.
Item 4 (N-way erasure)     : STILL PRESENT — 3rd+ fleet owners' ships vanish.
Item 5 (Step defaults)     : STILL PRESENT — but moot; step always provided.
Item 6 (episodeSteps-2)    : STILL PRESENT — use episodeSteps-2 in cwm_is_terminal.
Item 7 (X/Y label)         : STILL PRESENT — intentional; self-consistent; no swap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# ── Constants (mirrored from orbit_wars_original.py) ─────────────────────────
BOARD_SIZE: float = 100.0
CENTER: float = 50.0
SUN_RADIUS: float = 10.0
ROTATION_RADIUS_LIMIT: float = 50.0
COMET_RADIUS: float = 1.0
COMET_PRODUCTION: int = 1
COMET_SPAWN_STEPS: list[int] = [50, 150, 250, 350, 450]


# ── Lightweight planet/fleet structs ─────────────────────────────────────────
# We use plain lists internally (matching source) but provide helper namedtuples
# for readability in higher-level code.
from collections import namedtuple

Planet = namedtuple("Planet", ["id", "owner", "x", "y", "radius", "ships", "production"])
Fleet  = namedtuple("Fleet",  ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"])


@dataclass
class State:
    """Mutable game state for the CWM.

    All list fields are Python lists of plain lists (matching the obs format).
    We avoid numpy arrays here to keep serialisation trivial.
    """
    # Core game data
    planets: list          # list of [id, owner, x, y, radius, ships, production]
    fleets: list           # list of [id, owner, x, y, angle, from_planet_id, ships]
    initial_planets: list  # planet positions at game start (used for orbit calculation)

    # Comet tracking
    comets: list           # list of {planet_ids, paths, path_index}
    comet_planet_ids: list # planet IDs that are comets

    # Turn metadata
    step: int
    next_fleet_id: int
    angular_velocity: float
    num_players: int       # 2 or 4; cached at game start, never changes

    # Optional config fields (needed for cwm_is_terminal and fleet speed)
    episode_steps: int = 500
    ship_speed: float = 6.0
    comet_speed: float = 4.0


def _get(d: Any, key: str, default: Any) -> Any:
    """Get a value from a dict or SimpleNamespace/obs object."""
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


def _detect_num_players(obs: Any, config: Any) -> int:
    """Detect num_players from obs/config.

    Priority:
      1. config.agentCount  (cleanest; requires agent(obs, config) signature)
      2. count distinct non-(-1) owners in planets at game start (step 0)
      3. fallback: 2 (safe default for unknown state)
    """
    # Try config first
    agent_count = _get(config, "agentCount", None) if config is not None else None
    if agent_count is not None:
        return int(agent_count)

    # Fallback: count owners in planets
    planets = _get(obs, "planets", [])
    owners = {p[1] for p in planets if p[1] != -1}
    if len(owners) == 4:
        return 4
    if len(owners) == 2:
        return 2
    # Can't determine → default to 2
    return 2


def state_from_obs(obs: Any, config: Any = None, cached_num_players: int | None = None) -> State:
    """Construct a State from a kaggle_environments observation.

    Parameters
    ----------
    obs:
        The observation dict/SimpleNamespace as received by agent(obs, config).
    config:
        The config dict/SimpleNamespace (may be None). Used for num_players
        and config constants.
    cached_num_players:
        If already determined at game start, pass it here to avoid re-detection.
    """
    if cached_num_players is not None:
        num_players = cached_num_players
    else:
        num_players = _detect_num_players(obs, config)

    # Deep-copy lists so State mutations don't alias obs data
    planets = [list(p) for p in _get(obs, "planets", [])]
    fleets  = [list(f) for f in _get(obs, "fleets", [])]
    initial = [list(p) for p in _get(obs, "initial_planets", [])]

    # Comets: deep-copy dict structure; paths can be shared (read-only)
    raw_comets = _get(obs, "comets", [])
    comets = []
    for g in raw_comets:
        comets.append({
            "planet_ids": list(g["planet_ids"]),
            "paths":      g["paths"],        # read-only; no need to copy
            "path_index": g["path_index"],
        })

    comet_planet_ids = list(_get(obs, "comet_planet_ids", []))
    step             = int(_get(obs, "step", 0))
    next_fleet_id    = int(_get(obs, "next_fleet_id", 0))
    angular_velocity = float(_get(obs, "angular_velocity", 0.025))

    # Config constants
    episode_steps = int(_get(config, "episodeSteps", 500)) if config is not None else 500
    ship_speed    = float(_get(config, "shipSpeed", 6.0))  if config is not None else 6.0
    comet_speed   = float(_get(config, "cometSpeed", 4.0)) if config is not None else 4.0

    return State(
        planets=planets,
        fleets=fleets,
        initial_planets=initial,
        comets=comets,
        comet_planet_ids=comet_planet_ids,
        step=step,
        next_fleet_id=next_fleet_id,
        angular_velocity=angular_velocity,
        num_players=num_players,
        episode_steps=episode_steps,
        ship_speed=ship_speed,
        comet_speed=comet_speed,
    )


def obs_from_state(state: State) -> dict:
    """Convert State back to an obs-compatible dict (for testing transitions)."""
    return {
        "planets":          [list(p) for p in state.planets],
        "fleets":           [list(f) for f in state.fleets],
        "initial_planets":  [list(p) for p in state.initial_planets],
        "comets": [
            {
                "planet_ids": list(g["planet_ids"]),
                "paths":      g["paths"],
                "path_index": g["path_index"],
            }
            for g in state.comets
        ],
        "comet_planet_ids": list(state.comet_planet_ids),
        "step":             state.step,
        "next_fleet_id":    state.next_fleet_id,
        "angular_velocity": state.angular_velocity,
    }


def total_ships(state: State, player_id: int) -> float:
    """Total ships owned by player_id: sum over owned planets + owned fleets."""
    total = 0.0
    for p in state.planets:
        if p[1] == player_id:
            total += p[5]
    for f in state.fleets:
        if f[1] == player_id:
            total += f[6]
    return total


def planet_current_pos(planet_id: int, state: State) -> tuple[float, float]:
    """Current (x, y) of a planet, accounting for orbital rotation.

    Uses initial_planets + angular_velocity * step to compute orbiting position.
    Static planets return their stored position unchanged.
    """
    # Find initial planet entry
    initial = next((p for p in state.initial_planets if p[0] == planet_id), None)
    current = next((p for p in state.planets if p[0] == planet_id), None)
    if current is None:
        raise KeyError(f"Planet {planet_id} not found in state")
    if initial is None:
        return (current[2], current[3])

    dx = initial[2] - CENTER
    dy = initial[3] - CENTER
    r = math.sqrt(dx * dx + dy * dy)
    if r + current[4] < ROTATION_RADIUS_LIMIT:
        init_angle = math.atan2(dy, dx)
        cur_angle = init_angle + state.angular_velocity * state.step
        return (CENTER + r * math.cos(cur_angle), CENTER + r * math.sin(cur_angle))
    return (current[2], current[3])
