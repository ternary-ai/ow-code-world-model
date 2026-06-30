"""
mcts/actions.py — Action abstraction and discretisation for Orbit Wars MCTS.

Representation
==============
An "abstracted action" for one player is a tuple of per-planet decisions,
one entry per active planet (up to n_active_planets):

    abstracted = (
        (from_planet_id_0, target_planet_id_0_or_None, fraction_0),
        (from_planet_id_1, target_planet_id_1_or_None, fraction_1),
        ...
    )

  fraction ∈ {0.0, 0.5, 1.0}
    0.0  → no launch from this planet (target_id is None)
    0.5  → send floor(garrison * 0.5) ships, clamped to [1, garrison]
    1.0  → send all ships

get_action_candidates() returns a list of all such abstractions, built as
the Cartesian product of per-planet choices.  With defaults N=3, K=4:
  per-planet options = 1 (no-op) + K * 2 (K targets × 2 fractions) = 9
  total candidates ≤ 9^3 = 729 — manageable for MCTS to sample from.

abstracted_to_concrete() converts one abstracted action to the submission
format: [[from_planet_id, angle, num_ships], ...].

Angle convention: atan2(target[3] - source[3], target[2] - source[2])
using the CWM's native coordinate system (no swap; see cwm/state.py §X/Y).
Angle is always normalised to [0, 2π).
"""

from __future__ import annotations

import itertools
import math
from typing import Optional

from cwm.state import State, planet_current_pos, CENTER, ROTATION_RADIUS_LIMIT, SUN_RADIUS
from cwm.geometry import fleet_speed
from cwm.masking import sun_blocks_path
from cwm.intercept import solve_intercept
from cwm.action_space import generate_candidates


# Hard cap on the number of abstracted candidates materialised per node. Keeps
# node construction cheap so that search time is governed by the deadline, not
# combinatorial action enumeration. Chosen well above the default product
# (≤ 9**3 = 729) so default behaviour is unchanged.
_MAX_CANDIDATES = 4096

# Safety buffer added to the sun radius when testing whether a launch path
# crosses the sun. The engine destroys any fleet whose swept segment passes
# strictly within SUN_RADIUS of the centre; we keep a small margin so leading
# error or mid-flight drift does not feed ships into the sun.
_SUN_AVOID_RADIUS = SUN_RADIUS + 1.5

# Number of fixed-point iterations used to solve the lead/interception angle
# for a moving (orbiting) target. Converges in 2–3 iterations in practice.
_INTERCEPT_ITERS = 5



# ── Internal helpers ───────────────────────────────────────────────────────────

def _current_pos(planet: list, state: State) -> tuple[float, float]:
    """Current (x, y) of a planet, accounting for orbital rotation."""
    return planet_current_pos(planet[0], state)


def _initial_map(state: State) -> dict:
    """Map planet_id -> initial_planets entry (for orbital prediction)."""
    return {p[0]: p for p in state.initial_planets}


def _predict_pos_dict(planet: list, initial_map: dict, av: float,
                 future_step: float) -> tuple[float, float]:
    """Predicted (x, y) of *planet* at absolute game step *future_step*.

    Mirrors the engine's orbital update: orbiting planets advance their angle by
    angular_velocity per tick; static planets stay put. *future_step* may be
    fractional (used by the interception solver). Static / non-orbiting planets
    return their current stored position unchanged, so for them this reduces to
    the original aim-at-current-position behaviour.
    """
    init = initial_map.get(planet[0])
    if init is None:
        return (planet[2], planet[3])
    dx = init[2] - CENTER
    dy = init[3] - CENTER
    r = math.sqrt(dx * dx + dy * dy)
    if r + planet[4] < ROTATION_RADIUS_LIMIT:
        init_angle = math.atan2(dy, dx)
        ang = init_angle + av * future_step
        return (CENTER + r * math.cos(ang), CENTER + r * math.sin(ang))
    return (planet[2], planet[3])


def _intercept_aim(src_pos: tuple, target: list, initial_map: dict, av: float,
                   base_step: float, speed: float) -> tuple[float, float]:
    """Fixed-point lead solver: predicted intercept point for a moving target.

    A fleet leaves *src_pos* travelling in a straight line at *speed* units/tick.
    Because orbiting targets move while the fleet is in transit, aiming at the
    target's current position makes the fleet sail past where the planet *was*
    and run off the board. We iterate:
        t ← distance(src, aim) / speed     (ticks to reach the current guess)
        aim ← predicted target position at (base_step + t)
    until the aim point converges. For a static target the first prediction is
    already exact, so this returns its current position (no behaviour change).
    """
    aim = _predict_pos_dict(target, initial_map, av, base_step)
    if speed <= 0.0:
        return aim
    for _ in range(_INTERCEPT_ITERS):
        d = math.hypot(aim[0] - src_pos[0], aim[1] - src_pos[1])
        t_ticks = d / speed
        new_aim = _predict_pos_dict(target, initial_map, av, base_step + t_ticks)
        if (abs(new_aim[0] - aim[0]) < 1e-3
                and abs(new_aim[1] - aim[1]) < 1e-3):
            return new_aim
        aim = new_aim
    return aim


def _path_crosses_sun(src_pos: tuple, aim_pos: tuple) -> bool:
    """True iff the straight launch path src→aim passes through the sun disc
    (within _SUN_AVOID_RADIUS of the centre) and would be destroyed in transit.
    """
    return sun_blocks_path(src_pos, aim_pos, (CENTER, CENTER), _SUN_AVOID_RADIUS)


# ── Right-sizing (fleet economy) ────────────────────────────────────────────────
#
# The single biggest quality gap between this MCTS and the proven greedy
# "Orbital Flanker" agent was fleet economy: fixed-fraction launches (0.5 / 1.0)
# systematically OVER-send to weak targets (wasting ships that could capture a
# second planet) and UNDER-send to growing enemies (the attack bounces off a
# garrison that out-grew the in-flight fleet). The helpers below let the action
# layer offer a precisely-sized "fit" launch — exactly the ships needed to take
# the target on arrival — so MCTS searches over economical, non-redundant moves.

# Sentinel fraction value meaning "send exactly the ships needed to capture".
_RIGHT_SIZE_FIT = "fit"

# Margin added on top of an enemy planet's projected garrison so the capture
# clears with a small buffer (mirrors the Flanker's ENEMY_MARGIN default).
_ENEMY_MARGIN = 2

# Enemy garrisons grow by production each tick in transit; we project that
# growth but cap the look-ahead so a far target does not demand an absurd force.
_ENEMY_GROWTH_CAP_ETA = 40.0
_ENEMY_GROWTH_RATE = 0.6

# Angular tolerance (radians) for deciding a fleet is "heading toward" a planet
# when attributing in-flight ships to a target (contest / reserved detection).
_CONTEST_ANGLE = 0.15


def _current_pos_map(state: State) -> dict:
    """planet_id -> current (x, y), accounting for orbital rotation."""
    return {p[0]: planet_current_pos(p[0], state) for p in state.planets}


def _cached_pos_and_inbound(state: State) -> tuple[dict, dict]:
    """Return (pos_map, inbound) for *state*, memoised on the state object.

    Both are player-independent and depend only on the current planet positions
    and in-flight fleets. Inside MCTS, a parent node is expanded into many child
    actions, each calling abstracted_to_concrete on the SAME parent state; this
    cache turns that O(children) recomputation of the O(fleets×planets) ledger
    into a single computation per state. The cache lives in a plain attribute
    that _copy_state never propagates, so child states recompute correctly.
    """
    cache = getattr(state, "_rs_cache", None)
    if cache is not None:
        return cache
    pos_map = _current_pos_map(state)
    inbound = _build_inbound(state, pos_map)
    try:
        state._rs_cache = (pos_map, inbound)
    except (AttributeError, TypeError):
        pass                                  # slotted/frozen state — skip cache
    return pos_map, inbound


def _build_inbound(state: State, pos_map: dict) -> dict:
    """Map (fleet_owner, target_planet_id) -> total in-flight ships heading there.

    Each in-flight fleet is attributed to the single planet its heading best
    points at (within _CONTEST_ANGLE), mirroring the Flanker's reserved-target
    logic. Used to right-size launches: don't re-send ships a friendly fleet is
    already delivering (dedup / contest), and fold an enemy's inbound
    reinforcements into the force needed to take its planet.
    """
    inbound: dict = {}
    for f in state.fleets:
        fo, fx, fy, fang, src_pid, fships = f[1], f[2], f[3], f[4], f[5], f[6]
        best_id, best_diff = None, _CONTEST_ANGLE
        for p in state.planets:
            if p[0] == src_pid:
                continue
            px, py = pos_map[p[0]]
            bearing = math.atan2(py - fy, px - fx)
            diff = abs(math.atan2(math.sin(bearing - fang),
                                  math.cos(bearing - fang)))
            if diff < best_diff:
                best_diff, best_id = diff, p[0]
        if best_id is not None:
            key = (fo, best_id)
            inbound[key] = inbound.get(key, 0) + int(fships)
    return inbound


def _ships_needed(target: list, player_id: int, eta: float,
                  friendly_inbound: int, enemy_inbound: int = 0) -> int:
    """Minimum ships to capture/hold *target* on arrival, net of ships already
    inbound from friendly fleets.

    Neutral  : garrison + 1            (static garrison, no growth)
    Owned    : threat-aware defence — if enemy ships are inbound and would
               overwhelm the garrison (+ friendly reinforcements already in
               flight), send exactly the deficit + margin to survive; an owned
               planet that is safe needs no reinforcement (returns <= 0, so the
               ships stay free for offence — a key Flanker economy).
    Enemy    : projected_garrison + margin, capped at +50% over current,
               where projected_garrison = garrison + prod * min(eta, 40) * 0.6.

    The returned value may be <= 0, which signals the target is already
    sufficiently covered (friendly inbound, or — for owned planets — no live
    threat). The caller treats this as "no launch needed" (automatic dedup).
    """
    garrison = target[5]
    prod = target[6]
    owner = target[1]
    if owner == -1:
        needed = garrison + 1
    elif owner == player_id:
        # Defend only against a real, overwhelming threat. The garrison plus the
        # friendly reinforcements already inbound must cover the incoming enemy
        # ships; the shortfall (plus a margin) is what this launch must add.
        shortfall = enemy_inbound - (garrison + friendly_inbound)
        if shortfall <= 0:
            return 0                       # safe — keep ships free for offence
        return shortfall + _ENEMY_MARGIN
    else:
        eff_eta = min(eta, _ENEMY_GROWTH_CAP_ETA)
        projected = garrison + int(prod * eff_eta * _ENEMY_GROWTH_RATE)
        needed = projected + _ENEMY_MARGIN
        cap = garrison + max(_ENEMY_MARGIN, int(garrison * 0.5))
        needed = min(needed, cap)
    return needed - int(friendly_inbound)


def _prod_dist_score(target_planet: list, source_pos: tuple,
                     state: State, weakness: float = 0.0) -> float:
    """Score for target priority.

    Base score = production / (distance + ε): higher production and closer
    targets rank higher.

    *weakness* (∈ [0, 1]) folds the target's garrison into the score so the
    agent prefers weakly-defended planets it can actually capture (Strategic
    Principles 3 & 4 — attack weakness, preserve fleet efficiency):
        score = production / ((distance + ε) * (1 + weakness * garrison))
    weakness = 0.0 reproduces the original garrison-blind ranking exactly.
    """
    pos = _current_pos(target_planet, state)
    dx = pos[0] - source_pos[0]
    dy = pos[1] - source_pos[1]
    dist = math.sqrt(dx * dx + dy * dy) + 1e-6
    garrison = target_planet[5]
    return target_planet[6] / (dist * (1.0 + weakness * garrison))


def _angle_to(source_pos: tuple, target_pos: tuple) -> float:
    """Direction angle from source to target, in [0, 2π)."""
    angle = math.atan2(
        target_pos[1] - source_pos[1],
        target_pos[0] - source_pos[0],
    )
    if angle < 0.0:
        angle += 2.0 * math.pi
    return angle


# ── Main functions ─────────────────────────────────────────────────────────────

def get_action_candidates(
    state: State,
    player_id: int,
    k_targets: int = 4,
    n_active_planets: int = 3,
    k_reinforce: int = 0,
    fractions: tuple = (0.5, 1.0),
    target_weakness: float = 0.0,
    right_size: bool = False,
) -> list:
    """Return a list of all discretised player actions.

    Per owned planet (capped to top-n_active_planets by garrison):
      - Offensive targets: top-k_targets non-owned planets ranked by a
        weakness-aware production / distance score (current positions at
        decision time). *target_weakness* (∈ [0, 1], default 0) folds the
        target garrison in to prefer weakly-defended capturable planets; 0
        reproduces the original garrison-blind ranking.
      - Defensive reinforcement targets (when k_reinforce > 0 and enemies
        exist): up to k_reinforce OTHER owned planets, ranked by proximity to
        the nearest enemy planet (frontline planets first). This lets the agent
        consolidate ships onto threatened planets — a capability the greedy
        baseline lacks.
      - Ship fraction choices: {0 (no-op)} ∪ *fractions*.

    Returns a list of abstracted actions.  Each abstracted action is a
    flat tuple of (from_planet_id, target_planet_id_or_None, fraction)
    entries.  In right_size mode each source also has a "multi-spray"
    option that bundles all k_targets fit-launches into one entry,
    enabling simultaneous multi-front pressure without blowing up the
    action-space size (it counts as one Cartesian product slot).

    Always includes at least the all-no-op action (empty launches).

    Backward compatibility: with the defaults (k_reinforce=0,
    fractions=(0.5, 1.0)) the output is identical to the original 2-feature
    behaviour, so existing tuned weights are unaffected.
    """
    # Identify active own planets (owned + ships > 0), sorted by garrison desc
    own_planets = [
        p for p in state.planets
        if p[1] == player_id and p[5] > 0
    ]
    own_planets.sort(key=lambda p: p[5], reverse=True)
    active = own_planets[:n_active_planets]

    if not active:
        # No planets: single no-op action
        return [()]

    # All non-owned planets as potential offensive targets
    non_own = [p for p in state.planets if p[1] != player_id]

    # Enemy planets (owned by another player; neutrals are not threats) drive
    # the reinforcement ranking. All owned planets are reinforcement candidates.
    enemies = [p for p in state.planets if p[1] >= 0 and p[1] != player_id]
    all_own = [p for p in state.planets if p[1] == player_id]
    reinforce_enabled = k_reinforce > 0 and bool(enemies) and len(all_own) > 1

    def _threat(planet: list) -> float:
        """Negative distance to nearest enemy — higher = more threatened."""
        best = min(
            (planet[2] - e[2]) ** 2 + (planet[3] - e[3]) ** 2
            for e in enemies
        )
        return -best

    # Build per-planet option lists
    initial_map = _initial_map(state)
    av = state.angular_velocity
    ship_speed = state.ship_speed
    per_planet_options: list[list] = []
    for src in active:
        src_pos = _current_pos(src, state)

        # Use cwm.action_space.generate_candidates (Module 3) to find reachable
        # targets: it calls solve_intercept (Module 2) for convergence and
        # sun_blocks_path (Module 4) for occlusion, filtering any target whose
        # straight-line path is destroyed by the sun.  Use a nominal half-garrison
        # ship count so the filter is representative of actual sends.
        nominal_ships = max(1, src[5] // 2)
        _cwm_cands = generate_candidates(state, src[0], [nominal_ships])
        _reachable_ids = {c.target_planet_id for c in _cwm_cands}
        reachable = [t for t in non_own if t[0] in _reachable_ids]

        # Rank reachable non-owned planets by weakness-aware production/distance
        # score from this source (favours weakly-defended rich close planets
        # when target_weakness > 0).
        ranked = sorted(
            reachable,
            key=lambda t: _prod_dist_score(t, src_pos, state, target_weakness),
            reverse=True,
        )
        targets = ranked[:k_targets]

        # Reinforcement targets: most-threatened OTHER owned planets.
        if reinforce_enabled:
            reinforce_pool = sorted(
                (p for p in all_own if p[0] != src[0]),
                key=_threat,
                reverse=True,
            )
            targets = targets + reinforce_pool[:k_reinforce]

        # Each option is a tuple of one or more (from, target, fraction) entries
        # so that multi-launch "spray" options can bundle several entries.
        # abstracted_to_concrete already handles multiple same-source entries
        # via its committed dict — no changes needed there.
        options = [((src[0], None, 0.0),)]   # no-op
        if right_size:
            # Precisely-sized "fit" launch per target (capture economy). For
            # enemy-owned targets also offer an all-in (1.0) elimination push,
            # which the over-send guard in fit deliberately excludes. Neutrals
            # get only the fit option — over-garrisoning a neutral wastes ships.
            for tgt in targets:
                options.append(((src[0], tgt[0], _RIGHT_SIZE_FIT),))
                if tgt[1] >= 0 and tgt[1] != player_id:
                    if src[5] > 0:
                        options.append(((src[0], tgt[0], 1.0),))
            # Multi-spray: launch right-sized fleets to ALL k_targets at once.
            # Counts as one option in the Cartesian product so it doesn't
            # multiply the action space — each source can either pick a single
            # target OR do the full simultaneous spray.
            if len(targets) > 1:
                spray = tuple((src[0], tgt[0], _RIGHT_SIZE_FIT) for tgt in targets)
                options.append(spray)
        else:
            for tgt in targets:
                for frac in fractions:
                    ships = int(src[5] * frac)
                    if ships > 0:
                        options.append(((src[0], tgt[0], frac),))

        per_planet_options.append(options)

    # Bound the Cartesian product. With rich tunable settings (large k_targets,
    # k_reinforce, n_active_planets, fine fractions) the full product can reach
    # hundreds of thousands of combinations, which makes node construction —
    # not the time-bounded simulation loop — the dominant cost and blows the
    # per-turn budget. Trim the lowest-priority option from the widest planet
    # repeatedly until the product fits _MAX_CANDIDATES. Per-planet options are
    # appended in priority order (no-op first, best targets first), so trimming
    # from the end discards the least promising launches while always keeping
    # the no-op. Default settings (≤ 9**3 = 729) are never trimmed.
    def _product_size(opts: list[list]) -> int:
        size = 1
        for o in opts:
            size *= len(o)
        return size

    while _product_size(per_planet_options) > _MAX_CANDIDATES:
        widest = max(per_planet_options, key=len)
        if len(widest) <= 1:
            break
        widest.pop()

    # Cartesian product across all active planets; each per-planet option is a
    # tuple of (from, target, frac) entries, so flatten by concatenation.
    candidates = [sum(combo, ()) for combo in itertools.product(*per_planet_options)]
    return candidates



def abstracted_to_concrete(
    state: State,
    player_id: int,
    abstracted: tuple,
) -> list:
    """Convert one abstracted action to the submission format.

    Parameters
    ----------
    state : State
        Current game state (for current planet positions and garrison values).
    player_id : int
        The acting player.
    abstracted : tuple
        Tuple of (from_planet_id, target_planet_id_or_None, fraction) entries
        as returned by get_action_candidates().

    Returns
    -------
    list of [from_planet_id, angle_radians, num_ships]
        Ready to pass to cwm_apply_joint_action.  Empty list if nothing to launch.

    Guarantees (validated here as defensive depth):
      - 0 < num_ships <= current garrison
      - from_planet is owned by player_id
      - angle ∈ [0, 2π)
    """
    planet_map = {p[0]: p for p in state.planets}
    initial_map = _initial_map(state)
    av = state.angular_velocity
    ship_speed = state.ship_speed
    # Current positions and the in-flight inbound ledger are needed to size
    # "fit" launches: the precise force to take a target accounts for the
    # target's current position (for ETA) and any ships friendly fleets are
    # already delivering (dedup). Memoised on the state so a parent node
    # expanded into many child actions computes the O(fleets×planets) ledger
    # only once instead of once per child.
    pos_map, inbound = _cached_pos_and_inbound(state)
    # Enemy ships inbound to each planet (summed across all hostile owners),
    # used to size threat-aware defensive reinforcement of our own planets.
    enemy_inbound_to: dict = {}
    for (fo, tid), ships in inbound.items():
        if fo != player_id and fo != -1:
            enemy_inbound_to[tid] = enemy_inbound_to.get(tid, 0) + ships
    # Endgame banking: the game is scored (ships on planets + ships in transit)
    # at the terminal step (episode_steps - 2). In-flight fleets count exactly
    # the same as garrisoned ships, so a fleet that cannot ARRIVE before scoring
    # has zero capture upside and only risks dying in transit (sun / board edge
    # / comet). We therefore suppress any launch whose estimated arrival step is
    # past the terminal step. This is self-gating: for most of the game arrival
    # lands far before the terminus, so the rule has no effect until the endgame.
    terminal_step = (state.episode_steps if state.episode_steps > 0 else 500) - 2

    moves = []
    # Track how many ships have already been committed per source planet
    # (to handle multiple launches from the same planet correctly)
    committed: dict[int, int] = {}
    # Track ships this action already commits toward each target, so a second
    # "fit" entry aimed at the same planet does not re-send what the first one
    # already covers (intra-action dedup).
    committed_to_target: dict[int, int] = {}

    for from_id, target_id, fraction in abstracted:
        if fraction == 0.0 or target_id is None:
            continue

        src = planet_map.get(from_id)
        if src is None:
            continue
        if src[1] != player_id:
            continue

        already_committed = committed.get(from_id, 0)
        available = src[5] - already_committed
        if available <= 0:
            continue

        tgt = planet_map.get(target_id)
        if tgt is None:
            continue

        src_pos = pos_map.get(from_id) or _current_pos(src, state)

        if fraction == _RIGHT_SIZE_FIT:
            # Precisely-sized capture: estimate ETA with a nominal speed (full
            # available garrison), then ask for exactly the ships needed net of
            # friendly ships already inbound (existing fleets + this action's
            # prior launches at the same target).
            tgt_pos = pos_map.get(target_id, (tgt[2], tgt[3]))
            nominal_speed = fleet_speed(available, ship_speed)
            approx_dist = math.hypot(tgt_pos[0] - src_pos[0],
                                     tgt_pos[1] - src_pos[1])
            eta = approx_dist / nominal_speed if nominal_speed > 0.0 else approx_dist
            already_inbound = (inbound.get((player_id, target_id), 0)
                               + committed_to_target.get(target_id, 0))
            needed = _ships_needed(tgt, player_id, eta, already_inbound,
                                   enemy_inbound_to.get(target_id, 0))
            if needed <= 0:
                continue                       # target already covered — dedup
            if needed > available:
                continue                       # cannot fully capture — skip fit
            ships = needed
        else:
            ships = int(available * fraction)
        ships = max(0, min(ships, available))   # clamp defensively
        if ships <= 0:
            continue

        # Compute exact intercept angle via solve_intercept (Module 2): leads
        # the (possibly orbiting) target to its future position and returns the
        # firing angle in [0, 2π).  Returns None when no convergent solution
        # exists (e.g. sun-blocked in a degenerate configuration).
        speed = fleet_speed(ships, ship_speed)
        angle = solve_intercept(src_pos, tgt[0], ships, state.step, state)
        if angle is None:
            continue

        # Reconstruct approximate aim point from one step of the intercept
        # iteration — needed for the sun check and ETA estimate.  For static
        # targets this is exact; for orbiting targets it is a good first-order
        # approximation sufficient for both checks.
        _tgt_cur = pos_map.get(target_id, (tgt[2], tgt[3]))
        _d_cur = math.hypot(_tgt_cur[0] - src_pos[0], _tgt_cur[1] - src_pos[1])
        _eta_est = _d_cur / speed if speed > 0.0 else 0.0
        _aim = _predict_pos_dict(tgt, initial_map, av, state.step + _eta_est)

        # Final sun-safety check using the reconstructed aim point (Module 4).
        if _path_crosses_sun(src_pos, _aim):
            continue

        # Endgame banking: skip launches that cannot arrive before scoring.
        dist = math.hypot(_aim[0] - src_pos[0], _aim[1] - src_pos[1])
        if speed > 0.0 and state.step + dist / speed > terminal_step:
            continue

        moves.append([from_id, angle, ships])
        committed[from_id] = already_committed + ships
        committed_to_target[target_id] = (
            committed_to_target.get(target_id, 0) + ships
        )

    return moves
