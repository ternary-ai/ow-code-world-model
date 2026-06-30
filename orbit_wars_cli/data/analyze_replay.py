"""
data/analyze_replay.py — Post-mortem analyzer for an Orbit Wars Kaggle replay.

Reads a Kaggle episode replay JSON (the big `steps`/`configuration` file the
arena produces) and prints a COMPACT digest — it never dumps the raw replay, so
it is safe to run on multi-megabyte files without blowing up a chat context.

What it answers
---------------
The headline question this tool was built for is *"my agent sends ships but they
never arrive — why?"*. To answer it the analyzer reconstructs the lifecycle of
every fleet from the per-step observations (which are Kaggle's ground truth, not
our world model) and classifies how each fleet ended:

  ARRIVED        last seen on/adjacent to a planet  → delivered its ships
  SUN-DESTROYED  last seen inside/under the sun disc → crossed the sun and died
  OOB-LOST       last seen at the board edge         → ran off the board
  VANISHED       disappeared mid-board for no visible reason (engine/edge case)
  IN-FLIGHT      still travelling when the game ended

It also cross-checks ACTIONS against fleet creation: every launch order should
spawn exactly one fleet next step. Orders that spawn nothing are reported as
REJECTED (illegal / malformed / insufficient garrison), which is the other
common cause of "ships that never arrive".

Per-player it reports ships ordered, ships actually launched, ships delivered to
combat, and ships wasted (sun / OOB / vanished), plus a timeout check from the
`remainingOverageTime` trace.

Usage
-----
    python data/analyze_replay.py REPLAY.json [--seat N] [--max-examples K]

  --seat N         focus the per-fleet example listing on player N (default: all)
  --max-examples K examples of wasted fleets to print per category (default: 8)

Replay format assumptions (Kaggle kaggle_environments Orbit Wars)
-----------------------------------------------------------------
  replay["steps"]            : list of timesteps; each is a list of per-agent
                               dicts {observation, action, reward, status, info}.
  observation["planets"]     : [id, owner, x, y, radius, ships, production]
  observation["fleets"]      : [id, owner, x, y, angle, from_planet_id, ships]
  Player 0's observation carries the full global planet/fleet lists, so it is
  used as the authoritative world state each step.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

# Geometry (mirrors cwm/state.py; kept local so the tool is import-light).
BOARD_SIZE = 100.0
CENTER     = 50.0
SUN_RADIUS = 10.0
SHIP_SPEED = 6.0      # default configuration.shipSpeed (max fleet speed)

# Classification thresholds.
_SUN_MARGIN  = 2.0    # within SUN_RADIUS + margin of centre ⇒ counted as sun death
_EDGE_MARGIN = 2.0    # within margin of any board edge ⇒ counted as out-of-bounds
_ARRIVE_PAD  = 2.0    # within planet_radius + pad of a planet ⇒ counted as arrival


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def _fleet_speed(ships: int, max_speed: float = SHIP_SPEED) -> float:
    """Mirror of cwm.geometry.fleet_speed: bigger fleets fly faster, capped."""
    n = max(1, int(ships))
    if n <= 1:
        return 1.0
    frac = (math.log(n) / math.log(1000.0)) ** 1.5
    return min(max_speed, 1.0 + (max_speed - 1.0) * frac)


def _authoritative_obs(step: list) -> dict:
    """Return the per-step observation that carries the global planet/fleet lists.

    Player 0's observation always includes the full lists; fall back to any agent
    that has a non-empty planet list.
    """
    if step and "observation" in step[0] and step[0]["observation"].get("planets"):
        return step[0]["observation"]
    for agent in step:
        obs = agent.get("observation", {})
        if obs.get("planets"):
            return obs
    return step[0].get("observation", {}) if step else {}


def _nearest_planet(x: float, y: float, planets: list) -> tuple:
    """Return (planet, distance_to_surface) of the closest planet to (x, y)."""
    best = None
    best_d = math.inf
    for p in planets:
        d = _dist(x, y, p[2], p[3]) - p[4]   # distance to planet surface
        if d < best_d:
            best_d = d
            best = p
    return best, best_d


def _classify_end(last_fleet: list, planets: list,
                  vel: tuple | None = None) -> str:
    """Classify a fleet's fate from its last observed position.

    Because fleets are only observed at integer steps but move up to ~ship_speed
    units per tick, a fleet's last-seen position can be nearly a full step short
    of the planet it reaches on the next (unobserved) tick. To avoid mislabelling
    such genuine arrivals as VANISHED, we also test the one-step forward
    projection of the fleet along its heading (*vel*). The same projection makes
    OOB detection match the engine, which kills a fleet on the tick it would
    leave the board.
    """
    x, y = last_fleet[2], last_fleet[3]

    # One-step forward projection (observed velocity if available, else derive
    # speed from the ship count and heading angle).
    if vel is not None:
        px, py = x + vel[0], y + vel[1]
    else:
        ships = last_fleet[6] if len(last_fleet) > 6 else 1
        spd = _fleet_speed(ships)
        ang = last_fleet[4]
        px, py = x + spd * math.cos(ang), y + spd * math.sin(ang)

    # Sun death: crossing the central sun disc destroys a fleet.
    if (_dist(x, y, CENTER, CENTER) <= SUN_RADIUS + _SUN_MARGIN
            or _dist(px, py, CENTER, CENTER) <= SUN_RADIUS + _SUN_MARGIN):
        return "SUN-DESTROYED"

    # Arrival: sitting on/adjacent to a planet surface — test both the last seen
    # position and the one-step projection (catches arrivals the sampler missed).
    planet, d_surf = _nearest_planet(x, y, planets)
    if planet is not None and d_surf <= _ARRIVE_PAD:
        return "ARRIVED"
    planet_p, d_surf_p = _nearest_planet(px, py, planets)
    if planet_p is not None and d_surf_p <= _ARRIVE_PAD:
        return "ARRIVED"

    # Out of bounds: ran into a board edge (now, or on the next tick).
    if (x <= _EDGE_MARGIN or x >= BOARD_SIZE - _EDGE_MARGIN
            or y <= _EDGE_MARGIN or y >= BOARD_SIZE - _EDGE_MARGIN
            or px <= 0.0 or px >= BOARD_SIZE
            or py <= 0.0 or py >= BOARD_SIZE):
        return "OOB-LOST"

    # Disappeared mid-board with no planet nearby — anomalous.
    return "VANISHED"


def _observed_velocity(rec: dict) -> tuple | None:
    """Per-step velocity (vx, vy) inferred from a fleet's first/last observations.

    Returns None when the fleet was seen only once (no displacement available);
    callers then fall back to the heading-angle + ship-speed estimate.
    """
    dt = rec.get("last_step", 0) - rec.get("birth", 0)
    if dt <= 0:
        return None
    first, last = rec["first"], rec["last"]
    return ((last[2] - first[2]) / dt, (last[3] - first[3]) / dt)


def analyze(path: str, seat: int | None, max_examples: int) -> int:
    with open(path) as fh:
        replay = json.load(fh)

    steps = replay.get("steps", [])
    if not steps:
        print("No steps in replay.", file=sys.stderr)
        return 1

    n_agents   = len(steps[0])
    team_names = replay.get("info", {}).get("TeamNames", [])
    rewards    = replay.get("rewards", [])
    statuses   = replay.get("statuses", [])
    cfg        = replay.get("configuration", {})

    # ── Fleet lifecycle tracking ─────────────────────────────────────────────
    # fleet_id -> {owner, birth, first, last, last_step}
    live: dict[int, dict] = {}
    ended: list[dict] = []        # finalized fleet records
    planets_at: dict[int, list] = {}   # step -> planet list (for end classification)

    # ── Action ↔ fleet-spawn cross-check ─────────────────────────────────────
    ordered_ships  = [0] * n_agents   # ships requested in actions
    ordered_orders = [0] * n_agents   # number of launch orders
    rejected_orders = [0] * n_agents  # orders that produced no fleet

    prev_fleet_ids: set[int] = set()
    # Per-step record of (seat -> count of launch orders issued at that step).
    orders_by_step: list[list[int]] = []

    for si, step in enumerate(steps):
        obs = _authoritative_obs(step)
        planets = obs.get("planets", [])
        fleets  = obs.get("fleets", [])
        planets_at[si] = planets

        cur_ids = set()
        new_owner: dict[int, int] = {}   # fleet_id -> owner, for fleets seen now
        for f in fleets:
            fid = f[0]
            cur_ids.add(fid)
            new_owner[fid] = f[1]
            rec = live.get(fid)
            if rec is None:
                live[fid] = {
                    "owner": f[1], "birth": si, "first": f, "last": f,
                    "last_step": si,
                }
            else:
                rec["last"] = f
                rec["last_step"] = si

        # Fleets that disappeared this step → finalize them.
        for fid in prev_fleet_ids - cur_ids:
            rec = live.pop(fid, None)
            if rec is not None:
                rec["fate"] = _classify_end(
                    rec["last"], planets_at.get(si, planets),
                    _observed_velocity(rec),
                )
                ended.append(rec)

        # New fleet ids that appeared at THIS step, bucketed by owner seat.
        new_ids = cur_ids - prev_fleet_ids
        new_by_seat = [0] * n_agents
        for fid in new_ids:
            owner = new_owner.get(fid, -1)
            if 0 <= owner < n_agents:
                new_by_seat[owner] += 1

        # Count launch orders issued at this step (the action that should spawn
        # those new fleets on the NEXT step).
        step_orders = [0] * n_agents
        for ai, agent in enumerate(step):
            action = agent.get("action") or []
            for mv in action:
                if isinstance(mv, (list, tuple)) and len(mv) >= 3:
                    ordered_orders[ai] += 1
                    ordered_ships[ai]  += mv[2]
                    step_orders[ai]    += 1
        orders_by_step.append(step_orders)

        # Reconcile: orders issued at step si-1 should equal new fleets at si.
        if si > 0:
            for ai in range(n_agents):
                rej = orders_by_step[si - 1][ai] - new_by_seat[ai]
                if rej > 0:
                    rejected_orders[ai] += rej

        prev_fleet_ids = cur_ids

    # Finalize fleets still in flight at game end.
    last_planets = planets_at.get(len(steps) - 1, [])
    for fid, rec in live.items():
        rec["fate"] = "IN-FLIGHT"
        ended.append(rec)

    # ── Aggregate ────────────────────────────────────────────────────────────
    fates = ["ARRIVED", "SUN-DESTROYED", "OOB-LOST", "VANISHED", "IN-FLIGHT"]
    per_seat = {ai: {fate: 0 for fate in fates} for ai in range(n_agents)}
    per_seat_ships = {ai: {fate: 0 for fate in fates} for ai in range(n_agents)}
    for rec in ended:
        owner = rec["owner"]
        if owner not in per_seat:
            continue
        fate = rec["fate"]
        ships = rec["first"][6]
        per_seat[owner][fate] += 1
        per_seat_ships[owner][fate] += ships

    # ── Report ───────────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"REPLAY: {path}")
    print(f"  episode={replay.get('id')}  agents={n_agents}  "
          f"steps_played={len(steps)} / {cfg.get('episodeSteps')}")
    print(f"  teams={team_names}  rewards={rewards}  statuses={statuses}")
    print("=" * 70)

    # Timeout check.
    print("\n--- TIMEOUT CHECK (remainingOverageTime) ---")
    print(f"  actTimeout={cfg.get('actTimeout')}s  "
          f"agentTimeout={cfg.get('agentTimeout')}s  "
          f"runTimeout={cfg.get('runTimeout')}s")
    for ai in range(n_agents):
        start = steps[0][ai]["observation"].get("remainingOverageTime")
        endo = None
        for s in reversed(steps):
            v = s[ai]["observation"].get("remainingOverageTime")
            if v is not None:
                endo = v
                break
        used = (start - endo) if (start is not None and endo is not None) else None
        flag = ""
        if used is not None and start:
            if endo <= 0.5:
                flag = "  <<< OVERAGE EXHAUSTED (likely timeouts!)"
            elif used > start * 0.5:
                flag = "  <<< heavy overage use"
        print(f"  seat {ai}: start={start}  end={endo}  used={used}{flag}")

    # Action → spawn integrity.
    print("\n--- ORDER / LAUNCH INTEGRITY ---")
    for ai in range(n_agents):
        rej = rejected_orders[ai]
        tot = ordered_orders[ai]
        pct = (100.0 * rej / tot) if tot else 0.0
        flag = "  <<< many rejected orders!" if pct >= 10.0 else ""
        print(f"  seat {ai}: launch_orders={tot}  ships_ordered={ordered_ships[ai]}  "
              f"rejected≈{rej} ({pct:.1f}%){flag}")

    # Fleet fates.
    print("\n--- FLEET FATES (count / ships) ---")
    header = "  seat  " + "".join(f"{f:>16}" for f in fates)
    print(header)
    for ai in range(n_agents):
        row = f"  {ai:<5} "
        for fate in fates:
            row += f"{per_seat[ai][fate]:>6}/{per_seat_ships[ai][fate]:<9}"
        print(row)

    # Waste summary (the headline).
    print("\n--- WASTE SUMMARY (ships that never reached combat) ---")
    for ai in range(n_agents):
        launched = sum(per_seat_ships[ai][f] for f in fates)
        arrived  = per_seat_ships[ai]["ARRIVED"]
        sun      = per_seat_ships[ai]["SUN-DESTROYED"]
        oob      = per_seat_ships[ai]["OOB-LOST"]
        vanished = per_seat_ships[ai]["VANISHED"]
        inflight = per_seat_ships[ai]["IN-FLIGHT"]
        wasted   = sun + oob + vanished
        denom    = launched - inflight
        wpct     = (100.0 * wasted / denom) if denom else 0.0
        flag = "  <<< MAJOR WASTE" if wpct >= 25.0 else ""
        print(f"  seat {ai}: launched={launched}  arrived={arrived}  "
              f"WASTED={wasted} (sun={sun} oob={oob} vanished={vanished}) "
              f"= {wpct:.1f}% of resolved{flag}")

    # Examples of wasted fleets (to eyeball trajectories).
    cats = ["SUN-DESTROYED", "OOB-LOST", "VANISHED"]
    print(f"\n--- EXAMPLES (up to {max_examples} per category"
          + (f", seat {seat}" if seat is not None else "") + ") ---")
    for cat in cats:
        shown = 0
        for rec in ended:
            if rec["fate"] != cat:
                continue
            if seat is not None and rec["owner"] != seat:
                continue
            if shown >= max_examples:
                break
            f0 = rec["first"]
            fl = rec["last"]
            print(f"  [{cat}] fleet#{f0[0]} seat{rec['owner']} "
                  f"from_planet={f0[5]} ships={f0[6]} "
                  f"birth_step={rec['birth']} died_step={rec['last_step']+1} "
                  f"start=({f0[2]:.1f},{f0[3]:.1f}) "
                  f"last=({fl[2]:.1f},{fl[3]:.1f}) angle={f0[4]:.3f}")
            shown += 1
        if shown == 0:
            print(f"  [{cat}] none")

    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Analyze an Orbit Wars Kaggle replay JSON.")
    ap.add_argument("replay", help="Path to the replay JSON file.")
    ap.add_argument("--seat", type=int, default=None,
                    help="Focus fleet examples on this player seat.")
    ap.add_argument("--max-examples", type=int, default=8,
                    help="Examples of wasted fleets to print per category.")
    args = ap.parse_args(argv)
    return analyze(args.replay, args.seat, args.max_examples)


if __name__ == "__main__":
    raise SystemExit(main())
