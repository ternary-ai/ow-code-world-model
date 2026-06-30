"""
cwm/event_graph.py — Compact state abstraction via fleet arrival events (Module 5).

Replaces raw per-planet coordinates with a fixed-size encoding of upcoming
fleet arrivals, suitable as input to a value function or heuristic.

Fleet destination inference:
  In-flight fleets store their angle and source planet but NOT a destination
  planet ID.  We infer the destination by finding the planet whose bearing
  from the fleet's current position best matches the fleet's angle (within
  a tolerance of 0.15 radians), mirroring the approach in mcts/actions.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cwm.state import State
from cwm.geometry import fleet_speed

_DEST_ANGLE_TOL = 0.15   # radians; fleet is considered "heading toward" a planet
                          # if its angle deviates by less than this from the bearing


@dataclass
class FleetEvent:
    """One in-flight fleet's arrival event."""
    owner: int
    destination_planet_id: int
    eta_turn: int
    ships_arriving: int


def _infer_destination(
    fleet: list,
    planets: list,
) -> tuple[int | None, float]:
    """Return (destination_planet_id, distance) for the best-matching planet.

    Matches the fleet's angle against the bearing to each candidate planet
    (excluding the fleet's source planet).  Returns (None, 0.0) when no
    planet is within _DEST_ANGLE_TOL.
    """
    fid, owner, fx, fy, fangle, from_pid, fships = fleet
    best_pid, best_diff, best_dist = None, _DEST_ANGLE_TOL, 0.0

    for p in planets:
        if p[0] == from_pid:
            continue
        px, py = p[2], p[3]
        bearing = math.atan2(py - fy, px - fx)
        diff = abs(math.atan2(math.sin(bearing - fangle),
                              math.cos(bearing - fangle)))
        if diff < best_diff:
            best_diff = diff
            best_pid = p[0]
            best_dist = math.hypot(px - fx, py - fy)

    return best_pid, best_dist


def extract_events(state: State, horizon: int) -> list[FleetEvent]:
    """Return FleetEvents for all in-flight fleets arriving within `horizon` turns.

    `horizon` turns means the fleet's ETA (estimated turns from now) is <= horizon.
    Events are sorted by eta_turn ascending.
    """
    events: list[FleetEvent] = []

    for fleet in state.fleets:
        dest_pid, dist = _infer_destination(fleet, state.planets)
        if dest_pid is None:
            continue

        speed = fleet_speed(fleet[6], state.ship_speed)
        eta_ticks = dist / speed if speed > 0.0 else float("inf")
        eta_turn = int(state.step + eta_ticks)

        if eta_turn - state.step <= horizon:
            events.append(FleetEvent(
                owner=fleet[1],
                destination_planet_id=dest_pid,
                eta_turn=eta_turn,
                ships_arriving=fleet[6],
            ))

    events.sort(key=lambda e: e.eta_turn)
    return events


def encode_events(
    events: list[FleetEvent],
    num_planets: int,
    horizon: int,
    observing_player: int = 0,
    base_turn: int = 0,
) -> np.ndarray:
    """Encode fleet arrival events as a fixed-shape array.

    Returns shape (horizon, num_planets, 2):
      [turn_offset, planet_idx, 0] : net signed ships delta arriving
                                     (positive for observing_player, negative for others)
      [turn_offset, planet_idx, 1] : count of distinct fleets contributing

    planet_idx maps planet IDs to contiguous indices via the arrival order of
    destination_planet_id.  For full-state use, build a consistent planet_id→idx
    map before calling this function.

    Turn offsets beyond `horizon` are zero-padded.  Events outside the
    [base_turn, base_turn + horizon) window are ignored.
    """
    arr = np.zeros((horizon, num_planets, 2), dtype=np.float32)

    # Build a planet-id → column index map from events (stable, first-seen order).
    # Callers can supply events pre-filtered to a consistent planet set.
    pid_to_idx: dict[int, int] = {}
    next_idx = 0

    for event in events:
        turn_offset = event.eta_turn - base_turn
        if turn_offset < 0 or turn_offset >= horizon:
            continue

        pid = event.destination_planet_id
        if pid not in pid_to_idx:
            if next_idx >= num_planets:
                continue  # more planets than allocated; skip overflow
            pid_to_idx[pid] = next_idx
            next_idx += 1

        col = pid_to_idx[pid]
        sign = 1.0 if event.owner == observing_player else -1.0
        arr[turn_offset, col, 0] += sign * event.ships_arriving
        arr[turn_offset, col, 1] += 1.0

    return arr
