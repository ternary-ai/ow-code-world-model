"""
cwm/action_space.py — Candidate action generation (Module 3).

Replaces the continuous angle action space with a discrete enumeration indexed
by (target_planet, ship_count_tier), using solve_intercept for exact angles.
This is the action set ISMCTS branches over.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cwm.state import State, CENTER, SUN_RADIUS
from cwm.geometry import fleet_speed
from cwm.masking import sun_blocks_path
from cwm.intercept import solve_intercept


@dataclass
class CandidateAction:
    """One discrete action from a source planet."""
    source_planet_id: int
    target_planet_id: int
    ships: int
    angle: float              # firing angle in radians, [0, 2π)
    predicted_arrival_turn: int


def generate_candidates(
    state: State,
    source_planet_id: int,
    ship_count_tiers: list[int],
) -> list[CandidateAction]:
    """Return one CandidateAction per (target_planet, ship_count_tier) pair.

    The angle and predicted arrival turn for each candidate are computed via
    solve_intercept.  Pairs are excluded when:
      - sun_blocks_path returns True for the source→target direction
      - the tier exceeds the source planet's current garrison
      - solve_intercept returns None (no convergent solution found)
    """
    # Find source planet
    source = next((p for p in state.planets if p[0] == source_planet_id), None)
    if source is None:
        return []

    src_pos = (source[2], source[3])
    garrison = source[5]
    av = state.angular_velocity
    sun_pos = (CENTER, CENTER)

    # Build initial_planets map for arrival prediction
    initial_map = {p[0]: p for p in state.initial_planets}

    candidates: list[CandidateAction] = []

    for target in state.planets:
        if target[0] == source_planet_id:
            continue

        tgt_pos = (target[2], target[3])

        # Check sun occlusion using current positions
        if sun_blocks_path(src_pos, tgt_pos, sun_pos, SUN_RADIUS):
            continue

        for ships in ship_count_tiers:
            if ships > garrison:
                continue

            angle = solve_intercept(
                source_pos=src_pos,
                target_planet_id=target[0],
                fleet_size=ships,
                depart_turn=state.step,
                state=state,
            )
            if angle is None:
                continue

            # Estimate arrival: travel distance to intercept point / fleet speed
            speed = fleet_speed(ships, state.ship_speed)
            aim_x = src_pos[0] + math.cos(angle)
            aim_y = src_pos[1] + math.sin(angle)

            # Refine: use aim direction + one distance estimate
            tgt_init = initial_map.get(target[0])
            if tgt_init is not None:
                dx = tgt_init[2] - CENTER
                dy = tgt_init[3] - CENTER
                r = math.sqrt(dx * dx + dy * dy)
                from cwm.state import ROTATION_RADIUS_LIMIT
                if r + target[4] < ROTATION_RADIUS_LIMIT:
                    # Orbiting: compute position at estimated arrival
                    init_ang = math.atan2(dy, dx)
                    d0 = math.hypot(tgt_pos[0] - src_pos[0], tgt_pos[1] - src_pos[1])
                    eta0 = d0 / speed if speed > 0.0 else d0
                    future_ang = init_ang + av * (state.step + eta0)
                    aim_x = CENTER + r * math.cos(future_ang)
                    aim_y = CENTER + r * math.sin(future_ang)
                else:
                    aim_x, aim_y = tgt_pos
            else:
                aim_x, aim_y = tgt_pos

            dist = math.hypot(aim_x - src_pos[0], aim_y - src_pos[1])
            eta = dist / speed if speed > 0.0 else 0.0
            arrival = int(state.step + eta)

            candidates.append(CandidateAction(
                source_planet_id=source_planet_id,
                target_planet_id=target[0],
                ships=ships,
                angle=angle,
                predicted_arrival_turn=arrival,
            ))

    return candidates
