"""
cwm/combat.py — N-way combat resolution for the Orbit Wars CWM.

IMPLEMENTATION CHOICE: OPTION A (strict top-2, fleet-only)
===========================================================
Source: orbit_wars_original.py, interpreter() lines ~618-657.

The source does NOT include the garrison in the fleet battle. The full model is:

  Step A — Fleet-only top-2 battle:
    player_ships = {owner: sum(fleet.ships) for arriving fleets per owner}
    Sort descending by ships. ONLY sorted[0] and sorted[1] participate.
    sorted[2+] owners' ships VANISH with NO effect on the outcome.
    (Issue #1047 Item 4 — still present in source, replicated exactly.)

    survivor_ships = sorted[0].ships - sorted[1].ships
    Tie (sorted[0].ships == sorted[1].ships): survivor_ships = 0, survivor_owner = -1

  Step B — Fleet winner vs. garrison:
    if survivor_ships > 0 and survivor_owner == planet.owner:
        garrison += survivor_ships                          (reinforcement)
    elif survivor_ships > 0 and survivor_owner != planet.owner:
        garrison -= survivor_ships
        if garrison < 0:
            planet.owner = survivor_owner
            garrison = abs(garrison)                        (conquest)
    # Tie → garrison untouched, owner unchanged.

  Black hole (Issue #1047 Item 1): if the planet has been removed from the
  planet_map before combat (expiring comet), the entry is skipped and all
  arriving fleet ships vanish.

3-attacker derivation (OPTION A confirmation):
  A=100, B=60, C=40 arriving at planet owned by D with garrison=30.
  Fleet battle: sorted [(A,100),(B,60),(C,40)]
    top=A(100), second=B(60), survivor=40
    C's 40 ships VANISH (OPTION A)
    survivor_owner = A
  Fleet winner vs. garrison: planet owner=D ≠ A, garrison=30, 30-40=-10 < 0
    → planet.owner=A, garrison=10.
  Result: A captures with 10 ships. B and C both lose all ships.
"""

from __future__ import annotations

import copy
from cwm.state import State


def resolve_combat(state: State, combat_queue: dict) -> State:
    """Resolve all queued planet combats and return updated state.

    Parameters
    ----------
    state : State
        Current game state (used for planet list; not mutated).
    combat_queue : dict[int, list[tuple[int, int]]]
        Mapping of planet_id -> [(fleet_owner, num_ships), ...]
        Built by the interpreter from fleet collision detection.
        May include planet IDs for comets that have since been removed
        (mid-tick expiry) — these produce the "black hole" effect.

    Returns
    -------
    State
        New state with updated planet ship counts and ownership.
    """
    # Build a mutable pid->planet mapping for O(1) access and clean mutation.
    # Using list() copies so we don't mutate the input state's planet lists.
    planet_map: dict[int, list] = {p[0]: list(p) for p in state.planets}

    for planet_id, attackers in combat_queue.items():
        if not attackers:
            continue

        planet = planet_map.get(planet_id)
        if planet is None:
            # Black hole: planet was removed (expiring comet) before combat.
            # Arriving fleet ships vanish silently.
            # Source: orbit_wars_original.py interpreter() ~line 620:
            #   if not planet or not planet_fleets: continue
            # Issue #1047 Item 1 — replicated exactly from source.
            continue

        # Step A: Fleet-only top-2 battle.
        # Sum ships per fleet owner (garrison NOT included — separate fight below).
        player_ships: dict[int, int] = {}
        for owner, ships in attackers:
            player_ships[owner] = player_ships.get(owner, 0) + ships

        if not player_ships:
            continue

        sorted_players = sorted(
            player_ships.items(), key=lambda item: item[1], reverse=True
        )
        top_player, top_ships = sorted_players[0]

        if len(sorted_players) > 1:
            second_ships = sorted_players[1][1]
            survivor_ships = top_ships - second_ships
            # OPTION A: sorted_players[2:] owners' ships vanish without effect.
            # Source: only sorted[0] and sorted[1] are accessed.
            if sorted_players[0][1] == sorted_players[1][1]:
                # Exact tie: all fleet ships destroyed.
                survivor_ships = 0
            survivor_owner = top_player if survivor_ships > 0 else -1
        else:
            # Single fleet owner: wins the fleet battle outright.
            survivor_owner = top_player
            survivor_ships = top_ships

        # Step B: Fleet winner vs. garrison.
        if survivor_ships > 0:
            if planet[1] == survivor_owner:
                # Friendly reinforcement: add to garrison.
                planet[5] += survivor_ships
            else:
                # Attack garrison.
                planet[5] -= survivor_ships
                if planet[5] < 0:
                    # Conquest: attacker takes over with surplus ships.
                    planet[1] = survivor_owner
                    planet[5] = abs(planet[5])
                # If planet[5] >= 0 after subtraction: garrison holds, owner unchanged.

    new_state = copy.copy(state)
    new_state.planets = list(planet_map.values())
    return new_state
