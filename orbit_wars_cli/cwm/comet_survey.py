"""
cwm/comet_survey.py — Comet-mechanics investigative spike (Module 8).

Purpose: DATA COLLECTION, not a feature.  Runs num_episodes of the CWM
with random agents and logs per-turn comet state to a CSV for manual review.

Do not build comet-specific features (hazard maps, avoidance penalties) until
the spike's findings have been reviewed and comet mechanics confirmed.

Output columns:
  episode     : episode index (0-based)
  turn        : game turn (state.step before the tick)
  comet_positions : JSON-encoded list of [x, y] per active comet planet, or ""
  comet_planet_ids : JSON-encoded list of active comet planet IDs, or ""
  num_comets  : count of active comet planets
  garrison_delta : JSON-encoded map of planet_id → garrison change this turn
  ownership_change : JSON-encoded list of planet_ids whose owner changed
"""

from __future__ import annotations

import csv
import json
import math
import random

from cwm.state import State
from cwm.interpreter import cwm_apply_joint_action, cwm_is_terminal


# ── Minimal game-state factory ─────────────────────────────────────────────────

def _make_start_state(seed: int, num_players: int = 2) -> State:
    """Build a minimal starting state for a survey episode.

    Planets are placed symmetrically at corners, away from the sun.
    No comets at game start (comets spawn at turn 50, 150, ...).
    """
    rng = random.Random(seed)

    # Fixed-position symmetric map: 2 players, one planet each
    planet_positions = [
        (15.0, 15.0),   # player 0
        (85.0, 85.0),   # player 1
        (85.0, 15.0),   # neutral
        (15.0, 85.0),   # neutral
    ]
    positions = planet_positions[:max(num_players, 2)]

    planets = []
    for i, (x, y) in enumerate(positions):
        owner = i if i < num_players else -1
        planets.append([i, owner, x, y, 2.0, rng.randint(8, 15), rng.randint(1, 3)])

    initial = [list(p) for p in planets]

    return State(
        planets=planets,
        fleets=[],
        initial_planets=initial,
        comets=[],
        comet_planet_ids=[],
        step=0,
        next_fleet_id=0,
        angular_velocity=rng.uniform(0.025, 0.05),
        num_players=num_players,
        episode_steps=500,
        ship_speed=6.0,
        comet_speed=4.0,
    )


def _random_joint_action(state: State, rng: random.Random) -> list:
    """Produce a random joint action: each player randomly launches from one planet."""
    action: list = [[] for _ in range(state.num_players)]
    planet_map = {p[0]: p for p in state.planets}

    for player_id in range(state.num_players):
        own = [p for p in state.planets if p[1] == player_id and p[5] > 0]
        if not own:
            continue
        src = rng.choice(own)
        others = [p for p in state.planets if p[0] != src[0]]
        if not others:
            continue
        tgt = rng.choice(others)
        dx, dy = tgt[2] - src[2], tgt[3] - src[3]
        angle = math.atan2(dy, dx)
        ships = rng.randint(1, max(1, src[5] // 2))
        action[player_id] = [[src[0], angle, ships]]

    return action


def run_survey(
    num_episodes: int,
    out_path: str,
    max_turns: int = 500,
) -> None:
    """Run num_episodes of the CWM with random agents and write a CSV to out_path.

    Each row records one (episode, turn) pair with comet state and any
    garrison/ownership changes relative to the previous turn.
    """
    fieldnames = [
        "episode", "turn", "comet_positions", "comet_planet_ids",
        "num_comets", "garrison_delta", "ownership_change",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ep in range(num_episodes):
            rng = random.Random(ep * 1000)
            state = _make_start_state(seed=ep)
            prev_garrison = {p[0]: p[5] for p in state.planets}
            prev_owner = {p[0]: p[1] for p in state.planets}

            for _ in range(max_turns):
                if cwm_is_terminal(state):
                    break

                # Record comet data BEFORE the tick
                comet_positions = [
                    [p[2], p[3]]
                    for p in state.planets
                    if p[0] in state.comet_planet_ids
                ]
                comet_ids = list(state.comet_planet_ids)
                num_comets = len(comet_ids)

                action = _random_joint_action(state, rng)
                next_state = cwm_apply_joint_action(state, action)

                # Compute changes
                garrison_delta = {}
                for p in next_state.planets:
                    prev = prev_garrison.get(p[0], 0)
                    if p[5] != prev:
                        garrison_delta[p[0]] = p[5] - prev

                ownership_change = []
                for p in next_state.planets:
                    if p[1] != prev_owner.get(p[0]):
                        ownership_change.append(p[0])

                writer.writerow({
                    "episode": ep,
                    "turn": state.step,
                    "comet_positions": json.dumps(comet_positions) if comet_positions else "",
                    "comet_planet_ids": json.dumps(comet_ids) if comet_ids else "",
                    "num_comets": num_comets,
                    "garrison_delta": json.dumps(garrison_delta) if garrison_delta else "",
                    "ownership_change": json.dumps(ownership_change) if ownership_change else "",
                })

                prev_garrison = {p[0]: p[5] for p in next_state.planets}
                prev_owner = {p[0]: p[1] for p in next_state.planets}
                state = next_state
