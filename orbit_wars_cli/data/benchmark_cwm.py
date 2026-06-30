"""
data/benchmark_cwm.py — Benchmark cwm_apply_joint_action throughput.

Runs 1 000 no-op transitions on realistic mid-game 2p and 4p states,
reports wall-clock timing, and suggests default MCTS simulation counts
with ~20 % headroom under a 1-second budget.

Usage
-----
    cd /home/moebius/Projects/OW2
    .venv/bin/python orbit_wars_cli/data/benchmark_cwm.py
"""
from __future__ import annotations

import json
import os
import sys
import time

# Ensure orbit_wars_cli/ is on sys.path so cwm.* imports work.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cwm.state import state_from_obs
from cwm.interpreter import cwm_apply_joint_action

_DATA_DIR = os.path.join(os.path.dirname(__file__), "trajectories")


def _load_mid_states(mode: str, n_files: int = 5) -> list:
    """Return up to *n_files* mid-game State objects from recorded trajectories.

    Uses the transition at index len(transitions)//3 (~step 150 for a
    500-step game) as a representative mid-game position.
    """
    folder = os.path.join(_DATA_DIR, mode)
    files = sorted(f for f in os.listdir(folder) if f.endswith(".json"))[:n_files]
    states = []
    for fname in files:
        with open(os.path.join(folder, fname)) as fh:
            game = json.load(fh)
        transitions = game["transitions"]
        mid_idx = len(transitions) // 3
        t = transitions[mid_idx]
        obs = t["obs_t"]
        config = game.get("config", {})
        num_players = 2 if mode == "2p" else 4
        state = state_from_obs(obs, config, cached_num_players=num_players)
        states.append(state)
    return states


def _run_benchmark(states: list, num_players: int, n_iters: int = 1_000) -> dict:
    """Time *n_iters* no-op transitions; return timing stats.

    Uses a no-op joint action (empty move list per player) to measure
    pure interpreter overhead.  Includes a 50-iteration warm-up.
    """
    n = len(states)
    noop_joints = [[[] for _ in range(num_players)] for _ in states]

    # Warm-up (let Python stabilise)
    for i in range(50):
        cwm_apply_joint_action(states[i % n], noop_joints[i % n])

    t0 = time.monotonic()
    for i in range(n_iters):
        cwm_apply_joint_action(states[i % n], noop_joints[i % n])
    elapsed = time.monotonic() - t0

    ms_per_call  = elapsed / n_iters * 1e3
    calls_per_s  = n_iters / elapsed
    # 20 % headroom: use 80 % of 1-second budget for MCTS simulations
    suggested    = int(calls_per_s * 0.8)

    return {
        "ms_per_call":    round(ms_per_call, 3),
        "calls_per_s":    round(calls_per_s, 1),
        "suggested_sims": suggested,
    }


def main() -> None:
    print("=== CWM throughput benchmark (n=1000 no-op transitions each) ===\n")

    results: dict[str, dict] = {}
    for mode, np in (("2p", 2), ("4p", 4)):
        states = _load_mid_states(mode)
        r = _run_benchmark(states, num_players=np)
        results[mode] = r
        print(
            f"  {mode}: {r['ms_per_call']:.2f} ms/call  "
            f"{r['calls_per_s']:.0f} calls/s  "
            f"-> suggested_sims = {r['suggested_sims']}  "
            f"(1 s budget, 20 % headroom)"
        )

    print()
    print(f"  MCTS_SIMS_2P_DEFAULT = {results['2p']['suggested_sims']}")
    print(f"  MCTS_SIMS_4P_DEFAULT = {results['4p']['suggested_sims']}")
    print()
    print("Paste these values into mcts/search.py as the default simulation counts.")


if __name__ == "__main__":
    main()
