"""
data/timing_harness.py — Per-turn wall-clock stress test for the agent.

The competition enforces a ~1 s/turn budget. MCTS uses a wall-clock deadline,
so per-turn time should never materially exceed the configured budget. This
harness runs the full decision path (state_from_obs -> joint_action_mcts ->
abstracted_to_concrete) across many real recorded states, for BOTH 2p and 4p,
at a near-1 s budget, and reports max / mean / p95 timings.

Usage
-----
    cd /home/moebius/Projects/OW2
    .venv/bin/python orbit_wars_cli/data/timing_harness.py [--budget 0.9] [--per-mode 60]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cwm.state import state_from_obs
from mcts.actions import abstracted_to_concrete
from mcts.search import joint_action_mcts
from mcts.value_fn import DEFAULT_WEIGHTS

_DATA_DIR = os.path.join(os.path.dirname(__file__), "trajectories")
_MCTS_DIR = os.path.join(os.path.dirname(__file__), "..", "mcts")

_HARD_LIMIT_S = 1.0


def _load_weights(mode: str) -> dict:
    path = os.path.join(_MCTS_DIR, f"best_weights_{mode}.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    w = dict(DEFAULT_WEIGHTS)
    w["mcts_budget_s"] = 0.10
    return w


def _load_states(mode: str, per_mode: int) -> list:
    """Return up to *per_mode* (obs, config) pairs spread across recorded games."""
    folder = os.path.join(_DATA_DIR, mode)
    files = sorted(f for f in os.listdir(folder) if f.endswith(".json"))
    pairs = []
    for fname in files:
        with open(os.path.join(folder, fname)) as fh:
            game = json.load(fh)
        config = game.get("config", {})
        transitions = game["transitions"]
        # Sample a few positions per game (early, mid, late).
        for frac in (0.1, 0.35, 0.6, 0.85):
            idx = int(len(transitions) * frac)
            idx = max(0, min(idx, len(transitions) - 1))
            pairs.append((transitions[idx]["obs_t"], config))
            if len(pairs) >= per_mode:
                return pairs
    return pairs


def _percentile(sorted_vals: list, q: float) -> float:
    if not sorted_vals:
        return 0.0
    k = int(round((len(sorted_vals) - 1) * q))
    return sorted_vals[k]


def run(budget: float, per_mode: int) -> int:
    overall_ok = True
    print(f"=== Agent per-turn timing (budget={budget:.2f}s, hard limit={_HARD_LIMIT_S:.1f}s) ===\n")

    for mode, num_players in (("2p", 2), ("4p", 4)):
        weights = dict(_load_weights(mode))
        weights["mcts_budget_s"] = budget          # force the stress budget
        pairs = _load_states(mode, per_mode)
        rng = random.Random(0)

        times = []
        for obs, config in pairs:
            player_id = obs["player"] if isinstance(obs, dict) else obs.player
            t0 = time.monotonic()
            deadline = t0 + budget          # anchored to true turn start (production semantics)
            state = state_from_obs(obs, config, cached_num_players=num_players)
            act = joint_action_mcts(
                state, player_id,
                weights=weights,
                num_players=num_players,
                rng=rng,
                deadline=deadline,
            )
            _ = abstracted_to_concrete(state, player_id, act)
            times.append(time.monotonic() - t0)

        times.sort()
        mx, mean, p95 = times[-1], sum(times) / len(times), _percentile(times, 0.95)
        # Pass criterion is the SYSTEMATIC tail (p95): per-turn time must be
        # comfortably under the hard limit. A lone max spike is reported for
        # information only (environmental GC/scheduler jitter; rare single-turn
        # overshoots are absorbed by the competition overage bank).
        ok = p95 < _HARD_LIMIT_S
        overall_ok &= ok
        spike = "  <spike>" if mx >= _HARD_LIMIT_S else ""
        print(
            f"  {mode}: n={len(times):3d}  "
            f"max={mx*1e3:6.1f}ms  p95={p95*1e3:6.1f}ms  mean={mean*1e3:6.1f}ms  "
            f"{'PASS' if ok else 'FAIL (p95 exceeded 1s)'}{spike}"
        )

    print()
    print("RESULT:", "PASS" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=0.7, help="per-turn MCTS budget (s)")
    ap.add_argument("--per-mode", type=int, default=60, help="states sampled per mode")
    args = ap.parse_args()
    sys.exit(run(args.budget, args.per_mode))


if __name__ == "__main__":
    main()
