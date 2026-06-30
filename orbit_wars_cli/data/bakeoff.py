"""
data/bakeoff.py — ground-truth in-engine bake-off.

Pits MCTS (current best weights) and MCTS (a backup weights file) against the
greedy baseline using the arena head-to-head harness, so we can tell whether a
re-tune actually moved real strength — instead of trusting the self-referential
self-play pooled win-rate that hid the aiming bug.

Usage:
    python orbit_wars_cli/data/bakeoff.py --mode 4p --games 48 \
        --old /tmp/ow_weights_backup_20260619/best_weights_4p.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cli.arena import (  # noqa: E402
    greedy_agent,
    mcts_agent,
    mcts_agent_from_file,
    head_to_head,
)


def _fmt(res) -> str:
    lo, hi = res.ci()
    return (
        f"{res.a_name:>16} vs {res.b_name:<10} "
        f"score={res.a_rate*100:5.1f}%  "
        f"(W{res.a_wins}/D{res.draws}/L{res.a_losses})  "
        f"95%CI[{lo*100:4.1f}, {hi*100:4.1f}]"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="4p", choices=["2p", "4p"])
    ap.add_argument("--games", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--budget", type=float, default=None,
                    help="override mcts_budget_s for both MCTS agents")
    ap.add_argument("--old", default=None,
                    help="path to a backup weights JSON to also test vs greedy")
    args = ap.parse_args()

    greedy = greedy_agent()
    new = mcts_agent_from_file(args.mode, name=f"mcts_{args.mode}_new")
    if args.budget is not None:
        new.budget_s = args.budget

    agents = [new]
    if args.old:
        with open(args.old) as fh:
            old_w = json.load(fh)
        old = mcts_agent(old_w, name=f"mcts_{args.mode}_old")
        if args.budget is not None:
            old.budget_s = args.budget
        agents.append(old)

    print(f"bake-off  mode={args.mode}  games={args.games}  seed={args.seed}  "
          f"max_steps={args.max_steps}  budget={'file' if args.budget is None else args.budget}")
    print("-" * 78)
    for ag in agents:
        t0 = time.monotonic()
        res = head_to_head(ag, greedy, args.mode, args.games,
                           seed=args.seed, max_steps=args.max_steps)
        dt = time.monotonic() - t0
        print(_fmt(res) + f"   [{dt:5.1f}s, budget={ag.budget_s}]")
    print("-" * 78)
    print("Read: score>50% means the MCTS agent beats greedy in-engine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
