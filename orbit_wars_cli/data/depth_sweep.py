"""
data/depth_sweep.py — Controlled MCTS depth A/B at a realistic per-move budget.

Compares the tree-based SM-MCTS at several `max_depth` values against the greedy
baseline, holding everything else fixed. The 0.05 s arena budget is ~14x below
the production cap (0.70 s) and unfairly starves a deep tree of simulations, so
this harness evaluates at a realistic budget to measure the true value of depth.

Usage:
    python orbit_wars_cli/data/depth_sweep.py \
        --mode 2p --budget 0.30 --games 8 --max-steps 150 --depths 1 2 3
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cli.arena import greedy_agent, mcts_agent, head_to_head, _load_weights


def _depth_agent(mode: str, depth: int, budget: float):
    w = dict(_load_weights(mode))
    w["max_depth"] = depth
    w["mcts_budget_s"] = budget
    return mcts_agent(w, name=f"mcts_d{depth}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["2p", "4p"], default="2p")
    ap.add_argument("--budget", type=float, default=0.30)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    greedy = greedy_agent()
    print(f"=== depth sweep  mode={args.mode}  budget={args.budget}s  "
          f"games={args.games}  max_steps={args.max_steps} ===")
    for d in args.depths:
        agent = _depth_agent(args.mode, d, args.budget)
        t0 = time.monotonic()
        res = head_to_head(agent, greedy, args.mode, args.games,
                           seed=args.seed, max_steps=args.max_steps)
        dt = time.monotonic() - t0
        lo, hi = res.ci()
        print(f"  depth={d}: W/D/L={res.a_wins:.0f}/{res.draws:.0f}/{res.a_losses:.0f}"
              f"  rate={res.a_rate:.3f}  95% CI=[{lo:.3f},{hi:.3f}]"
              f"  ({dt:.0f}s)")


if __name__ == "__main__":
    main()
