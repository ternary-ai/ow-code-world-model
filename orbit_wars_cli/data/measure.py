"""
data/measure.py — Parallel, high-precision win-rate measurement vs greedy.

The trainer evaluates each candidate on only a handful of games (noisy: a 16-game
win-rate has a 95% CI spanning ~±0.24). This harness plays MANY games of the
current best_weights_{mode}.json vs greedy, split across worker processes, and
reports an aggregate win-rate with a Wilson 95% confidence interval — so we can
tell real strength from sampling noise, and measure at the true production
budget (0.70 s) rather than the throttled tuning budget.

Usage:
    python orbit_wars_cli/data/measure.py --mode 4p --games 64 --budget 0.70 \
        --workers 6 --max-steps 120 [--weights PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cli.arena import greedy_agent, mcts_agent, head_to_head, wilson_interval

_MCTS_DIR = os.path.join(os.path.dirname(__file__), "..", "mcts")


def _load_weights(mode: str, path: str | None) -> dict:
    if path is None:
        path = os.path.join(_MCTS_DIR, f"best_weights_{mode}.json")
    with open(path) as fh:
        return json.load(fh)


def _chunk_worker(task: tuple) -> tuple:
    """Play one chunk of games vs greedy. Returns (a_score, games)."""
    weights, mode, n_games, seed, max_steps, budget = task
    w = dict(weights)
    if budget is not None:
        w["mcts_budget_s"] = budget
    cand = mcts_agent(w, name="best")
    res = head_to_head(
        cand, greedy_agent(), mode,
        games=n_games, seed=seed, max_steps=max_steps,
    )
    return res.a_score, res.games


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["2p", "4p"], required=True)
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--budget", type=float, default=None,
                    help="Per-turn MCTS budget (default: use the weights file).")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=120, dest="max_steps")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    weights = _load_weights(args.mode, args.weights)

    # Split games into per-worker chunks (multiple of num_players keeps seat
    # rotation balanced within each chunk).
    num_players = 2 if args.mode == "2p" else 4
    n_chunks = max(1, args.workers)
    base = max(num_players, (args.games // n_chunks // num_players) * num_players)
    chunks = []
    remaining = args.games
    i = 0
    while remaining > 0:
        g = min(base, remaining)
        chunks.append((weights, args.mode, g, args.seed + i * 100003,
                       args.max_steps, args.budget))
        remaining -= g
        i += 1

    total_games = sum(c[2] for c in chunks)
    budget_str = f"{args.budget:.2f}s (override)" if args.budget is not None \
        else f"{weights.get('mcts_budget_s', 0.10):.2f}s (from weights)"
    print(f"Measuring {args.mode} best vs greedy: {total_games} games, "
          f"budget={budget_str}, workers={args.workers}, max_steps={args.max_steps}")

    total_score = 0.0
    played = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for a_score, games in pool.map(_chunk_worker, chunks):
            total_score += a_score
            played += games

    rate = total_score / played if played else 0.0
    lo, hi = wilson_interval(total_score, played)
    fair = 1.0 / num_players
    print(f"\n  {args.mode}: win_rate = {rate:.3f}  (95% CI [{lo:.3f}, {hi:.3f}])"
          f"  over {played} games")
    print(f"  fair share = {fair:.3f}  =>  {rate / fair:.2f}x fair share")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
