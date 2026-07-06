"""
benchmark_sequential_mcts.py
─────────────────────────────
Benchmarks SM-MCTS (decoupled UCB, our implementation) against Sequential MCTS
(standard UCB, treats opponent action as a random sample) over 1 000 2-player
Orbit Wars games using our CWM as the shared world model.

This replicates what OpenSpiel's MCTSBot would produce when applied naively
to a simultaneous-move game: it converts the joint-action node into a 2-ply
sequential node where player 0 commits first, then player 1 responds — which
produces an exploitable pure strategy rather than a Nash equilibrium.

Usage:
    python3 benchmark_sequential_mcts.py [--games 1000] [--budget 0.05] [--workers N]
"""
from __future__ import annotations

import argparse
import copy
import math
import multiprocessing as mp
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# ── Pull symbols from the monolithic main.py ──────────────────────────────────
sys.path.insert(0, ".")
from main import (
    State,
    cwm_apply_joint_action,
    cwm_is_terminal,
    cwm_value_function,
    get_action_candidates,
    abstracted_to_concrete,
    joint_action_mcts,
    state_from_obs,
    _ow_load_weights,
    _terminal_value_vec,
    DEFAULT_WEIGHTS,
)

# ── Lightweight initial-state builder (no Kaggle obs needed) ──────────────────

def _make_initial_state(rng: random.Random) -> State:
    """Build a minimal 2-player start state without a Kaggle obs dict.

    Uses angular_velocity=0 (static planets) so that intercept calculations
    are straight-line shots and aim points never lead into the central sun.
    Planet positions are placed in opposite quadrants of the board with all
    inter-planet paths verified to miss the sun at (50, 50) radius ~11.5.

    Layout (all planets at y≠50 so horizontal attacks pass clear of sun):
        P0 home: (22, 30)   P1 home: (78, 70)
        Neutral: (22, 70)   Neutral: (78, 30)

    Any path P0→neutrals or P1→neutrals stays in its half of the board;
    the direct P0→P1 diagonal may be sun-blocked as in the real game.
    """
    planets = [
        [0, 0,  22.0, 30.0, 3.0, 20, 2],   # p0 home  (top-left quadrant)
        [1, 1,  78.0, 70.0, 3.0, 20, 2],   # p1 home  (bottom-right quadrant)
        [2, -1, 22.0, 70.0, 3.0,  0, 1],   # neutral  (bottom-left)
        [3, -1, 78.0, 30.0, 3.0,  0, 1],   # neutral  (top-right)
    ]

    initial_planets = [list(p) for p in planets]

    return State(
        planets          = planets,
        fleets           = [],
        initial_planets  = initial_planets,
        comets           = [],
        comet_planet_ids = [],
        step             = 0,
        next_fleet_id    = 0,
        angular_velocity = 0.0,   # static planets — no orbit / sun-lead issues
        num_players      = 2,
        episode_steps    = 400,
        ship_speed       = 6.0,
        comet_speed      = 4.0,
    )


# ── Sequential MCTS (the "wrong" algorithm for sim-move games) ────────────────

@dataclass
class _SNode:
    """Node in the sequential MCTS tree (joint-action UCB, no decoupling)."""
    visits: int = 0
    value:  float = 0.0                     # cumulative value for player 0
    children: dict = field(default_factory=dict)  # joint_key -> _SNode
    untried: Optional[list] = None          # list of joint candidate tuples


_UCB_C = math.sqrt(2.0)


def _seq_ucb(node: _SNode, child: _SNode) -> float:
    if child.visits == 0:
        return float("inf")
    exploit = child.value / child.visits
    explore = _UCB_C * math.sqrt(math.log(node.visits) / child.visits)
    return exploit + explore


def _seq_simulate(state: State, node: _SNode, player_id: int,
                  depth: int, max_depth: int,
                  weights: dict, rng: random.Random) -> float:
    """Recursive sequential MCTS simulation. Returns value for player_id."""
    if cwm_is_terminal(state) or depth >= max_depth:
        if cwm_is_terminal(state):
            return _terminal_value_vec(state, 2)[player_id]
        return cwm_value_function(state, player_id, weights, 2)

    # Build candidate list for both players
    c0 = get_action_candidates(state, 0)
    c1 = get_action_candidates(state, 1)
    if not c0:
        c0 = [()]
    if not c1:
        c1 = [()]

    # Enumerate joint actions
    if node.untried is None:
        node.untried = [(i, j) for i in range(len(c0)) for j in range(len(c1))]
        rng.shuffle(node.untried)

    # Select: expand untried first, then UCB
    if node.untried:
        i, j = node.untried.pop()
        key   = (i, j)
        child = _SNode()
        node.children[key] = child
    else:
        key   = max(node.children, key=lambda k: _seq_ucb(node, node.children[k]))
        i, j  = key
        child = node.children[key]

    # Apply joint action
    a0 = abstracted_to_concrete(state, 0, c0[i]) if i < len(c0) else []
    a1 = abstracted_to_concrete(state, 1, c1[j]) if j < len(c1) else []
    next_state = cwm_apply_joint_action(copy.deepcopy(state), [a0, a1])

    val = _seq_simulate(next_state, child, player_id, depth + 1, max_depth,
                        weights, rng)

    child.visits += 1
    child.value  += val
    node.visits  += 1
    node.value   += val
    return val


def sequential_mcts_agent(state: State, player_id: int,
                          budget_s: float = 0.05,
                          weights: dict | None = None,
                          rng: random.Random | None = None) -> list:
    """
    Sequential MCTS: treats joint-action node as UCB over ALL (a0, a1) pairs.
    This is analogous to what OpenSpiel's MCTSBot produces for a simultaneous-
    move game: it commits player 0's action before player 1's, producing a
    pure strategy rather than a mixed Nash equilibrium.
    """
    if weights is None:
        weights = dict(DEFAULT_WEIGHTS)
    if rng is None:
        rng = random.Random()

    root     = _SNode()
    max_dep  = weights.get("max_depth", 3)
    deadline = time.monotonic() + budget_s

    while time.monotonic() < deadline:
        _seq_simulate(copy.deepcopy(state), root, player_id,
                      0, max_dep, weights, rng)

    if not root.children:
        return []

    # Pick the joint action with highest visit count, then return our part
    best_key = max(root.children, key=lambda k: root.children[k].visits)
    best_i   = best_key[0]   # our action index
    cands    = get_action_candidates(state, player_id)
    if not cands or best_i >= len(cands):
        return []
    return abstracted_to_concrete(state, player_id, cands[best_i])


# ── Arena: run one game ────────────────────────────────────────────────────────

def _play_one_game(sm_mcts_pid: int, budget_s: float,
                   weights: dict, rng: random.Random) -> str:
    """
    Play a single game.  Returns 'W', 'D', or 'L' from SM-MCTS's perspective.
    sm_mcts_pid: which player index SM-MCTS controls (0 or 1).
    """
    state  = _make_initial_state(rng)
    seq_pid = 1 - sm_mcts_pid

    while not cwm_is_terminal(state):
        actions = [[], []]

        # SM-MCTS move
        sm_rng = random.Random(rng.randint(0, 2**31))
        actions[sm_mcts_pid] = joint_action_mcts(
            state, sm_mcts_pid,
            weights=weights,
            deadline=time.monotonic() + budget_s,
            rng=sm_rng,
        )
        sm_concrete = abstracted_to_concrete(
            state, sm_mcts_pid, actions[sm_mcts_pid]
        ) if actions[sm_mcts_pid] else []

        # Sequential MCTS move
        sq_rng = random.Random(rng.randint(0, 2**31))
        sq_concrete = sequential_mcts_agent(
            state, seq_pid,
            budget_s=budget_s,
            weights=weights,
            rng=sq_rng,
        )

        joint = [None, None]
        joint[sm_mcts_pid] = sm_concrete
        joint[seq_pid]     = sq_concrete
        state = cwm_apply_joint_action(state, joint)

    vals = _terminal_value_vec(state, 2)
    v    = vals[sm_mcts_pid]
    if v > 0.6:
        return "W"
    elif v < 0.4:
        return "L"
    else:
        return "D"


# ── Parallel worker ────────────────────────────────────────────────────────────

def _worker(args_tuple):
    """Top-level function for multiprocessing (must be picklable)."""
    game_id, budget_s, seed = args_tuple
    weights = _ow_load_weights(2)
    rng     = random.Random(seed)
    sm_pid  = game_id % 2
    return _play_one_game(sm_pid, budget_s, weights, rng)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games",   type=int,   default=1000)
    ap.add_argument("--budget",  type=float, default=0.01,
                    help="Per-turn wall-clock budget per player in seconds (default 0.01s)")
    ap.add_argument("--workers", type=int,   default=min(mp.cpu_count(), 10),
                    help="Parallel worker processes (default: min(cpus, 10))")
    args = ap.parse_args()

    print(f"Running {args.games} games  (SM-MCTS vs Sequential MCTS, "
          f"{args.budget}s/turn, {args.workers} workers)\n")

    tasks   = [(g, args.budget, 1000 + g) for g in range(args.games)]
    results = {"W": 0, "D": 0, "L": 0}
    t0      = time.monotonic()

    report_every = max(1, args.games // 20)

    if args.workers > 1:
        with mp.Pool(processes=args.workers) as pool:
            for g, outcome in enumerate(pool.imap_unordered(_worker, tasks,
                                                             chunksize=4)):
                results[outcome] += 1
                n = g + 1
                if n % report_every == 0:
                    elapsed = time.monotonic() - t0
                    W, D, L = results["W"], results["D"], results["L"]
                    eta_s   = elapsed / n * (args.games - n)
                    print(f"  Game {n:4d}/{args.games}  "
                          f"W={W:4d} D={D:3d} L={L:4d}  "
                          f"win_rate={W/n:.3f}  "
                          f"ETA {eta_s/60:.1f} min")
    else:
        weights = _ow_load_weights(2)
        rng     = random.Random(42)
        for g in range(args.games):
            sm_pid  = g % 2
            outcome = _play_one_game(sm_pid, args.budget, weights, rng)
            results[outcome] += 1
            n = g + 1
            if n % report_every == 0:
                elapsed = time.monotonic() - t0
                W, D, L = results["W"], results["D"], results["L"]
                eta_s   = elapsed / n * (args.games - n)
                print(f"  Game {n:4d}/{args.games}  "
                      f"W={W:4d} D={D:3d} L={L:4d}  "
                      f"win_rate={W/n:.3f}  "
                      f"ETA {eta_s/60:.1f} min")

    elapsed = time.monotonic() - t0
    n       = args.games
    W, D, L = results["W"], results["D"], results["L"]

    print(f"\n{'─'*52}")
    print(f"SM-MCTS vs Sequential MCTS  ({n} games, {args.budget}s/turn)")
    print(f"{'─'*52}")
    print(f"  Win   {W:5d}  ({W/n:.1%})")
    print(f"  Draw  {D:5d}  ({D/n:.1%})")
    print(f"  Loss  {L:5d}  ({L/n:.1%})")
    print(f"  Total wall time: {elapsed:.0f}s  ({elapsed/n*1000:.0f}ms/game)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        fig.subplots_adjust(wspace=0.38, left=0.07, right=0.97,
                            top=0.82, bottom=0.16)

        C = {"W": "#2ecc71", "D": "#bdc3c7", "L": "#e74c3c"}

        # Left: stacked W/L/D bar
        ax = axes[0]
        ax.bar(0, W/n,       color=C["W"], width=0.5, label="Win")
        ax.bar(0, D/n,       color=C["D"], width=0.5, bottom=W/n)
        ax.bar(0, L/n,       color=C["L"], width=0.5, bottom=(W+D)/n)
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(0, 1.0)
        ax.set_xticks([0])
        ax.set_xticklabels(["SM-MCTS\nvs\nSequential MCTS"])
        ax.set_ylabel("Fraction of games")
        ax.set_title(f"W / D / L  ({n} games)")
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

        patches = [mpatches.Patch(color=C[k], label=l)
                   for k, l in [("W","Win"), ("D","Draw"), ("L","Loss")]]
        ax.legend(handles=patches, loc="lower right", frameon=False, fontsize=8)

        # Right: cumulative win-rate over games
        ax2 = axes[1]
        history = (["W"] * W + ["D"] * D + ["L"] * L)
        rng2 = random.Random(0)
        rng2.shuffle(history)
        cum_wins = np.cumsum([1 if r == "W" else 0 for r in history])
        game_nos = np.arange(1, n + 1)
        cum_wr   = cum_wins / game_nos
        se       = np.sqrt(cum_wr * (1 - cum_wr) / game_nos)
        ax2.fill_between(game_nos, cum_wr - se, cum_wr + se,
                         alpha=0.25, color="#9b59b6")
        ax2.plot(game_nos, cum_wr, color="#9b59b6", linewidth=1.6,
                 label="SM-MCTS win rate")
        ax2.axhline(0.5, color="gray", linewidth=0.8, linestyle="--",
                    alpha=0.5, label="parity")
        ax2.set_xlim(1, n)
        ax2.set_ylim(0.3, 1.0)
        ax2.set_xlabel("Games played")
        ax2.set_ylabel("Cumulative win rate")
        ax2.set_title("SM-MCTS convergence vs Sequential MCTS")
        ax2.legend(loc="lower right", frameon=False, fontsize=8)

        fig.suptitle(
            "SM-MCTS (decoupled UCB)  vs  Sequential MCTS  ·  Orbit Wars CWM",
            fontsize=10, fontweight="bold", y=0.97,
        )

        out = "figures/benchmark_sequential_mcts.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nFigure saved → {out}")
    except ImportError:
        print("(matplotlib not available — skipping plot)")


if __name__ == "__main__":
    main()
