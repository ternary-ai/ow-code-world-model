"""
cli/validate.py — `orbit-wars validate` subcommand.

Reports two accuracy metrics then runs the full test suite:

  1. Transition accuracy  — % of CWM steps that exactly reproduce the
                            ground-truth kaggle_environments successor state.
                            Source: tests/test_transitions.py (14 413 steps).

  2. Win-rate accuracy    — MCTS agent win rate vs. a greedy baseline over
                            N short self-play games per mode (2p and 4p).
                            Win = sole rank-1 finish; draw = tied for first.

Usage (via main.py):
    orbit-wars validate [--verbose] [--fast] [--eval-games N]

Options:
    --verbose         Pass -v to pytest for per-test output.
    --fast            Skip test_transitions.py (saves ~30 s).
    --eval-games N    Games per mode for win-rate check (default: 10).
                      Set 0 to skip the win-rate check.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

# Ensure orbit_wars_cli/ is on sys.path when imported from main.py.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Transition accuracy ───────────────────────────────────────────────────────

class _CountPlugin:
    """Minimal pytest plugin to capture pass/fail counts."""
    last_passed: int = 0
    last_total:  int = 0

    def pytest_terminal_summary(self, terminalreporter, exitstatus, config):
        passed = len(terminalreporter.stats.get("passed", []))
        failed = len(terminalreporter.stats.get("failed", []))
        error  = len(terminalreporter.stats.get("error",  []))
        _CountPlugin.last_passed = passed
        _CountPlugin.last_total  = passed + failed + error


def _transition_accuracy(tests_dir: str) -> tuple[int, int]:
    """Run test_transitions.py; return (passed, total)."""
    import pytest
    plugin = _CountPlugin()
    pytest.main(
        [os.path.join(tests_dir, "test_transitions.py"),
         "-q", "--tb=no", "--no-header"],
        plugins=[plugin],
    )
    return plugin.last_passed, plugin.last_total


# ── Win-rate accuracy ─────────────────────────────────────────────────────────

def _win_rate_accuracy(n_games: int, seed: int = 42) -> dict:
    """Run MCTS agent vs greedy baseline; return result dict per mode.

    Returns:
        {"2p": {"wins": int, "draws": int, "losses": int, "games": int}, "4p": {...}}
    """
    from cwm.state import state_from_obs
    from cwm.interpreter import cwm_apply_joint_action, cwm_is_terminal, cwm_get_rewards
    from mcts.actions import abstracted_to_concrete
    from mcts.search import joint_action_mcts, _greedy_abstract
    from mcts.value_fn import DEFAULT_WEIGHTS

    _MCTS_DIR = os.path.join(os.path.dirname(__file__), "..", "mcts")
    _DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "trajectories")

    def _load_weights(mode: str) -> dict:
        path = os.path.join(_MCTS_DIR, f"best_weights_{mode}.json")
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)
        w = dict(DEFAULT_WEIGHTS)
        w["mcts_budget_s"] = 0.10
        return w

    def _load_states(mode: str, n: int) -> list:
        folder = os.path.join(_DATA_DIR, mode)
        files  = sorted(f for f in os.listdir(folder) if f.endswith(".json"))[:n]
        states = []
        for fname in files:
            with open(os.path.join(folder, fname)) as fh:
                game = json.load(fh)
            obs    = game["transitions"][0]["obs_t"]
            config = game.get("config", {})
            np_    = 2 if mode == "2p" else 4
            states.append(state_from_obs(obs, config, cached_num_players=np_))
        return states

    results = {}
    rng = random.Random(seed)

    for mode, num_players in (("2p", 2), ("4p", 4)):
        weights = _load_weights(mode)
        budget  = weights.get("mcts_budget_s", 0.10)
        states  = _load_states(mode, n=min(n_games, 5))
        wins = draws = losses = 0

        for i in range(n_games):
            state = states[i % len(states)]
            for _ in range(80):
                if cwm_is_terminal(state):
                    break
                joint = [[] for _ in range(num_players)]
                act0 = joint_action_mcts(
                    state, 0,
                    time_budget_s=budget,
                    weights=weights,
                    num_players=num_players,
                    rng=rng,
                )
                joint[0] = abstracted_to_concrete(state, 0, act0)
                for pid in range(1, num_players):
                    act = _greedy_abstract(state, pid)
                    joint[pid] = abstracted_to_concrete(state, pid, act)
                state = cwm_apply_joint_action(state, joint)

            rewards  = cwm_get_rewards(state)
            own      = rewards[0]
            best_opp = max(rewards[1:]) if len(rewards) > 1 else 0.0
            if own > best_opp:
                wins += 1
            elif own == best_opp:
                draws += 1
            else:
                losses += 1

        results[mode] = {"wins": wins, "draws": draws,
                         "losses": losses, "games": n_games}

    return results


# ── Public entry point ────────────────────────────────────────────────────────

def run(
    verbose: bool = False,
    fast: bool = False,
    eval_games: int = 10,
) -> int:
    """Run validation and print accuracy metrics. Returns pytest exit code."""
    try:
        import pytest
    except ImportError:
        print("ERROR: pytest is not installed.  Run: pip install pytest",
              file=sys.stderr)
        return 1

    tests_dir = os.path.join(os.path.dirname(__file__), "..", "tests")

    # ── 1. Transition accuracy ────────────────────────────────────────────────
    if not fast:
        print("── Transition accuracy (running test_transitions.py) ──")
        t0 = time.monotonic()
        passed, total = _transition_accuracy(tests_dir)
        elapsed = time.monotonic() - t0
        pct = 100.0 * passed / total if total else 0.0
        print(
            f"   {passed}/{total} steps match ground truth  "
            f"({pct:.2f}%)  [{elapsed:.1f}s]\n"
        )
    else:
        print("── Transition accuracy: skipped (--fast)\n")

    # ── 2. Win-rate accuracy ──────────────────────────────────────────────────
    if eval_games > 0:
        print(f"── Win-rate accuracy (MCTS vs greedy, {eval_games} games/mode) ──")
        t0 = time.monotonic()
        wr = _win_rate_accuracy(eval_games)
        elapsed = time.monotonic() - t0
        for mode, r in wr.items():
            n      = r["games"]
            wr_pct = 100.0 * r["wins"] / n
            dr_pct = 100.0 * r["draws"] / n
            random_baseline = 100 // (2 if mode == "2p" else 4)
            print(
                f"   {mode}: {r['wins']}W / {r['draws']}D / {r['losses']}L  "
                f"win_rate={wr_pct:.1f}%  draw_rate={dr_pct:.1f}%"
                f"  (random-baseline≈{random_baseline}%)"
            )
        print(f"   [{elapsed:.1f}s]\n")
    else:
        print("── Win-rate accuracy: skipped (--eval-games 0)\n")

    # ── 3. Full test suite ────────────────────────────────────────────────────
    print("── Full test suite ──")
    args = [tests_dir, "--tb=short"]
    if verbose:
        args.append("-v")
    else:
        args.append("-q")
    if fast:
        args += ["--ignore", os.path.join(tests_dir, "test_transitions.py")]

    return pytest.main(args)
