"""
cli/arena.py — head-to-head evaluation harness with ELO + win-rate CIs.

Unlike the truncated self-play eval in train.py (player 0 vs the rest), the
arena pits *named* agents against one another with proper seat rotation to
remove first-mover bias, plays full-length (or long) games, and reports:

  - head-to-head win/draw/loss with a Wilson 95% confidence interval, and
  - an ELO table when more than two agents compete.

Agents
------
An agent is a small wrapper exposing
    decide(state, player_id, num_players, rng, deadline) -> concrete_action
plus a `name`. Built-ins:
    greedy_agent()                      — the fast greedy heuristic
    mcts_agent(weights, name)           — MCTS with a given weight dict
    mcts_agent_from_file(mode, name)    — MCTS loading best_weights_{mode}.json

Used by:
    cli/train.py acceptance gating (optional), and direct CLI:
        python orbit_wars_cli/main.py arena --mode 2p --games 100
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cwm.state import state_from_obs
from cwm.interpreter import cwm_apply_joint_action, cwm_is_terminal, cwm_get_rewards
from mcts.actions import abstracted_to_concrete
from mcts.search import joint_action_mcts, _greedy_abstract
from mcts.value_fn import DEFAULT_WEIGHTS

_MCTS_DIR = os.path.join(os.path.dirname(__file__), "..", "mcts")
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "trajectories")


# ── Agents ─────────────────────────────────────────────────────────────────────

@dataclass
class Agent:
    """A named decision policy. `fn(state, pid, num_players, rng, deadline)`
    must return a concrete action list for `cwm_apply_joint_action`."""
    name: str
    fn: object
    budget_s: float = 0.10

    def decide(self, state, player_id, num_players, rng):
        return self.fn(state, player_id, num_players, rng, self.budget_s)


def greedy_agent(name: str = "greedy") -> Agent:
    def _fn(state, pid, num_players, rng, budget_s):
        return abstracted_to_concrete(state, pid, _greedy_abstract(state, pid))
    return Agent(name=name, fn=_fn, budget_s=0.0)


def mcts_agent(weights: dict, name: str = "mcts") -> Agent:
    budget = weights.get("mcts_budget_s", 0.10)

    def _fn(state, pid, num_players, rng, budget_s):
        act = joint_action_mcts(
            state, pid,
            weights=weights,
            num_players=num_players,
            rng=rng,
            deadline=time.monotonic() + budget_s,
        )
        return abstracted_to_concrete(state, pid, act)

    return Agent(name=name, fn=_fn, budget_s=budget)


def _load_weights(mode: str) -> dict:
    path = os.path.join(_MCTS_DIR, f"best_weights_{mode}.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    w = dict(DEFAULT_WEIGHTS)
    w["mcts_budget_s"] = 0.10
    return w


def mcts_agent_from_file(mode: str, name: str | None = None) -> Agent:
    weights = _load_weights(mode)
    return mcts_agent(weights, name or f"mcts_{mode}")


# ── Start-state loading ────────────────────────────────────────────────────────

def load_start_states(mode: str, n: int | None = None) -> list:
    """Return step-0 State objects from recorded trajectories for *mode*."""
    folder = os.path.join(_DATA_DIR, mode)
    files = sorted(f for f in os.listdir(folder) if f.endswith(".json"))
    if n is not None:
        files = files[:n]
    np_ = 2 if mode == "2p" else 4
    states = []
    for fname in files:
        with open(os.path.join(folder, fname)) as fh:
            game = json.load(fh)
        obs = game["transitions"][0]["obs_t"]
        config = game.get("config", {})
        states.append(state_from_obs(obs, config, cached_num_players=np_))
    return states


# ── Single match ───────────────────────────────────────────────────────────────

def play_match(
    seat_agents: list,
    start_state,
    num_players: int,
    max_steps: int = 500,
    rng: random.Random | None = None,
) -> list:
    """Play one game. `seat_agents[i]` controls seat i.

    Returns per-seat result: 1.0 win (sole rank-1), 0.5 draw (tied rank-1),
    0.0 loss — matching the competition's ELO scoring.
    """
    if rng is None:
        rng = random.Random()
    # Fresh deep copy so the shared start_state is never mutated.
    from cwm.interpreter import _copy_state
    state = _copy_state(start_state)

    for _ in range(max_steps):
        if cwm_is_terminal(state):
            break
        joint = [[] for _ in range(num_players)]
        for pid in range(num_players):
            joint[pid] = seat_agents[pid].decide(state, pid, num_players, rng)
        state = cwm_apply_joint_action(state, joint)

    rewards = cwm_get_rewards(state)
    top = max(rewards) if rewards else 0.0
    n_top = sum(1 for r in rewards if r == top)
    out = []
    for r in rewards:
        if r == top and n_top == 1:
            out.append(1.0)
        elif r == top:
            out.append(0.5)
        else:
            out.append(0.0)
    return out


# ── Statistics ─────────────────────────────────────────────────────────────────

def wilson_interval(wins: float, n: int, z: float = 1.96) -> tuple:
    """Wilson score 95% CI for a win *rate* (draws count as 0.5 wins).

    Returns (low, high) on the [0, 1] scale.
    """
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


# ── Two-agent head-to-head ─────────────────────────────────────────────────────

@dataclass
class H2HResult:
    a_name: str
    b_name: str
    games: int
    a_score: float          # total points for A (win=1, draw=0.5)
    a_wins: int
    draws: int
    a_losses: int

    @property
    def a_rate(self) -> float:
        return self.a_score / self.games if self.games else 0.0

    def ci(self) -> tuple:
        return wilson_interval(self.a_score, self.games)


def head_to_head(
    agent_a: Agent,
    agent_b: Agent,
    mode: str,
    games: int,
    seed: int = 0,
    max_steps: int = 500,
) -> H2HResult:
    """Play *games* games of A vs B in *mode*, rotating seats to cancel
    first-mover advantage. In 4p, A occupies one rotating seat and B fills the
    other three.
    """
    num_players = 2 if mode == "2p" else 4
    states = load_start_states(mode)
    rng = random.Random(seed)

    a_score = a_wins = draws = a_losses = 0.0
    for g in range(games):
        start = states[g % len(states)]
        a_seat = g % num_players                  # rotate A's seat
        seat_agents = [agent_b] * num_players
        seat_agents[a_seat] = agent_a
        results = play_match(seat_agents, start, num_players, max_steps, rng)
        r = results[a_seat]
        a_score += r
        if r == 1.0:
            a_wins += 1
        elif r == 0.5:
            draws += 1
        else:
            a_losses += 1

    return H2HResult(
        a_name=agent_a.name, b_name=agent_b.name, games=games,
        a_score=a_score, a_wins=int(a_wins), draws=int(draws),
        a_losses=int(a_losses),
    )


def head_to_head_field(
    agent_a: Agent,
    field: list,
    mode: str,
    games: int,
    seed: int = 0,
    max_steps: int = 500,
) -> H2HResult:
    """Play *games* games of A against a heterogeneous pool of opponents.

    A rotates seats each game; the (num_players-1) opponent seats are filled
    by cycling through *field*.  More representative than 3 clones of the
    same agent, which is the key flaw in vanilla 4p head_to_head.
    """
    num_players = 2 if mode == "2p" else 4
    states = load_start_states(mode)
    rng = random.Random(seed)

    a_score = a_wins = draws = a_losses = 0.0
    for g in range(games):
        start  = states[g % len(states)]
        a_seat = g % num_players
        seat_agents: list = []
        opp_slot = 0
        for pid in range(num_players):
            if pid == a_seat:
                seat_agents.append(agent_a)
            else:
                seat_agents.append(field[(g + opp_slot) % len(field)])
                opp_slot += 1
        results = play_match(seat_agents, start, num_players, max_steps, rng)
        r = results[a_seat]
        a_score += r
        if r == 1.0:    a_wins   += 1
        elif r == 0.5:  draws    += 1
        else:           a_losses += 1

    return H2HResult(
        a_name=agent_a.name,
        b_name="field(" + ",".join(a.name for a in field[:3]) + ")",
        games=games,
        a_score=a_score, a_wins=int(a_wins), draws=int(draws),
        a_losses=int(a_losses),
    )


# ── Round-robin ELO ─────────────────────────────────────────────────────────────

def round_robin_elo(
    agents: list,
    mode: str,
    games_per_pair: int,
    seed: int = 0,
    max_steps: int = 500,
    k: float = 24.0,
) -> dict:
    """Round-robin tournament; return {name: elo}. ELO seeded at 1000."""
    elo = {a.name: 1000.0 for a in agents}
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            res = head_to_head(
                agents[i], agents[j], mode, games_per_pair,
                seed=seed + i * 100 + j, max_steps=max_steps,
            )
            sa = res.a_rate                       # actual score for agent i
            ea = _expected(elo[agents[i].name], elo[agents[j].name])
            delta = k * (sa - ea)
            elo[agents[i].name] += delta
            elo[agents[j].name] -= delta
    return elo


# ── CLI entry ───────────────────────────────────────────────────────────────────

def run(
    mode: str = "both",
    games: int = 100,
    seed: int = 0,
    max_steps: int = 500,
    challenger: str | None = None,
    budget: float | None = None,
) -> int:
    """Default arena: current best (from file) vs greedy baseline, with CI.

    If *challenger* is a path to a weights JSON, run challenger vs current best.
    """
    modes = ["2p", "4p"] if mode == "both" else [mode]
    print(f"orbit-wars arena  modes={modes}  games={games}  seed={seed}  max_steps={max_steps}")

    for m in modes:
        best = mcts_agent_from_file(m, name=f"best_{m}")
        if budget is not None:
            best.budget_s = budget

        if challenger:
            with open(challenger) as fh:
                cw = json.load(fh)
            chal = mcts_agent(cw, name="challenger")
            if budget is not None:
                chal.budget_s = budget
            opp = best
        else:
            chal = best
            opp = greedy_agent()

        res = head_to_head(chal, opp, m, games, seed=seed, max_steps=max_steps)
        lo, hi = res.ci()
        print(
            f"\n=== {m}: {res.a_name} vs {res.b_name} ===\n"
            f"  games={res.games}  W/D/L={res.a_wins}/{res.draws}/{res.a_losses}\n"
            f"  win_rate={res.a_rate:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]\n"
            f"  {'BETTER (CI>0.5)' if lo > 0.5 else 'WORSE (CI<0.5)' if hi < 0.5 else 'INCONCLUSIVE'}"
        )

    return 0


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["2p", "4p", "both"], default="both")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=500, dest="max_steps")
    ap.add_argument("--challenger", default=None,
                    help="path to a weights JSON to test vs current best")
    ap.add_argument("--budget", type=float, default=None,
                    help="override per-turn MCTS budget (s)")
    args = ap.parse_args()
    raise SystemExit(run(
        mode=args.mode, games=args.games, seed=args.seed,
        max_steps=args.max_steps, challenger=args.challenger, budget=args.budget,
    ))


if __name__ == "__main__":
    main()
