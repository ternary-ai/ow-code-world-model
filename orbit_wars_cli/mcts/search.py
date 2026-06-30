"""
mcts/search.py — MCTS agent for Orbit Wars.

joint_action_mcts(state, player_id, config, time_budget_s,
                  weights, num_players, rng, deadline) -> abstracted_action

Strategy
--------
A simultaneous-move MCTS (SM-MCTS) tree with **decoupled UCB** at every node:
each player keeps an independent UCB table over its own action candidates, and
the joint action is the per-player UCB selection. Each player maximises its own
component of the leaf value vector — in 2p this is exactly zero-sum/adversarial
(v0 + v1 ≈ 1); in 4p each opponent plays a best response to the current tree.

Multi-step lookahead
--------------------
Simulations descend the tree up to `max_depth` plies, expanding one new node per
simulation and evaluating leaves with cwm_value_function (terminal states use
exact win/draw/loss = 1.0/0.5/0.0, matching the competition's ELO scoring).

Progressive widening
--------------------
To keep the large per-player branching factor (≤729) tractable under a 1 s
budget, each node reveals candidates gradually: the number of considered
actions grows as ceil(pw_c * visits**pw_alpha), ordered by a fast
production/distance prior so promising moves are searched first.

Subtree reuse
-------------
The root tree is cached per player_id between turns. On the next turn the cached
child whose state signature matches the newly observed state is promoted to root,
warm-starting search with previously accumulated statistics.

Wall-clock budget
-----------------
time_budget_s sets a relative budget; `deadline` (absolute time.monotonic())
overrides it and is anchored by callers to the true turn start so total per-turn
time respects the competition limit regardless of setup cost.

Tunable via the weights dict: 'max_depth', 'pw_c', 'pw_alpha'.
"""
from __future__ import annotations

import math
import random
import time

from cwm.state import State
from cwm.interpreter import cwm_apply_joint_action, cwm_is_terminal, cwm_get_rewards
from cwm.symmetry import canonical_transform_id, apply_transform
from cwm.opponent_model import ArchetypeModel, classify_opponent, sample_opponent_action
from mcts.actions import get_action_candidates, abstracted_to_concrete
from mcts.value_fn import cwm_value_function, DEFAULT_WEIGHTS

_UCB_C = math.sqrt(2)   # UCB1 exploration constant

# Multi-step lookahead and progressive-widening defaults (override via weights).
DEFAULT_MAX_DEPTH = 3
DEFAULT_PW_C      = 4.0
DEFAULT_PW_ALPHA  = 0.5

# Per-player cached search tree, keyed by player_id, for cross-turn subtree reuse.
_TREE_CACHE: dict = {}

# ── Archetype model registry (Module 7) ───────────────────────────────────────
# Callers can register a fitted ArchetypeModel per opponent player_id via
# set_archetype_model().  When set, opponent_action_from_archetype() draws
# archetype-biased concrete moves instead of the generic greedy policy.

_ARCHETYPE_MODELS: dict = {}     # player_id -> ArchetypeModel
_ARCHETYPE_HISTORIES: dict = {}  # player_id -> list[observed_actions]


def set_archetype_model(
    player_id: int,
    model: ArchetypeModel,
    observed_history: list | None = None,
) -> None:
    """Register an archetype model for opponent *player_id* (Module 7)."""
    _ARCHETYPE_MODELS[player_id] = model
    _ARCHETYPE_HISTORIES[player_id] = observed_history or []


def opponent_action_from_archetype(
    state: State,
    player_id: int,
    rng: random.Random | None = None,
) -> list | None:
    """Return archetype-sampled concrete moves for *player_id*, or None if no model.

    Used by external callers (e.g. arena / validate) as a drop-in replacement
    for the greedy policy when replay-log data is available.
    """
    model = _ARCHETYPE_MODELS.get(player_id)
    if model is None:
        return None
    import numpy as np
    history = _ARCHETYPE_HISTORIES.get(player_id, [])
    probs = classify_opponent(history, model)
    seed = rng.randint(0, 2 ** 31) if rng is not None else 0
    np_rng = np.random.default_rng(seed)
    return sample_opponent_action(state, probs, model, np_rng)


# ── Heuristic policy for 4p opponent sampling ─────────────────────────────────

def _greedy_abstract(state: State, player_id: int) -> tuple:
    """O(N_planets) greedy: send half ships from richest owned planet toward
    the highest production/distance non-owned planet.

    Returns an abstracted action tuple (one entry) or () for no-op.
    """
    own = [p for p in state.planets if p[1] == player_id and p[5] > 1]
    if not own:
        return ()
    src = max(own, key=lambda p: p[5])
    non_own = [p for p in state.planets if p[1] != player_id]
    if not non_own:
        return ()
    sx, sy = src[2], src[3]
    tgt = max(
        non_own,
        key=lambda t: t[6] / (math.sqrt((t[2] - sx) ** 2 + (t[3] - sy) ** 2) + 1e-9),
    )
    return ((src[0], tgt[0], 0.5),)


# ── Leaf evaluation ────────────────────────────────────────────────────────────

def _terminal_value_vec(state: State, num_players: int) -> list:
    """Exact reward vector for a terminal state: 1.0 win / 0.5 draw / 0.0 loss
    per player, matching the competition's ELO scoring (only rank-1 = win)."""
    rewards = cwm_get_rewards(state)
    vec = [0.0] * num_players
    for pid in range(num_players):
        others = [rewards[o] for o in range(num_players) if o != pid]
        best_other = max(others) if others else 0.0
        if rewards[pid] > best_other:
            vec[pid] = 1.0
        elif rewards[pid] == best_other:
            vec[pid] = 0.5
        else:
            vec[pid] = 0.0
    return vec


def _heuristic_value_vec(state: State, num_players: int, weights: dict) -> list:
    """Per-player heuristic value vector, evaluated in the canonical orientation.

    Applies the 4-fold symmetry canonicalization (Module 6) before calling the
    value function so that equivalent board positions share the same evaluation
    regardless of which quadrant the home planet occupies.
    """
    tid = canonical_transform_id(state, observing_player=0)
    if tid != 0:
        state = apply_transform(state, tid)
    return [
        cwm_value_function(state, pid, weights, num_players)
        for pid in range(num_players)
    ]


# ── Candidate prior ordering ────────────────────────────────────────────────────

def _order_by_prior(state: State, candidates: list,
                    target_weakness: float = 0.0) -> list:
    """Order abstracted candidates by a fast production/distance prior so that
    progressive widening reveals the most promising moves first.

    Score of a candidate = Σ over its launch entries of
        target_production / ((source→target distance + 1)
                             * (1 + target_weakness * target_garrison)) * fraction.
    *target_weakness* (∈ [0, 1], default 0) biases the prior toward
    weakly-defended targets, mirroring get_action_candidates; 0 reproduces the
    original garrison-blind prior. No-op entries contribute nothing. Ties keep
    original (stable) order.
    """
    if len(candidates) <= 1:
        return list(candidates)
    pmap = {p[0]: p for p in state.planets}

    def score(cand: tuple) -> float:
        s = 0.0
        for entry in cand:
            from_id, tgt_id, frac = entry
            if tgt_id is None:
                continue
            src = pmap.get(from_id)
            tgt = pmap.get(tgt_id)
            if src is None or tgt is None:
                continue
            # A right-sized "fit" launch (non-numeric fraction) is a high-value
            # precise capture; weight it like a full commitment for ordering.
            f = frac if isinstance(frac, (int, float)) else 1.0
            d = math.hypot(tgt[2] - src[2], tgt[3] - src[3])
            s += tgt[6] / ((d + 1.0) * (1.0 + target_weakness * tgt[5])) * f
        return s

    return sorted(candidates, key=score, reverse=True)


# ── Search tree node ────────────────────────────────────────────────────────────

class _Node:
    """A simultaneous-move search node with decoupled per-player UCB tables."""

    __slots__ = ("state", "terminal", "num_players", "ordered", "stats",
                 "children", "n")

    def __init__(self, state: State, num_players: int, act_params: tuple = (4, 3, 0, (0.5, 1.0), 0.0, False)):
        self.state = state
        self.num_players = num_players
        self.terminal = cwm_is_terminal(state)
        self.n = 0
        self.children: dict = {}
        if self.terminal:
            self.ordered = None
            self.stats = None
        else:
            k_targets, n_active, k_reinforce, fractions, target_weakness, right_size = act_params
            # Per-player candidate lists (prior-ordered) and UCB stats.
            self.ordered = [
                _order_by_prior(state, get_action_candidates(
                    state, pid,
                    k_targets=k_targets,
                    n_active_planets=n_active,
                    k_reinforce=k_reinforce,
                    fractions=fractions,
                    target_weakness=target_weakness,
                    right_size=right_size,
                ), target_weakness)
                for pid in range(num_players)
            ]
            # action -> [value_sum, visits]
            self.stats = [dict() for _ in range(num_players)]


def _select_action(node: _Node, pid: int, pw_c: float, pw_alpha: float) -> tuple:
    """Decoupled-UCB action choice for one player with progressive widening."""
    ordered = node.ordered[pid]
    k = len(ordered)
    allowed = max(1, min(k, math.ceil(pw_c * (node.n + 1) ** pw_alpha)))
    stats = node.stats[pid]
    log_n = math.log(node.n + 1)
    best = ordered[0]
    best_score = -math.inf
    for a in ordered[:allowed]:
        st = stats.get(a)
        if st is None or st[1] == 0:
            return a                     # explore unvisited (prior order) first
        ucb = st[0] / st[1] + _UCB_C * math.sqrt(log_n / st[1])
        if ucb > best_score:
            best_score = ucb
            best = a
    return best


def _value_vec(node: _Node, num_players: int, weights: dict) -> list:
    """Evaluate a freshly created node: exact if terminal, else heuristic."""
    if node.terminal:
        return _terminal_value_vec(node.state, num_players)
    return _heuristic_value_vec(node.state, num_players, weights)


def _simulate(node: _Node, num_players: int, weights: dict,
              depth: int, max_depth: int, pw_c: float, pw_alpha: float,
              act_params: tuple) -> list:
    """One MCTS simulation from *node*; returns the leaf value vector."""
    if node.terminal:
        return _terminal_value_vec(node.state, num_players)
    if depth >= max_depth:
        return _heuristic_value_vec(node.state, num_players, weights)

    actions = tuple(
        _select_action(node, pid, pw_c, pw_alpha)
        for pid in range(num_players)
    )
    child = node.children.get(actions)
    if child is None:
        joint = [
            abstracted_to_concrete(node.state, pid, actions[pid])
            for pid in range(num_players)
        ]
        next_s = cwm_apply_joint_action(node.state, joint)
        child = _Node(next_s, num_players, act_params)
        node.children[actions] = child
        value = _value_vec(child, num_players, weights)   # evaluate new leaf
    else:
        value = _simulate(child, num_players, weights,
                          depth + 1, max_depth, pw_c, pw_alpha, act_params)

    # ── Backpropagate into this node's per-player tables ──────────────────────
    node.n += 1
    for pid in range(num_players):
        st = node.stats[pid].get(actions[pid])
        if st is None:
            node.stats[pid][actions[pid]] = [value[pid], 1]
        else:
            st[0] += value[pid]
            st[1] += 1
    return value


# ── Cross-turn subtree reuse ────────────────────────────────────────────────────

def _state_sig(state: State) -> tuple:
    """Compact signature for matching a cached child to a newly observed state.

    Uses step plus per-planet (id, owner, ships) and per-fleet (id, owner, ships),
    which uniquely identify the post-transition position for reuse purposes.
    """
    planets = tuple(sorted((p[0], p[1], round(p[5], 3)) for p in state.planets))
    fleets = tuple(sorted((f[0], f[1], round(f[6], 3)) for f in state.fleets))
    return (state.step, planets, fleets)


def _reuse_root(player_id: int, state: State, num_players: int,
                act_params: tuple) -> _Node:
    """Return a warm-started root from cache if the observed state matches a
    cached child; otherwise a fresh node. Safe: any mismatch rebuilds."""
    cached = _TREE_CACHE.get(player_id)
    if cached is not None and not cached.terminal and cached.state.step < state.step:
        sig = _state_sig(state)
        for child in cached.children.values():
            if not child.terminal and _state_sig(child.state) == sig:
                return child            # promote matching subtree to root
    return _Node(state, num_players, act_params)


# ── Main MCTS entry point ─────────────────────────────────────────────────────

def joint_action_mcts(
    state: State,
    player_id: int,
    config=None,
    time_budget_s: float = 1.0,
    weights: dict | None = None,
    num_players: int | None = None,
    rng: random.Random | None = None,
    deadline: float | None = None,
) -> tuple:
    """Run tree-based SM-MCTS and return the best abstracted action for *player_id*.

    Parameters
    ----------
    state        : current game state (not mutated).
    player_id    : 0-indexed player we are deciding for.
    config       : unused; accepted for signature compatibility.
    time_budget_s: wall-clock seconds allowed (used only when *deadline* is None).
    weights      : value-function weight dict (see mcts.value_fn.DEFAULT_WEIGHTS).
                   May also carry MCTS knobs: 'max_depth', 'pw_c', 'pw_alpha'.
    num_players  : override if different from state.num_players.
    rng          : seeded random.Random for reproducibility (default: unseeded).
    deadline     : absolute time.monotonic() deadline. When provided it overrides
                   *time_budget_s* and bounds total search time from the caller's
                   true turn start (including this function's own setup cost),
                   guaranteeing the agent respects the per-turn wall-clock limit
                   regardless of state-construction / candidate-generation cost.

    Returns
    -------
    tuple
        Best abstracted action as produced by get_action_candidates().
    """
    if weights is None:
        weights = dict(DEFAULT_WEIGHTS)
    if num_players is None:
        num_players = state.num_players
    if rng is None:
        rng = random.Random()

    # Action-space knobs (default values reproduce the original behaviour, so
    # existing tuned weights are unaffected).
    k_targets   = int(weights.get("k_targets", 4))
    n_active    = int(weights.get("n_active_planets", 3))
    k_reinforce = int(weights.get("k_reinforce", 0))
    fractions   = ((0.25, 0.5, 0.75, 1.0) if weights.get("fine_fractions", False)
                   else (0.5, 1.0))
    target_weakness = float(weights.get("target_weakness", 0.0))
    # Right-sizing (precise capture economy) is on by default; legacy weight
    # dicts without the key opt in automatically. Set right_size=false to fall
    # back to fixed-fraction launches.
    right_size  = bool(weights.get("right_size", True))
    act_params  = (k_targets, n_active, k_reinforce, fractions, target_weakness,
                   right_size)

    candidates_self = get_action_candidates(
        state, player_id,
        k_targets=k_targets, n_active_planets=n_active,
        k_reinforce=k_reinforce, fractions=fractions,
        target_weakness=target_weakness, right_size=right_size,
    )
    if len(candidates_self) == 1:
        return candidates_self[0]       # nothing to decide

    max_depth = int(weights.get("max_depth", DEFAULT_MAX_DEPTH))
    pw_c      = float(weights.get("pw_c", DEFAULT_PW_C))
    pw_alpha  = float(weights.get("pw_alpha", DEFAULT_PW_ALPHA))

    if deadline is None:
        deadline = time.monotonic() + time_budget_s

    root = _reuse_root(player_id, state, num_players, act_params)

    while time.monotonic() < deadline:
        _simulate(root, num_players, weights, 0, max_depth, pw_c, pw_alpha,
                  act_params)

    # Cache root for next turn's subtree reuse.
    _TREE_CACHE[player_id] = root

    # Robust child: most-visited action for player_id, tie-broken by mean value.
    self_stats = root.stats[player_id]
    if not self_stats:
        return candidates_self[0]
    return max(
        self_stats.items(),
        key=lambda kv: (kv[1][1], kv[1][0] / kv[1][1]),
    )[0]
