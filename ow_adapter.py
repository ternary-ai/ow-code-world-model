"""ow_adapter.py — Orbit Wars WorldModel adapter for the smm library.

Wraps the functions in main.py (cwm_apply_joint_action, cwm_is_terminal, …)
into the smm.WorldModel protocol so you can use SMBot and CMAESTrainer with
Orbit Wars without modifying main.py.

Usage
─────
    from ow_adapter import OrbitWarsWM, ow_weight_spec, make_initial_state
    from smm import SMBot, CMAESTrainer, WeightSpec
    import time, random

    wm  = OrbitWarsWM()
    bot = SMBot(
        wm,
        num_players  = 2,
        weights      = ow_load_weights(2),
        max_depth    = 3,
        pw_c         = 2.0,
        pw_alpha     = 0.5,
        action_kwargs= {
            "k_targets": 3, "n_active_planets": 2,
            "k_reinforce": 2, "fractions": (0.25, 0.5, 0.75, 1.0),
            "right_size": True,
        },
    )

    state   = make_initial_state()
    action  = bot.choose_action(state, 0, deadline=time.monotonic() + 0.17)
    concrete = wm.to_concrete(state, 0, action)

Training example
────────────────
    spec = ow_weight_spec()
    trainer = CMAESTrainer(
        world_model     = wm,
        num_players     = 2,
        weight_spec     = spec,
        initial_state_fn= make_initial_state,
        eval_games      = 6,
        pool_size       = 5,
        popsize         = 12,
        budget_s        = 0.05,
        bot_kwargs      = {
            "max_depth": 3, "pw_c": 2.0, "pw_alpha": 0.5,
            "action_kwargs": {
                "k_targets": 3, "n_active_planets": 2,
                "k_reinforce": 2, "fractions": (0.25, 0.5, 0.75, 1.0),
                "right_size": True,
            },
        },
    )
    best = trainer.run(n_gens=100)
    print(best)
"""
from __future__ import annotations

import copy
import math
import random
from typing import Any

from main import (
    State,
    abstracted_to_concrete,
    cwm_apply_joint_action,
    cwm_is_terminal,
    cwm_value_function,
    get_action_candidates,
    _terminal_value_vec,
    _ow_load_weights,
    DEFAULT_WEIGHTS,
)
from smm import WeightSpec


# ── OrbitWarsWM ────────────────────────────────────────────────────────────────

class OrbitWarsWM:
    """Wraps main.py's Orbit Wars CWM functions into the smm.WorldModel protocol.

    The adapter is a thin shim — no game logic lives here.  All physics,
    terminal detection, and evaluation are delegated to the functions in
    main.py, which remain the single source of truth.
    """

    # ── WorldModel protocol methods ──────────────────────────────────────────

    def apply_joint_action(self, state: State, joint_action: list) -> State:
        return cwm_apply_joint_action(state, joint_action)

    def is_terminal(self, state: State) -> bool:
        return cwm_is_terminal(state)

    def terminal_values(self, state: State, num_players: int) -> list[float]:
        return _terminal_value_vec(state, num_players)

    def value_vector(
        self,
        state: State,
        num_players: int,
        weights: dict,
    ) -> list[float]:
        return [
            cwm_value_function(state, pid, weights, num_players)
            for pid in range(num_players)
        ]

    def action_candidates(
        self,
        state: State,
        player_id: int,
        k_targets: int = 4,
        n_active_planets: int = 3,
        k_reinforce: int = 0,
        fractions: tuple = (0.5, 1.0),
        target_weakness: float = 0.0,
        right_size: bool = True,
    ) -> list:
        return get_action_candidates(
            state, player_id,
            k_targets=k_targets,
            n_active_planets=n_active_planets,
            k_reinforce=k_reinforce,
            fractions=fractions,
            target_weakness=target_weakness,
            right_size=right_size,
        )

    def to_concrete(
        self,
        state: State,
        player_id: int,
        abstract_action: Any,
    ) -> list:
        if not abstract_action:
            return []
        return abstracted_to_concrete(state, player_id, abstract_action)

    def state_signature(self, state: State) -> tuple:
        """Compact hash for cross-turn subtree matching."""
        planets = tuple(sorted((p[0], p[1], round(p[5], 3)) for p in state.planets))
        fleets  = tuple(sorted((f[0], f[1], round(f[6], 3)) for f in state.fleets))
        return (state.step, planets, fleets)


# ── Convenience helpers ────────────────────────────────────────────────────────

def ow_load_weights(num_players: int) -> dict:
    """Load the hardcoded CMA-ES-tuned weights from main.py."""
    return _ow_load_weights(num_players)


def make_initial_state(
    rng: random.Random | None = None,
    num_players: int = 2,
    episode_steps: int = 200,
) -> State:
    """Build a symmetric start state for self-play training.

    Planet positions avoid the central sun so all inter-planet fleet paths
    are valid at game start.  angular_velocity=0 (static planets) eliminates
    orbit-lead complexity during training.

    episode_steps defaults to 200 (not the real 500) so training games
    terminate faster.  Pass episode_steps=500 to match real game conditions.
    """
    planets = [
        [0, 0,  22.0, 30.0, 3.0, 20, 2],   # P0 home  (top-left)
        [1, 1,  78.0, 70.0, 3.0, 20, 2],   # P1 home  (bottom-right)
        [2, -1, 22.0, 70.0, 3.0,  0, 1],   # neutral  (bottom-left)
        [3, -1, 78.0, 30.0, 3.0,  0, 1],   # neutral  (top-right)
    ]
    if num_players == 4:
        planets += [
            [4, 2,  30.0, 22.0, 3.0, 20, 2],
            [5, 3,  70.0, 78.0, 3.0, 20, 2],
        ]

    return State(
        planets          = planets,
        fleets           = [],
        initial_planets  = [list(p) for p in planets],
        comets           = [],
        comet_planet_ids = [],
        step             = 0,
        next_fleet_id    = 0,
        angular_velocity = 0.0,
        num_players      = num_players,
        episode_steps    = episode_steps,
        ship_speed       = 6.0,
        comet_speed      = 4.0,
    )


def ow_weight_spec(num_players: int = 2) -> list[WeightSpec]:
    """Return the 24-dimensional CMA-ES search space for Orbit Wars.

    Bounds and defaults are tuned to keep heuristic weights in sensible
    ranges and avoid degenerate evaluations (all-zero or all-one weights).
    The `opp_aggregation`, `max_depth`, `pw_c`, `pw_alpha`, `k_targets`,
    `n_active_planets`, `k_reinforce`, `fine_fractions`, `right_size` and
    `mcts_budget_s` hyper-parameters are treated as fixed here; they can
    be swept separately via a grid search or added as free dimensions.
    """
    base = ow_load_weights(num_players)
    return [
        # Value-function weights
        WeightSpec("w_material",        0.0, 1.0, base.get("w_material",       0.5)),
        WeightSpec("w_production",      0.0, 1.0, base.get("w_production",     0.5)),
        WeightSpec("w_control",         0.0, 1.0, base.get("w_control",        0.5)),
        WeightSpec("w_offense",         0.0, 1.0, base.get("w_offense",        0.3)),
        WeightSpec("w_cohesion",        0.0, 1.0, base.get("w_cohesion",       0.3)),
        WeightSpec("w_centrality",      0.0, 1.0, base.get("w_centrality",     0.5)),
        WeightSpec("w_threat",          0.0, 1.0, base.get("w_threat",         0.2)),
        WeightSpec("w_anti_leader",     0.0, 1.0, base.get("w_anti_leader",    0.0)),
        WeightSpec("w_neutral_access",  0.0, 1.0, base.get("w_neutral_access", 0.2)),
        WeightSpec("w_incoming_threat", 0.0, 1.0, base.get("w_incoming_threat",0.2)),
        WeightSpec("w_time_fleet",      0.0, 1.0, base.get("w_time_fleet",     0.5)),
        WeightSpec("w_prod_density",    0.0, 1.0, base.get("w_prod_density",   0.5)),
        WeightSpec("w_phase_material",  0.0, 1.0, base.get("w_phase_material", 0.4)),
        WeightSpec("w_event_fleet",     0.0, 1.0, base.get("w_event_fleet",    0.4)),
        # Action-space tuning
        WeightSpec("target_weakness",   0.0, 1.0, base.get("target_weakness",  0.0)),
    ]
