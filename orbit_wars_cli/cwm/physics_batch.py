"""
cwm/physics_batch.py — Batch game-state simulator (Module 1).

Wraps B independent game states so they can be advanced together as a single
call, enabling ISMCTS determinizations to share Python call overhead.

Design: BatchGameState holds a plain list of State objects.  simulate_batch
delegates to cwm_apply_joint_action for each state, guaranteeing numerical
identity with the scalar simulator.  Future work: replace the per-state loop
with numpy-vectorised orbital/fleet updates for genuine sub-linear scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cwm.state import State
from cwm.interpreter import cwm_apply_joint_action


@dataclass
class BatchGameState:
    """B independent game states."""
    states: list  # list[State], length B

    @property
    def batch_size(self) -> int:
        return len(self.states)


@dataclass
class BatchActions:
    """B joint actions, one per state in a BatchGameState."""
    actions: list  # list[joint_action], length B
    # Each joint_action is a list of per-player move lists:
    # [[from_planet_id, angle, ships], ...] for each player.


def simulate_batch(states: BatchGameState, actions: BatchActions) -> BatchGameState:
    """Apply one full game tick to all B states and return B successor states.

    Results are numerically identical to calling
    ``cwm_apply_joint_action(state, action)`` once per (state, action) pair.
    """
    next_states = [
        cwm_apply_joint_action(state, action)
        for state, action in zip(states.states, actions.actions)
    ]
    return BatchGameState(states=next_states)
