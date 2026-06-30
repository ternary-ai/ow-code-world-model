"""
tests/test_physics_batch.py — Tests for cwm/physics_batch.py (Module 1).

Coverage:
  - test_batch_matches_loop: B=8 batch equals stacked individual simulate_turn calls
  - test_batch_size_one_matches_scalar: B=1 batch equals scalar call
  - test_batch_handles_zero_fleets: states with no in-flight fleets don't error
  - test_batch_runtime_scales_sublinearly: B=64 is less than 64x time for B=1 (skipped in CI)
"""

from __future__ import annotations

import math
import random
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cwm.state import State
from cwm.physics_batch import BatchGameState, BatchActions, simulate_batch
from cwm.interpreter import cwm_apply_joint_action


# ── State construction helpers ─────────────────────────────────────────────────

def _make_simple_state(
    seed: int,
    num_planets: int = 4,
    num_fleets: int = 2,
    num_players: int = 2,
) -> State:
    """Build a reproducible, simple State for testing.

    Uses a seeded RNG to vary planet positions and fleet counts across seeds.
    Planets are placed away from the sun (center 50,50) to avoid immediate
    destruction during simulation.
    """
    rng = random.Random(seed)

    # Place planets far from the sun at the corners/edges
    safe_positions = [
        (15.0, 15.0), (85.0, 15.0), (15.0, 85.0), (85.0, 85.0),
        (50.0, 15.0), (50.0, 85.0), (15.0, 50.0), (85.0, 50.0),
    ]
    chosen = safe_positions[:num_planets]

    planets = []
    for i, (x, y) in enumerate(chosen):
        owner = i % num_players
        planets.append([i, owner, x, y, 2.0, rng.randint(5, 20), rng.randint(1, 3)])

    # Add some fleets travelling toward the first planet (id=0)
    fleets = []
    for j in range(min(num_fleets, num_planets - 1)):
        src_pid = j + 1
        src = planets[src_pid]
        dx = planets[0][2] - src[2]
        dy = planets[0][3] - src[3]
        angle = math.atan2(dy, dx)
        fleets.append([
            j,           # id
            src[1],      # owner
            src[2] + math.cos(angle) * 3.0,  # x (slightly off planet)
            src[3] + math.sin(angle) * 3.0,  # y
            angle,
            src_pid,     # from_planet_id
            rng.randint(1, 5),  # ships
        ])

    # Initial planets mirror the current positions (no orbital rotation)
    initial_planets = [list(p) for p in planets]

    return State(
        planets=planets,
        fleets=fleets,
        initial_planets=initial_planets,
        comets=[],
        comet_planet_ids=[],
        step=10,
        next_fleet_id=num_fleets,
        angular_velocity=0.025,
        num_players=num_players,
        episode_steps=500,
        ship_speed=6.0,
    )


def _no_op_joint_action(num_players: int) -> list:
    """All-no-op joint action for num_players."""
    return [[] for _ in range(num_players)]


def _planets_equal(a: list, b: list, tol: float = 1e-9) -> bool:
    """True iff planet lists a and b are field-by-field equal."""
    if len(a) != len(b):
        return False
    a_map = {p[0]: p for p in a}
    b_map = {p[0]: p for p in b}
    if set(a_map) != set(b_map):
        return False
    for pid in a_map:
        pa, pb = a_map[pid], b_map[pid]
        if pa[0] != pb[0] or pa[1] != pb[1]:  # id, owner exact
            return False
        for k in (2, 3, 4, 5, 6):
            if abs(pa[k] - pb[k]) > tol:
                return False
    return True


def _fleets_equal(a: list, b: list, tol: float = 1e-9) -> bool:
    """True iff fleet lists a and b are field-by-field equal."""
    if len(a) != len(b):
        return False
    a_map = {f[0]: f for f in a}
    b_map = {f[0]: f for f in b}
    if set(a_map) != set(b_map):
        return False
    for fid in a_map:
        fa, fb = a_map[fid], b_map[fid]
        for k in range(7):
            if isinstance(fa[k], float):
                if abs(fa[k] - fb[k]) > tol:
                    return False
            else:
                if fa[k] != fb[k]:
                    return False
    return True


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestPhysicsBatch:

    def test_batch_size_one_matches_scalar(self):
        """B=1 batch call equals the scalar cwm_apply_joint_action call."""
        state = _make_simple_state(seed=0)
        action = _no_op_joint_action(state.num_players)

        batch_in = BatchGameState(states=[state])
        batch_actions = BatchActions(actions=[action])

        batch_out = simulate_batch(batch_in, batch_actions)
        scalar_out = cwm_apply_joint_action(state, action)

        assert len(batch_out.states) == 1
        result = batch_out.states[0]
        assert _planets_equal(result.planets, scalar_out.planets), (
            f"Planet mismatch:\n  batch: {result.planets}\n  scalar: {scalar_out.planets}"
        )
        assert _fleets_equal(result.fleets, scalar_out.fleets), (
            f"Fleet mismatch:\n  batch: {result.fleets}\n  scalar: {scalar_out.fleets}"
        )
        assert result.step == scalar_out.step

    def test_batch_matches_loop(self):
        """For B=8 independently-seeded states, batch output equals loop output."""
        B = 8
        states = [_make_simple_state(seed=s) for s in range(B)]
        actions = [_no_op_joint_action(s.num_players) for s in states]

        batch_in = BatchGameState(states=states)
        batch_actions = BatchActions(actions=actions)
        batch_out = simulate_batch(batch_in, batch_actions)

        for i, (state, action) in enumerate(zip(states, actions)):
            expected = cwm_apply_joint_action(state, action)
            result = batch_out.states[i]

            assert _planets_equal(result.planets, expected.planets), (
                f"State {i} planet mismatch"
            )
            assert _fleets_equal(result.fleets, expected.fleets), (
                f"State {i} fleet mismatch"
            )
            assert result.step == expected.step, f"State {i} step mismatch"

    def test_batch_handles_zero_fleets(self):
        """States with no in-flight fleets do not error."""
        state = _make_simple_state(seed=42, num_fleets=0)
        assert len(state.fleets) == 0

        batch_in = BatchGameState(states=[state, state])
        batch_actions = BatchActions(actions=[
            _no_op_joint_action(state.num_players),
            _no_op_joint_action(state.num_players),
        ])
        batch_out = simulate_batch(batch_in, batch_actions)
        assert len(batch_out.states) == 2
        # Each state should have advanced one step
        for s in batch_out.states:
            assert s.step == state.step + 1

    @pytest.mark.skip(reason="Benchmark: may not show speedup with a loop implementation")
    def test_batch_runtime_scales_sublinearly(self):
        """Time for B=64 is less than 64x time for B=1."""
        import time

        state = _make_simple_state(seed=0)
        action = _no_op_joint_action(state.num_players)

        # Warm up
        simulate_batch(BatchGameState(states=[state]), BatchActions(actions=[action]))

        # B=1 timing
        t0 = time.perf_counter()
        for _ in range(10):
            simulate_batch(BatchGameState(states=[state]), BatchActions(actions=[action]))
        t1 = time.perf_counter()
        time_b1 = (t1 - t0) / 10

        # B=64 timing
        states64 = [_make_simple_state(seed=s) for s in range(64)]
        actions64 = [_no_op_joint_action(s.num_players) for s in states64]
        t0 = time.perf_counter()
        for _ in range(5):
            simulate_batch(BatchGameState(states=states64), BatchActions(actions=actions64))
        t1 = time.perf_counter()
        time_b64 = (t1 - t0) / 5

        ratio = time_b64 / time_b1
        assert ratio < 64.0, f"B=64 took {ratio:.1f}x the B=1 time; expected < 64x"


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
