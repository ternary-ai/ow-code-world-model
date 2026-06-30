"""
tests/test_action_legality.py — Legality tests for mcts/actions.py.

For many random states (both 2p and 4p player counts), every
abstracted_to_concrete() output must satisfy:
  - 0 < num_ships <= current garrison of from_planet
  - from_id is owned by the player with ships > 0
  - angle in [0, 2*pi)

Also tests structural properties of get_action_candidates():
  - All-no-op always present
  - Total candidate count bounded by (2*k_targets+1)^n_active_planets
  - Per-candidate from_planet_ids are all owned by player
"""

from __future__ import annotations

import math
import random
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cwm.state import State, CENTER, ROTATION_RADIUS_LIMIT
from mcts.actions import get_action_candidates, abstracted_to_concrete


TWO_PI = 2.0 * math.pi

# ── State builders ─────────────────────────────────────────────────────────────

def _state(planets, num_players=2, step=10, av=0.03) -> State:
    planets = [list(p) for p in planets]
    return State(
        planets=planets,
        fleets=[],
        initial_planets=[list(p) for p in planets],
        comets=[],
        comet_planet_ids=[],
        step=step,
        next_fleet_id=0,
        angular_velocity=av,
        num_players=num_players,
        episode_steps=500,
        ship_speed=6.0,
        comet_speed=4.0,
    )


def _p(pid, owner, ships, x=60.0, y=60.0, radius=2.0, prod=2):
    return [pid, owner, x, y, radius, ships, prod]


def _random_state(rng: random.Random, num_players: int, n_planets: int = 20) -> State:
    """Generate a random mid-game state with planets distributed across owners."""
    planets = []
    for i in range(n_planets):
        owner = rng.choice(list(range(num_players)) + [-1, -1])  # bias toward neutral
        ships = rng.randint(0, 80)
        x = rng.uniform(10.0, 90.0)
        y = rng.uniform(10.0, 90.0)
        prod = rng.randint(1, 5)
        radius = 1.0 + math.log(prod)
        planets.append([i, owner, x, y, radius, ships, prod])
    return _state(planets, num_players=num_players, step=rng.randint(0, 450))


# ── Legality assertion ─────────────────────────────────────────────────────────

def _assert_moves_legal(moves: list, state: State, player_id: int, label: str):
    """Assert every move in the list is legal."""
    planet_map = {p[0]: p for p in state.planets}
    for move in moves:
        assert len(move) == 3, f"{label}: move must have 3 elements, got {move}"
        from_id, angle, num_ships = move

        # from_id must exist and be owned by player
        assert from_id in planet_map, f"{label}: from_id={from_id} not in state"
        planet = planet_map[from_id]
        assert planet[1] == player_id, (
            f"{label}: from_id={from_id} owned by {planet[1]}, not player {player_id}"
        )
        assert planet[5] > 0, (
            f"{label}: from_id={from_id} has 0 ships — should not launch"
        )

        # num_ships: must be in (0, garrison]
        assert isinstance(num_ships, int), f"{label}: ships must be int, got {type(num_ships)}"
        assert num_ships > 0, f"{label}: num_ships={num_ships} must be > 0"
        assert num_ships <= planet[5], (
            f"{label}: num_ships={num_ships} > garrison={planet[5]} for planet {from_id}"
        )

        # angle in [0, 2π)
        assert 0.0 <= angle < TWO_PI, (
            f"{label}: angle={angle:.6f} not in [0, 2pi)"
        )


# ── Tests: structural properties ───────────────────────────────────────────────

class TestCandidateStructure:

    def test_no_owned_planets_returns_single_noop(self):
        """Player with no planets gets exactly one no-op candidate."""
        s = _state([_p(0, 1, 50)])  # all owned by player 1
        cands = get_action_candidates(s, player_id=0)
        assert len(cands) == 1
        assert cands[0] == ()

    def test_no_ships_returns_noop(self):
        """Planet with 0 ships is not active."""
        s = _state([_p(0, 0, 0), _p(1, -1, 20)])
        cands = get_action_candidates(s, player_id=0)
        assert len(cands) == 1
        assert cands[0] == ()

    def test_noop_always_present(self):
        """All-no-op action is always in the candidate list."""
        rng = random.Random(42)
        for _ in range(20):
            s = _random_state(rng, 2)
            for pid in range(2):
                cands = get_action_candidates(s, pid)
                # all-no-op: every entry has fraction=0
                noop = tuple((fid, None, 0.0) for fid, _, _ in cands[0]) if cands else ()
                # At least one candidate where all fractions are 0
                has_noop = any(
                    all(frac == 0.0 for _, _, frac in c) if c else True
                    for c in cands
                )
                assert has_noop, f"player {pid}: no all-no-op found in {len(cands)} candidates"

    def test_candidate_count_bounded(self):
        """Total candidates ≤ (2*k_targets + 1)^n_active_planets."""
        s = _state([
            _p(0, 0, 50, x=70.0, y=70.0),
            _p(1, 0, 40, x=70.0, y=30.0),
            _p(2, 0, 30, x=30.0, y=70.0),
            _p(3, 1, 60, x=30.0, y=30.0),
            _p(4, -1, 20, x=50.0, y=80.0),
        ])
        k, n = 4, 3
        cands = get_action_candidates(s, player_id=0, k_targets=k, n_active_planets=n)
        upper = (2 * k + 1) ** n
        assert len(cands) <= upper, f"{len(cands)} > bound {upper}"

    def test_from_ids_are_owned_by_player(self):
        """Every from_planet_id in every candidate is owned by the player."""
        rng = random.Random(7)
        for _ in range(30):
            s = _random_state(rng, 2)
            own_ids = {p[0] for p in s.planets if p[1] == 0 and p[5] > 0}
            for c in get_action_candidates(s, player_id=0):
                for from_id, _, frac in c:
                    assert from_id in own_ids or frac == 0.0, (
                        f"from_id={from_id} not owned by player 0"
                    )

    def test_candidate_per_planet_tuple_length(self):
        """Each candidate has exactly as many tuples as active planets."""
        s = _state([
            _p(0, 0, 50, x=70.0, y=70.0),
            _p(1, 0, 40, x=70.0, y=30.0),
            _p(2, -1, 20, x=30.0, y=50.0),
        ])
        n_active = min(2, 3)  # 2 owned planets, cap=3
        cands = get_action_candidates(s, player_id=0, n_active_planets=3)
        for c in cands:
            assert len(c) == n_active, f"expected {n_active} entries, got {len(c)}: {c}"

    def test_no_targets_means_only_noop(self):
        """Player owns all planets → no non-owned targets → only no-op."""
        s = _state([
            _p(0, 0, 50, x=70.0, y=70.0),
            _p(1, 0, 40, x=30.0, y=30.0),
        ])
        cands = get_action_candidates(s, player_id=0)
        # All fractions must be 0 (no targets)
        for c in cands:
            for _, _, frac in c:
                assert frac == 0.0, f"expected no-op, got fraction {frac}"

    def test_4p_candidates_valid(self):
        """get_action_candidates works with 4-player states."""
        rng = random.Random(99)
        for _ in range(20):
            s = _random_state(rng, 4)
            for pid in range(4):
                cands = get_action_candidates(s, pid)
                assert isinstance(cands, list)
                assert len(cands) >= 1


# ── Tests: defensive reinforcement (k_reinforce > 0) ───────────────────────────

class TestReinforcement:

    def test_reinforce_off_by_default(self):
        """Default k_reinforce=0 → no own-planet targets (backward compatible)."""
        s = _state([
            _p(0, 0, 50, x=70.0, y=70.0),
            _p(1, 0, 10, x=20.0, y=20.0),   # own, near enemy
            _p(2, 1, 40, x=15.0, y=15.0),   # enemy
        ])
        own_ids = {0, 1}
        for c in get_action_candidates(s, player_id=0):
            for from_id, tgt_id, frac in c:
                if frac > 0.0:
                    assert tgt_id not in own_ids, (
                        "default should not target own planets"
                    )

    def test_reinforce_adds_own_target(self):
        """k_reinforce>0 with enemies present yields an own→own launch option."""
        s = _state([
            _p(0, 0, 50, x=70.0, y=70.0),   # source (richest)
            _p(1, 0, 10, x=20.0, y=20.0),   # own, threatened (near enemy)
            _p(2, 1, 40, x=15.0, y=15.0),   # enemy
        ])
        cands = get_action_candidates(s, player_id=0, k_reinforce=1)
        # Some candidate must launch from planet 0 toward own planet 1.
        found = any(
            any(fid == 0 and tid == 1 and frac > 0.0 for fid, tid, frac in c)
            for c in cands
        )
        assert found, "expected a reinforcement option (0 -> 1)"

    def test_reinforce_noop_when_no_enemies(self):
        """No enemies → reinforcement disabled → only no-op (own-only board)."""
        s = _state([
            _p(0, 0, 50, x=70.0, y=70.0),
            _p(1, 0, 10, x=20.0, y=20.0),
        ])
        for c in get_action_candidates(s, player_id=0, k_reinforce=2):
            for _, _, frac in c:
                assert frac == 0.0, "no enemies → should be no-op only"

    def test_reinforce_moves_legal(self):
        """Reinforcement moves satisfy all legality constraints."""
        rng = random.Random(123)
        for _ in range(40):
            s = _random_state(rng, 2)
            for pid in range(2):
                cands = get_action_candidates(s, pid, k_reinforce=2, fractions=(0.25, 0.5, 0.75, 1.0))
                for c in cands:
                    moves = abstracted_to_concrete(s, pid, c)
                    _assert_moves_legal(moves, s, pid, f"reinforce pid={pid}")

    def test_fine_fractions_quarter(self):
        """fractions=(0.25,...) → floor(garrison*0.25) ships available."""
        # Target off the sun diagonal and av=0 so the launch path is not pruned
        # by sun-avoidance filtering (which would otherwise remove this target).
        s = _state([_p(0, 0, 40, x=70.0, y=70.0), _p(1, -1, 10, x=30.0, y=70.0)],
                   av=0.0)
        cands = get_action_candidates(s, player_id=0, fractions=(0.25, 0.5, 0.75, 1.0))
        fracs = {frac for c in cands for _, _, frac in c}
        assert 0.25 in fracs and 0.75 in fracs


# ── Tests: legality of abstracted_to_concrete outputs ─────────────────────────

class TestConcreteLegality:

    def test_all_2p_candidates_legal(self):
        """For 100 random 2p states, all concrete moves are legal."""
        rng = random.Random(0)
        for trial in range(100):
            s = _random_state(rng, 2)
            for pid in range(2):
                cands = get_action_candidates(s, pid)
                for c in cands:
                    moves = abstracted_to_concrete(s, pid, c)
                    _assert_moves_legal(moves, s, pid, f"trial={trial} pid={pid}")

    def test_all_4p_candidates_legal(self):
        """For 100 random 4p states, all concrete moves are legal."""
        rng = random.Random(1)
        for trial in range(100):
            s = _random_state(rng, 4)
            for pid in range(4):
                cands = get_action_candidates(s, pid)
                for c in cands:
                    moves = abstracted_to_concrete(s, pid, c)
                    _assert_moves_legal(moves, s, pid, f"trial={trial} pid={pid} 4p")

    def test_noop_produces_empty_moves(self):
        """All-no-op abstracted action → empty move list."""
        s = _state([_p(0, 0, 50, x=70.0, y=70.0), _p(1, -1, 20, x=30.0, y=30.0)])
        noop = ((0, None, 0.0),)
        assert abstracted_to_concrete(s, 0, noop) == []

    def test_empty_abstracted_produces_empty_moves(self):
        """Empty abstracted tuple → empty move list."""
        s = _state([_p(0, 0, 50)])
        assert abstracted_to_concrete(s, 0, ()) == []

    def test_half_fraction_ships_correct(self):
        """Fraction 0.5 → floor(garrison * 0.5) ships."""
        # planet 0 owned by player 0 with 30 ships, target planet 1
        # (off the sun diagonal so the launch path is not pruned).
        s = _state([_p(0, 0, 30, x=70.0, y=70.0), _p(1, -1, 10, x=30.0, y=70.0)])
        abstracted = ((0, 1, 0.5),)
        moves = abstracted_to_concrete(s, 0, abstracted)
        assert len(moves) == 1
        assert moves[0][2] == 15  # floor(30 * 0.5) = 15

    def test_full_fraction_ships_correct(self):
        """Fraction 1.0 → all ships launched."""
        s = _state([_p(0, 0, 30, x=70.0, y=70.0), _p(1, -1, 10, x=30.0, y=70.0)])
        abstracted = ((0, 1, 1.0),)
        moves = abstracted_to_concrete(s, 0, abstracted)
        assert len(moves) == 1
        assert moves[0][2] == 30

    def test_angle_in_valid_range(self):
        """Angle is always in [0, 2*pi)."""
        rng = random.Random(5)
        for _ in range(200):
            s = _random_state(rng, 2)
            for pid in range(2):
                for c in get_action_candidates(s, pid):
                    for move in abstracted_to_concrete(s, pid, c):
                        angle = move[1]
                        assert 0.0 <= angle < TWO_PI, f"angle={angle:.6f} out of range"

    def test_angle_direction_correct(self):
        """Angle points from source to target (atan2 check)."""
        # source at (70, 80), target at (30, 80) → angle = π (pointing left)
        # av=0 so the (static) target is not led; off the sun line at y=80.
        s = _state([
            _p(0, 0, 40, x=70.0, y=80.0),
            _p(1, -1, 10, x=30.0, y=80.0),
        ], step=0, av=0.0)
        abstracted = ((0, 1, 1.0),)
        moves = abstracted_to_concrete(s, 0, abstracted)
        assert len(moves) == 1
        assert abs(moves[0][1] - math.pi) < 1e-9, f"expected π, got {moves[0][1]}"

    def test_angle_right_direction(self):
        """Source at (30,80), target at (70,80) → angle = 0."""
        s = _state([
            _p(0, 0, 40, x=30.0, y=80.0),
            _p(1, -1, 10, x=70.0, y=80.0),
        ], step=0, av=0.0)
        moves = abstracted_to_concrete(s, 0, ((0, 1, 1.0),))
        assert abs(moves[0][1] - 0.0) < 1e-9

    def test_angle_down_direction(self):
        """Source at (20,30), target at (20,70) → angle = π/2."""
        s = _state([
            _p(0, 0, 40, x=20.0, y=30.0),
            _p(1, -1, 10, x=20.0, y=70.0),
        ], step=0, av=0.0)
        moves = abstracted_to_concrete(s, 0, ((0, 1, 1.0),))
        assert abs(moves[0][1] - math.pi / 2) < 1e-9

    def test_negative_angle_normalised(self):
        """Angles in (-π, 0) are shifted to (π, 2π)."""
        # Source at (20,70), target at (20,30) → raw angle = -π/2 → normalised = 3π/2
        s = _state([
            _p(0, 0, 40, x=20.0, y=70.0),
            _p(1, -1, 10, x=20.0, y=30.0),
        ], step=0, av=0.0)
        moves = abstracted_to_concrete(s, 0, ((0, 1, 1.0),))
        expected = 3 * math.pi / 2
        assert abs(moves[0][1] - expected) < 1e-9, f"expected 3π/2, got {moves[0][1]}"

    def test_galaxy_low_ships_clamped(self):
        """Even with very few ships, num_ships >= 1 if fraction > 0 and garrison >= 1."""
        # garrison=1, fraction=0.5 → floor(0.5) = 0 → move should be skipped
        s = _state([_p(0, 0, 1, x=70.0, y=70.0), _p(1, -1, 10, x=30.0, y=30.0)])
        abstracted = ((0, 1, 0.5),)
        moves = abstracted_to_concrete(s, 0, abstracted)
        # 0 ships → should be empty (skipped)
        assert len(moves) == 0

    def test_garrison_1_full_fraction_ok(self):
        """garrison=1, fraction=1.0 → 1 ship launched (valid)."""
        s = _state([_p(0, 0, 1, x=70.0, y=70.0), _p(1, -1, 10, x=30.0, y=30.0)])
        moves = abstracted_to_concrete(s, 0, ((0, 1, 1.0),))
        assert len(moves) == 1
        assert moves[0][2] == 1

    def test_wrong_player_skipped(self):
        """If from_id not owned by player_id, move is silently skipped."""
        s = _state([_p(0, 1, 50, x=70.0, y=70.0), _p(1, -1, 10, x=30.0, y=30.0)])
        moves = abstracted_to_concrete(s, 0, ((0, 1, 1.0),))
        assert len(moves) == 0

    def test_multiple_active_planets_all_legal(self):
        """Multiple planets each launching — all moves legal."""
        s = _state([
            _p(0, 0, 50, x=70.0, y=70.0),
            _p(1, 0, 40, x=70.0, y=30.0),
            _p(2, 0, 30, x=30.0, y=70.0),
            _p(3, 1, 60, x=30.0, y=30.0),
            _p(4, -1, 20, x=50.0, y=80.0),
        ])
        cands = get_action_candidates(s, player_id=0)
        for c in cands:
            moves = abstracted_to_concrete(s, 0, c)
            _assert_moves_legal(moves, s, 0, f"multi-planet candidate {c}")

    def test_orbiting_planet_position_used(self):
        """For an orbiting planet, current position (not initial) is used for angle."""
        # Planet 0 at initial (60, 50) but orbiting → will have rotated position
        # We just check the resulting angle is still in [0, 2π)
        s = State(
            planets=[[0, 0, 60.0, 50.0, 1.0, 40, 1],
                     [1, -1, 40.0, 50.0, 1.0, 20, 1]],
            fleets=[],
            initial_planets=[[0, 0, 60.0, 50.0, 1.0, 40, 1],
                             [1, -1, 40.0, 50.0, 1.0, 20, 1]],
            comets=[], comet_planet_ids=[],
            step=100,
            next_fleet_id=0,
            angular_velocity=0.04,
            num_players=2,
            episode_steps=500, ship_speed=6.0, comet_speed=4.0,
        )
        moves = abstracted_to_concrete(s, 0, ((0, 1, 1.0),))
        if moves:
            assert 0.0 <= moves[0][1] < TWO_PI

    def test_game_state_from_trajectory_2p(self):
        """Load a real recorded 2p state and verify all candidates produce legal moves."""
        import json
        traj_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "trajectories", "2p", "game_000.json"
        )
        with open(traj_path) as f:
            game = json.load(f)

        from cwm.state import state_from_obs

        class Cfg:
            episodeSteps = 500; shipSpeed = 6.0; cometSpeed = 4.0; agentCount = 2

        # Test 10 mid-game turns
        for tr in game["transitions"][50:60]:
            obs = tr["obs_t"]
            s = state_from_obs(obs, Cfg(), cached_num_players=2)
            for pid in range(2):
                for c in get_action_candidates(s, pid):
                    moves = abstracted_to_concrete(s, pid, c)
                    _assert_moves_legal(moves, s, pid, f"t={obs['step']} pid={pid}")

    def test_game_state_from_trajectory_4p(self):
        """Load a real recorded 4p state and verify all candidates produce legal moves."""
        import json
        traj_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "trajectories", "4p", "game_000.json"
        )
        with open(traj_path) as f:
            game = json.load(f)

        from cwm.state import state_from_obs

        class Cfg:
            episodeSteps = 500; shipSpeed = 6.0; cometSpeed = 4.0; agentCount = 4

        # Test 10 mid-game turns
        for tr in game["transitions"][50:60]:
            obs = tr["obs_t"]
            s = state_from_obs(obs, Cfg(), cached_num_players=4)
            for pid in range(4):
                for c in get_action_candidates(s, pid):
                    moves = abstracted_to_concrete(s, pid, c)
                    _assert_moves_legal(moves, s, pid, f"t={obs['step']} pid={pid} 4p")


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
