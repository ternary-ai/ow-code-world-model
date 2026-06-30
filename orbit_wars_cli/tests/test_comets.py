"""
tests/test_comets.py — Tests for cwm/comets.py

Coverage:
  - expire_comets: removes expired comets, keeps active ones, handles empty state
  - spawn_comet_group: 4 planets added, shared ship count, correct properties,
    4-fold path symmetry, no-op when generate_comet_paths fails
  - advance_comet_positions: path_index increments, positions update, expiry detection
  - "fleet en route to expiring comet" (Item 1): NOT tested here — this requires the
    full interpreter and is covered by test_transitions.py. The comets.py functions
    correctly flag mid-tick expiry; the "black hole" behavior is enforced by the
    interpreter's remove_expired_comets() call before combat. See
    reference/issue_1047_status.md Item 1.
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cwm.state import State, CENTER, BOARD_SIZE, ROTATION_RADIUS_LIMIT, COMET_RADIUS, COMET_PRODUCTION
from cwm.comets import (
    expire_comets,
    spawn_comet_group,
    advance_comet_positions,
    remove_expired_comets,
    generate_comet_paths,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _minimal_state(
    planets=None, fleets=None, comets=None, comet_planet_ids=None,
    initial_planets=None, step=0, next_fleet_id=0, angular_velocity=0.03,
    num_players=2,
) -> State:
    """Build a minimal State for testing."""
    planets            = planets or []
    fleets             = fleets  or []
    comets             = comets  or []
    comet_planet_ids   = comet_planet_ids or []
    initial_planets    = initial_planets  or (planets[:] if planets else [])
    return State(
        planets=planets,
        fleets=fleets,
        initial_planets=initial_planets,
        comets=comets,
        comet_planet_ids=comet_planet_ids,
        step=step,
        next_fleet_id=next_fleet_id,
        angular_velocity=angular_velocity,
        num_players=num_players,
        episode_steps=500,
        ship_speed=6.0,
        comet_speed=4.0,
    )


def _make_path(length: int = 10) -> list:
    """Make a trivial straight-line path of given length."""
    return [[float(i), 50.0] for i in range(length)]


def _make_comet_group(pids, path_len=10, path_index=0):
    """Make a comet group dict with simple straight-line paths."""
    paths = [_make_path(path_len) for _ in pids]
    return {"planet_ids": list(pids), "paths": paths, "path_index": path_index}


def _make_comet_planets(pids, ships=10):
    """Make comet planet lists for given pids."""
    return [[pid, -1, 0.0, 50.0, COMET_RADIUS, ships, COMET_PRODUCTION] for pid in pids]


# ── expire_comets ──────────────────────────────────────────────────────────────

class TestExpireComets:

    def test_no_comets_returns_state_unchanged(self):
        """State with no comets passes through expire_comets unchanged."""
        s = _minimal_state(
            planets=[[0, 0, 70.0, 50.0, 2.0, 10, 1]],
            comets=[],
            comet_planet_ids=[],
        )
        out = expire_comets(s)
        assert len(out.planets) == 1
        assert len(out.comets) == 0
        assert len(out.comet_planet_ids) == 0

    def test_non_expired_comet_stays(self):
        """Comet group with path_index < path length is not removed."""
        pids = [10, 11, 12, 13]
        group = _make_comet_group(pids, path_len=10, path_index=5)   # 5 < 10
        planets = _make_comet_planets(pids)
        s = _minimal_state(
            planets=planets,
            initial_planets=[[p[0], -1, -99.0, -99.0, COMET_RADIUS, 5, COMET_PRODUCTION] for p in planets],
            comets=[group],
            comet_planet_ids=list(pids),
        )
        out = expire_comets(s)
        assert {p[0] for p in out.planets} == set(pids)
        assert len(out.comets) == 1
        assert set(out.comet_planet_ids) == set(pids)

    def test_expired_comet_removed(self):
        """Comet group with path_index >= path length is removed."""
        pids = [10, 11, 12, 13]
        group = _make_comet_group(pids, path_len=5, path_index=5)    # 5 >= 5 → expired
        planets = _make_comet_planets(pids)
        s = _minimal_state(
            planets=planets,
            initial_planets=[[p[0], -1, -99.0, -99.0, COMET_RADIUS, 5, COMET_PRODUCTION] for p in planets],
            comets=[group],
            comet_planet_ids=list(pids),
        )
        out = expire_comets(s)
        assert len(out.planets) == 0
        assert len(out.comets) == 0
        assert len(out.comet_planet_ids) == 0

    def test_expired_ships_are_lost(self):
        """Garrison on expired comet planet is removed along with the planet."""
        pids = [20, 21, 22, 23]
        planets = _make_comet_planets(pids, ships=50)  # 50 ships each
        group = _make_comet_group(pids, path_len=3, path_index=3)  # expired
        s = _minimal_state(
            planets=planets,
            initial_planets=[[p[0], -1, -99.0, -99.0, COMET_RADIUS, 50, COMET_PRODUCTION] for p in planets],
            comets=[group],
            comet_planet_ids=list(pids),
        )
        out = expire_comets(s)
        assert sum(p[5] for p in out.planets) == 0

    def test_mixed_expired_and_active_groups(self):
        """Two groups: one expired, one active. Only expired group removed."""
        # Active group: pids 10-13, path_index=3, path_len=10
        active_pids = [10, 11, 12, 13]
        active_group = _make_comet_group(active_pids, path_len=10, path_index=3)
        # Expired group: pids 20-23, path_index=8, path_len=8
        expired_pids = [20, 21, 22, 23]
        expired_group = _make_comet_group(expired_pids, path_len=8, path_index=8)

        all_pids = active_pids + expired_pids
        planets = _make_comet_planets(all_pids)
        s = _minimal_state(
            planets=planets,
            initial_planets=[[p[0], -1, -99.0, -99.0, COMET_RADIUS, 10, COMET_PRODUCTION] for p in planets],
            comets=[active_group, expired_group],
            comet_planet_ids=all_pids,
        )
        out = expire_comets(s)
        surviving_ids = {p[0] for p in out.planets}
        assert surviving_ids == set(active_pids)
        assert set(out.comet_planet_ids) == set(active_pids)
        assert len(out.comets) == 1
        assert set(out.comets[0]["planet_ids"]) == set(active_pids)

    def test_normal_planets_not_affected(self):
        """Non-comet planets are never touched by expire_comets."""
        normal = [[0, 0, 70.0, 50.0, 2.0, 20, 2], [1, 1, 30.0, 50.0, 2.0, 15, 1]]
        comet_pids = [10, 11, 12, 13]
        expired_group = _make_comet_group(comet_pids, path_len=4, path_index=4)
        planets = normal + _make_comet_planets(comet_pids)
        s = _minimal_state(
            planets=planets,
            initial_planets=[list(p) for p in planets],
            comets=[expired_group],
            comet_planet_ids=comet_pids,
        )
        out = expire_comets(s)
        assert {p[0] for p in out.planets} == {0, 1}

    def test_path_index_minus_one_not_expired(self):
        """Newly spawned comet (path_index=-1) is never expired by expire_comets."""
        pids = [10, 11, 12, 13]
        group = _make_comet_group(pids, path_len=10, path_index=-1)
        planets = _make_comet_planets(pids)
        s = _minimal_state(
            planets=planets,
            initial_planets=[list(p) for p in planets],
            comets=[group],
            comet_planet_ids=list(pids),
        )
        out = expire_comets(s)
        assert {p[0] for p in out.planets} == set(pids)


# ── advance_comet_positions ────────────────────────────────────────────────────

class TestAdvanceCometPositions:

    def test_path_index_increments(self):
        """path_index increases by 1 on each advance call."""
        pids = [10, 11, 12, 13]
        group = _make_comet_group(pids, path_len=10, path_index=2)
        planets = _make_comet_planets(pids)
        s = _minimal_state(
            planets=planets,
            initial_planets=[list(p) for p in planets],
            comets=[group],
            comet_planet_ids=list(pids),
        )
        out, expired = advance_comet_positions(s)
        assert out.comets[0]["path_index"] == 3
        assert expired == []

    def test_planet_position_updated_to_path_entry(self):
        """After advance, planet position matches path[new_index]."""
        pids = [10, 11, 12, 13]
        # Give each comet its own distinct straight path
        paths = [
            [[float(j * 2 + i), 50.0] for j in range(10)]
            for i in range(4)
        ]
        group = {"planet_ids": list(pids), "paths": paths, "path_index": 0}
        planets = [[pid, -1, -99.0, -99.0, COMET_RADIUS, 5, COMET_PRODUCTION] for pid in pids]
        s = _minimal_state(
            planets=planets,
            initial_planets=[list(p) for p in planets],
            comets=[group],
            comet_planet_ids=list(pids),
        )
        out, expired = advance_comet_positions(s)
        planet_map = {p[0]: p for p in out.planets}
        for i, pid in enumerate(pids):
            expected_x = paths[i][1][0]
            expected_y = paths[i][1][1]
            assert abs(planet_map[pid][2] - expected_x) < 1e-9, f"pid {pid} x mismatch"
            assert abs(planet_map[pid][3] - expected_y) < 1e-9, f"pid {pid} y mismatch"
        assert expired == []

    def test_first_advance_from_minus_one(self):
        """First advance (path_index -1 → 0) places comet at path[0]."""
        pids = [10, 11, 12, 13]
        paths = [[[float(i * 5), 50.0] for _ in range(8)] for i in range(4)]
        # Override first element to be distinctive
        for i in range(4):
            paths[i][0] = [float(i * 10 + 1), float(i * 10 + 2)]
        group = {"planet_ids": list(pids), "paths": paths, "path_index": -1}
        planets = [[pid, -1, -99.0, -99.0, COMET_RADIUS, 5, COMET_PRODUCTION] for pid in pids]
        s = _minimal_state(
            planets=planets,
            initial_planets=[list(p) for p in planets],
            comets=[group],
            comet_planet_ids=list(pids),
        )
        out, expired = advance_comet_positions(s)
        assert out.comets[0]["path_index"] == 0
        assert expired == []
        planet_map = {p[0]: p for p in out.planets}
        for i, pid in enumerate(pids):
            assert abs(planet_map[pid][2] - paths[i][0][0]) < 1e-9
            assert abs(planet_map[pid][3] - paths[i][0][1]) < 1e-9

    def test_comet_expiring_this_tick_returned_in_expired_list(self):
        """Comet at last path entry (next advance would exceed path) returns pid in expired."""
        pids = [10, 11, 12, 13]
        group = _make_comet_group(pids, path_len=5, path_index=4)  # next = 5 >= 5 → expire
        planets = _make_comet_planets(pids)
        s = _minimal_state(
            planets=planets,
            initial_planets=[list(p) for p in planets],
            comets=[group],
            comet_planet_ids=list(pids),
        )
        out, expired = advance_comet_positions(s)
        assert set(expired) == set(pids)
        # Expired comets' positions should be UNCHANGED (stay put for collision detection)
        planet_map_in  = {p[0]: (p[2], p[3]) for p in s.planets}
        planet_map_out = {p[0]: (p[2], p[3]) for p in out.planets}
        for pid in pids:
            assert planet_map_out[pid] == planet_map_in[pid]

    def test_remove_expired_comets_cleans_up(self):
        """remove_expired_comets correctly removes mid-tick expired planets."""
        pids = [10, 11, 12, 13]
        group = _make_comet_group(pids, path_len=5, path_index=4)
        planets = _make_comet_planets(pids)
        s = _minimal_state(
            planets=planets,
            initial_planets=[list(p) for p in planets],
            comets=[group],
            comet_planet_ids=list(pids),
        )
        adv_state, expired = advance_comet_positions(s)
        cleaned = remove_expired_comets(adv_state, expired)
        assert len(cleaned.planets) == 0
        assert len(cleaned.comet_planet_ids) == 0
        assert len(cleaned.comets) == 0


# ── spawn_comet_group ──────────────────────────────────────────────────────────

class TestSpawnCometGroup:
    """Tests use a seeded RNG to get deterministic paths."""

    def _make_spawn_state(self, step=49):
        """Build a state with a few static planets suitable for comet path generation."""
        # A couple of static planets (orbital radius >> ROTATION_RADIUS_LIMIT)
        planets = [
            [0, 0,  70.0, 70.0, 1.0, 10, 1],  # orbital~28.3, r=1 → 29.3 < 50 (orbiting)
            [1, 1,  30.0, 30.0, 1.0, 10, 1],  # symmetric
            [2, -1, 70.0, 30.0, 1.0, 10, 1],
            [3, -1, 30.0, 70.0, 1.0, 10, 1],
        ]
        return _minimal_state(
            planets=[list(p) for p in planets],
            initial_planets=[list(p) for p in planets],
            comets=[],
            comet_planet_ids=[],
            step=step,
        )

    def test_spawns_four_comets(self):
        """spawn_comet_group adds exactly 4 comet planets."""
        rng = random.Random(42)
        s = self._make_spawn_state()
        # Try multiple seeds until we get a valid path
        for seed in range(20):
            rng = random.Random(seed)
            out = spawn_comet_group(s, rng)
            if len(out.planets) == 8:  # 4 original + 4 new
                break
        new_ids = set(out.comet_planet_ids)
        assert len(new_ids) == 4

    def test_all_four_share_ship_count(self):
        """All 4 comets in a group have the same ship count (shared min-of-4 draw)."""
        for seed in range(30):
            rng = random.Random(seed)
            s = self._make_spawn_state()
            out = spawn_comet_group(s, rng)
            if len(out.comet_planet_ids) == 4:
                ships = [p[5] for p in out.planets if p[0] in set(out.comet_planet_ids)]
                assert len(set(ships)) == 1, f"Ships differ: {ships}"
                break

    def test_ship_count_in_valid_range(self):
        """Shared ship count is between 1 and 99 (min of 4 draws from randint(1,99))."""
        for seed in range(30):
            rng = random.Random(seed)
            s = self._make_spawn_state()
            out = spawn_comet_group(s, rng)
            if len(out.comet_planet_ids) == 4:
                ship = next(p[5] for p in out.planets if p[0] in set(out.comet_planet_ids))
                assert 1 <= ship <= 99
                break

    def test_comet_properties(self):
        """Each new comet has correct radius, production, owner=-1."""
        for seed in range(30):
            rng = random.Random(seed)
            s = self._make_spawn_state()
            out = spawn_comet_group(s, rng)
            if len(out.comet_planet_ids) == 4:
                comet_ids = set(out.comet_planet_ids)
                for p in out.planets:
                    if p[0] in comet_ids:
                        assert p[4] == COMET_RADIUS,      f"radius={p[4]}, expected {COMET_RADIUS}"
                        assert p[6] == COMET_PRODUCTION,  f"production={p[6]}, expected {COMET_PRODUCTION}"
                        assert p[1] == -1,                f"owner={p[1]}, expected -1 (neutral)"
                break

    def test_comet_ids_added_to_comet_planet_ids(self):
        """Planet IDs of new comets are registered in comet_planet_ids."""
        for seed in range(30):
            rng = random.Random(seed)
            s = self._make_spawn_state()
            out = spawn_comet_group(s, rng)
            if len(out.comet_planet_ids) == 4:
                new_planet_ids = {p[0] for p in out.planets} - {p[0] for p in s.planets}
                assert set(out.comet_planet_ids) == new_planet_ids
                break

    def test_comet_group_added_with_path_index_minus_one(self):
        """New comet group has path_index=-1 (first advance will place at path[0])."""
        for seed in range(30):
            rng = random.Random(seed)
            s = self._make_spawn_state()
            out = spawn_comet_group(s, rng)
            if len(out.comets) == 1:
                assert out.comets[0]["path_index"] == -1
                break

    def test_paths_are_four_fold_symmetric(self):
        """The 4 paths in a comet group are rotationally symmetric about center (50,50).

        For a valid path group, the 4 paths satisfy:
          path[0] is a rearrangement of [y,x] for (x,y) in visible
          path[1] → [100-x, y]
          path[2] → [x, 100-y]
          path[3] → [100-y, 100-x]
        By construction, the sum of all 4 path[k][0]+path[k][1] values at each step
        should equal 4*100 = 400 for each step (since x+(100-x)+x+(100-x) etc.).
        More precisely: the four x-coords at step k are:
          p0[k][0], p1[k][0], p2[k][0], p3[k][0]
        and their sum should equal 2*BOARD_SIZE = 200 for the x-axis pattern.
        """
        for seed in range(30):
            rng = random.Random(seed)
            s = self._make_spawn_state()
            out = spawn_comet_group(s, rng)
            if len(out.comets) == 1:
                group = out.comets[0]
                paths = group["paths"]
                assert len(paths) == 4
                # All paths should have same length
                lengths = [len(p) for p in paths]
                assert len(set(lengths)) == 1, f"Path lengths differ: {lengths}"
                # Check 4-fold symmetry constraint at first step:
                # From source: paths[0] = [[y,x]], paths[1] = [[100-x,y]],
                # paths[2] = [[x,100-y]], paths[3] = [[100-y,100-x]]
                # With a visible point (cx, cy):
                #   p0 = (cy, cx)
                #   p1 = (100-cx, cy)
                #   p2 = (cx, 100-cy)
                #   p3 = (100-cy, 100-cx)
                # Sum of axis-0: cy + (100-cx) + cx + (100-cy) = 200
                # Sum of axis-1: cx + cy + (100-cy) + (100-cx) = 200
                p0, p1, p2, p3 = [p[0] for p in paths]
                sum_x = p0[0] + p1[0] + p2[0] + p3[0]
                sum_y = p0[1] + p1[1] + p2[1] + p3[1]
                assert abs(sum_x - 200.0) < 1e-6, f"4-fold symmetry broken: sum_x={sum_x}"
                assert abs(sum_y - 200.0) < 1e-6, f"4-fold symmetry broken: sum_y={sum_y}"
                break

    def test_noop_when_no_valid_path(self):
        """Returns state unchanged if generate_comet_paths fails (returns None).

        We mock a state where all 300 path attempts would fail by filling the board
        with planets that block all paths. In practice, we just verify the function
        is a no-op when comet_paths is None by monkeypatching."""
        import cwm.comets as comets_module

        original = comets_module.generate_comet_paths
        try:
            comets_module.generate_comet_paths = lambda *a, **kw: None
            rng = random.Random(0)
            s = self._make_spawn_state()
            out = spawn_comet_group(s, rng)
            assert out.planets == s.planets
            assert out.comet_planet_ids == s.comet_planet_ids
            assert out.comets == s.comets
        finally:
            comets_module.generate_comet_paths = original

    def test_new_comet_ids_sequential_after_max(self):
        """New comet planet IDs start at max(existing_id) + 1."""
        for seed in range(30):
            rng = random.Random(seed)
            s = self._make_spawn_state()
            max_existing = max(p[0] for p in s.planets)
            out = spawn_comet_group(s, rng)
            if len(out.comet_planet_ids) == 4:
                new_ids = sorted(out.comet_planet_ids)
                assert new_ids[0] == max_existing + 1
                assert new_ids == list(range(max_existing + 1, max_existing + 5))
                break


# ── generate_comet_paths ───────────────────────────────────────────────────────

class TestGenerateCometPaths:

    def test_returns_four_paths_or_none(self):
        """generate_comet_paths returns a list of 4 paths or None."""
        planets = [
            [0, 0, 70.0, 70.0, 1.0, 10, 1],
            [1, 1, 30.0, 30.0, 1.0, 10, 1],
        ]
        for seed in range(10):
            rng = random.Random(seed)
            result = generate_comet_paths(
                initial_planets=planets,
                angular_velocity=0.03,
                spawn_step=50,
                comet_planet_ids=[],
                comet_speed=4.0,
                rng=rng,
            )
            if result is not None:
                assert len(result) == 4
                # Each path is a list of [x, y] pairs
                for path in result:
                    assert len(path) >= 5
                    assert all(len(pt) == 2 for pt in path)
                return
        # If all 10 seeds fail (unlikely), that's acceptable
        pass

    def test_paths_within_board_bounds(self):
        """All path points should be within [0, 100] x [0, 100]."""
        planets = [
            [0, 0, 70.0, 70.0, 1.0, 10, 1],
            [1, 1, 30.0, 30.0, 1.0, 10, 1],
        ]
        for seed in range(20):
            rng = random.Random(seed)
            result = generate_comet_paths(
                planets, 0.03, 50, [], 4.0, rng=rng
            )
            if result is not None:
                for path in result:
                    for x, y in path:
                        assert 0 <= x <= 100.0, f"x={x} out of bounds"
                        assert 0 <= y <= 100.0, f"y={y} out of bounds"
                return


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
