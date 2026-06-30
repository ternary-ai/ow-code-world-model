"""
tests/test_event_graph.py — Tests for cwm/event_graph.py (Module 5).

Coverage:
  - test_extract_events_excludes_beyond_horizon
  - test_extract_events_includes_boundary_turn
  - test_encode_shape_is_fixed
  - test_encode_sums_multiple_fleets_same_planet_same_turn
  - test_two_states_same_event_graph_encode_identically
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cwm.event_graph import FleetEvent, extract_events, encode_events
from cwm.state import State


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_state(
    planets: list | None = None,
    fleets: list | None = None,
    step: int = 0,
    ship_speed: float = 6.0,
) -> State:
    if planets is None:
        planets = [
            [0, 0, 15.0, 15.0, 1.0, 10, 1],
            [1, 1, 85.0, 15.0, 1.0, 10, 1],
        ]
    return State(
        planets=[list(p) for p in planets],
        fleets=[list(f) for f in (fleets or [])],
        initial_planets=[list(p) for p in planets],
        comets=[],
        comet_planet_ids=[],
        step=step,
        next_fleet_id=len(fleets) if fleets else 0,
        angular_velocity=0.025,
        num_players=2,
        episode_steps=500,
        ship_speed=ship_speed,
    )


def _fleet_pointing_at(fid, owner, src_pos, dst_pos, ships, from_pid):
    """Build a fleet [id, owner, x, y, angle, from_planet_id, ships] at src_pos."""
    angle = math.atan2(dst_pos[1] - src_pos[1], dst_pos[0] - src_pos[0])
    return [fid, owner, src_pos[0], src_pos[1], angle, from_pid, ships]


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestExtractEvents:

    def test_extract_events_excludes_beyond_horizon(self):
        """A fleet arriving after `horizon` turns is excluded."""
        # Fleet is far away — it will take many turns to arrive
        # source planet: planet 0 at (15, 15)
        # target planet: planet 1 at (85, 15)
        # distance ≈ 70 units, speed=6 → eta ≈ 70/6 ≈ 11.7 turns
        fleet = _fleet_pointing_at(0, 0, (15.0, 15.0), (85.0, 15.0),
                                   ships=5, from_pid=0)
        state = _make_state(fleets=[fleet])

        events = extract_events(state, horizon=5)
        # eta ≈ 12 > 5 → excluded
        assert len(events) == 0

    def test_extract_events_includes_boundary_turn(self):
        """A fleet arriving exactly at `horizon` turns out is included.

        fleet_speed(1000, 6.0) == 6.0 exactly.
        Distance = 18 units → ETA = 18/6 = 3 turns → eta_turn = 3.
        With horizon=3, 3-0 = 3 ≤ 3 → included.
        """
        planets = [
            [0, 0, 15.0, 15.0, 1.0, 10, 1],
            [1, 1, 85.0, 15.0, 1.0, 10, 1],
        ]
        # 1000 ships → fleet_speed == 6.0; distance from (67,15) to (85,15) = 18
        fleet = _fleet_pointing_at(0, 0, (67.0, 15.0), (85.0, 15.0),
                                   ships=1000, from_pid=0)
        state = _make_state(planets=planets, fleets=[fleet], ship_speed=6.0)

        horizon = 3
        events = extract_events(state, horizon=horizon)
        # eta = 18/6 = 3 → eta_turn = 3 → offset 3 ≤ horizon 3 → included
        assert len(events) == 1

    def test_no_fleets_returns_empty(self):
        """State with no fleets returns an empty list."""
        state = _make_state()
        assert extract_events(state, horizon=10) == []

    def test_events_sorted_by_eta(self):
        """Events are returned sorted by eta_turn ascending.

        Use 1000-ship fleets for max speed (6.0) so ETA is predictable:
        f0: distance=12 → eta=2; f1: distance=6 → eta=1. Both within horizon=10.
        """
        planets = [
            [0, 0, 15.0, 15.0, 1.0, 10, 1],
            [1, 1, 85.0, 15.0, 1.0, 10, 1],
        ]
        # f0: 12 units away → eta=2; f1: 6 units away → eta=1
        f0 = _fleet_pointing_at(0, 0, (73.0, 15.0), (85.0, 15.0), ships=1000, from_pid=0)
        f1 = _fleet_pointing_at(1, 0, (79.0, 15.0), (85.0, 15.0), ships=1000, from_pid=0)

        state = _make_state(planets=planets, fleets=[f0, f1])
        events = extract_events(state, horizon=10)
        assert len(events) == 2
        assert events[0].eta_turn <= events[1].eta_turn


class TestEncodeEvents:

    def test_encode_shape_is_fixed(self):
        """Output shape is always (horizon, num_planets, 2) regardless of event count."""
        for num_events in (0, 1, 5):
            events = [
                FleetEvent(owner=0, destination_planet_id=0, eta_turn=1, ships_arriving=5)
                for _ in range(num_events)
            ]
            arr = encode_events(events, num_planets=3, horizon=4)
            assert arr.shape == (4, 3, 2), (
                f"Expected shape (4, 3, 2) with {num_events} events, got {arr.shape}"
            )

    def test_encode_sums_multiple_fleets_same_planet_same_turn(self):
        """Two fleets of the same owner at the same planet/turn are summed."""
        events = [
            FleetEvent(owner=0, destination_planet_id=0, eta_turn=1, ships_arriving=5),
            FleetEvent(owner=0, destination_planet_id=0, eta_turn=1, ships_arriving=7),
        ]
        arr = encode_events(events, num_planets=2, horizon=3)
        # Planet 0, turn_offset=1: both fleets contributed → sum = 5+7 = 12 (positive: player 0)
        assert arr[1, 0, 0] == 12.0, f"Expected 12 ships, got {arr[1, 0, 0]}"
        # Fleet count should be 2
        assert arr[1, 0, 1] == 2.0, f"Expected 2 fleets, got {arr[1, 0, 1]}"

    def test_encode_opponent_ships_negative(self):
        """Ships owned by the opponent (player 1) produce a negative delta."""
        events = [
            FleetEvent(owner=1, destination_planet_id=0, eta_turn=0, ships_arriving=8),
        ]
        arr = encode_events(events, num_planets=2, horizon=2, observing_player=0)
        # turn_offset=0, planet 0: opponent's ships → negative
        assert arr[0, 0, 0] == -8.0

    def test_encode_zero_pads_empty_turns(self):
        """Turns with no events are zero-filled."""
        events = [
            FleetEvent(owner=0, destination_planet_id=0, eta_turn=2, ships_arriving=5),
        ]
        arr = encode_events(events, num_planets=2, horizon=4)
        # turn offset 0 and 1 should be zero
        assert arr[0, :, :].sum() == 0
        assert arr[1, :, :].sum() == 0

    def test_two_states_same_event_graph_encode_identically(self):
        """Two event lists with identical events produce identical encoded arrays."""
        ev1 = [FleetEvent(owner=0, destination_planet_id=1, eta_turn=2, ships_arriving=10)]
        ev2 = [FleetEvent(owner=0, destination_planet_id=1, eta_turn=2, ships_arriving=10)]
        a1 = encode_events(ev1, num_planets=3, horizon=5)
        a2 = encode_events(ev2, num_planets=3, horizon=5)
        np.testing.assert_array_equal(a1, a2)

    def test_encode_returns_numpy_array(self):
        """encode_events returns an np.ndarray."""
        arr = encode_events([], num_planets=2, horizon=3)
        assert isinstance(arr, np.ndarray)


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
