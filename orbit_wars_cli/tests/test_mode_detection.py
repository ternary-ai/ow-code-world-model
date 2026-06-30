"""
tests/test_mode_detection.py — 2p vs 4p game-mode detection.

The engine must determine the player count (2 or 4) from obs/config at game
start and cache it. Detection priority (see cwm/state.py):
  1. config.agentCount               (preferred; agent(obs, config) signature)
  2. distinct non-(-1) planet owners at step 0
  3. fallback default of 2

Caching contract: once determined at step 0, the value must NOT change later in
the game (players can be eliminated, making the live owner count unreliable).
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cwm.state import _detect_num_players, state_from_obs


# ── obs builders ───────────────────────────────────────────────────────────────

def _planet(pid, owner, x=50.0, y=50.0):
    # [id, owner, x, y, radius, ships, production]
    return [pid, owner, x, y, 1.0, 10, 1]


def _obs(owners, step=0):
    """Build a minimal obs dict with one planet per owner (plus a neutral)."""
    planets = [_planet(i, o, x=10.0 + 5 * i) for i, o in enumerate(owners)]
    planets.append(_planet(len(owners), -1, x=90.0))  # neutral planet
    return {
        "player": 0,
        "step": step,
        "planets": planets,
        "fleets": [],
        "initial_planets": [list(p) for p in planets],
        "comets": [],
        "comet_planet_ids": [],
        "next_fleet_id": len(planets),
        "angular_velocity": 0.03,
    }


# ── config.agentCount path ─────────────────────────────────────────────────────

@pytest.mark.parametrize("count", [2, 4])
def test_detect_from_config_dict(count):
    obs = _obs(list(range(count)))
    config = {"agentCount": count}
    assert _detect_num_players(obs, config) == count


@pytest.mark.parametrize("count", [2, 4])
def test_detect_from_config_namespace(count):
    obs = _obs(list(range(count)))
    config = SimpleNamespace(agentCount=count)
    assert _detect_num_players(obs, config) == count


def test_config_takes_priority_over_owner_count():
    # Only 2 distinct owners visible, but config says 4 → trust config.
    obs = _obs([0, 1])
    assert _detect_num_players(obs, {"agentCount": 4}) == 4


# ── owner-count fallback (no config) ───────────────────────────────────────────

@pytest.mark.parametrize("count", [2, 4])
def test_detect_from_owners_no_config(count):
    obs = _obs(list(range(count)))
    assert _detect_num_players(obs, None) == count


def test_detect_ignores_neutral_owners():
    # 2 real players plus several neutral (-1) planets.
    obs = _obs([0, 1])
    obs["planets"].extend([_planet(50, -1, x=20.0), _planet(51, -1, x=30.0)])
    assert _detect_num_players(obs, None) == 2


def test_fallback_default_is_2_when_undeterminable():
    # Unusual owner count (e.g. 3 distinct) → safe default of 2.
    obs = _obs([0, 1, 2])
    assert _detect_num_players(obs, None) == 2


# ── caching contract ───────────────────────────────────────────────────────────

def test_state_from_obs_respects_cached_num_players():
    obs = _obs([0, 1])
    state = state_from_obs(obs, config=None, cached_num_players=4)
    assert state.num_players == 4  # cached value wins, no re-detection


def test_cached_value_survives_midgame_elimination():
    # 4p game where, mid-game, only 1 player still controls planets.
    # With cached_num_players=4 the engine must keep reporting 4.
    obs = _obs([0], step=200)            # single surviving owner
    state = state_from_obs(obs, config=None, cached_num_players=4)
    assert state.num_players == 4


def test_uncached_detection_used_when_none():
    obs4 = _obs([0, 1, 2, 3])
    state = state_from_obs(obs4, config={"agentCount": 4}, cached_num_players=None)
    assert state.num_players == 4
