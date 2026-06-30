"""
tests/test_transitions.py — CWM transition fidelity tests against recorded trajectories.

For each recorded triple (obs_t, joint_action_t, obs_{t+1}):
  1. state_t = state_from_obs(obs_t)
  2. out = cwm_apply_joint_action(state_t, joint_action_t, spawn_rng=comet_rng)
     where comet_rng = random.Random(f"orbit_wars-comet-{episode_seed}-{t+1}")
     (only used when (t+1) in COMET_SPAWN_STEPS)
  3. expected = state_from_obs(obs_{t+1})
  4. Compare field-by-field

Target: 100% on both 2p and 4p sets independently.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cwm.state import state_from_obs, State, COMET_SPAWN_STEPS

TRAJ_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "trajectories")
FLOAT_TOL = 1e-6   # tolerance for position/angle comparisons

# ── Load trajectory files ──────────────────────────────────────────────────────

def _load_trajectories(subdir: str) -> list[dict]:
    path = os.path.join(TRAJ_DIR, subdir)
    games = []
    for fname in sorted(os.listdir(path)):
        if fname.endswith(".json"):
            with open(os.path.join(path, fname)) as f:
                games.append(json.load(f))
    return games


# ── Comparison helpers ─────────────────────────────────────────────────────────

def _approx_eq(a, b, tol=FLOAT_TOL):
    return abs(a - b) <= tol


def _compare_planets(out_planets: list, exp_planets: list, label: str) -> list[str]:
    """Compare planet lists field by field. Returns list of mismatch strings."""
    errors = []
    exp_map = {p[0]: p for p in exp_planets}
    out_map = {p[0]: p for p in out_planets}

    # Check for missing/extra planets
    missing = set(exp_map) - set(out_map)
    extra   = set(out_map) - set(exp_map)
    if missing:
        errors.append(f"{label}: missing planet IDs {sorted(missing)}")
    if extra:
        errors.append(f"{label}: extra planet IDs {sorted(extra)}")

    for pid in sorted(set(out_map) & set(exp_map)):
        o = out_map[pid]
        e = exp_map[pid]
        # id, owner (exact int)
        if o[0] != e[0]:
            errors.append(f"{label} planet {pid}: id mismatch {o[0]} vs {e[0]}")
        if o[1] != e[1]:
            errors.append(f"{label} planet {pid}: owner {o[1]} vs {e[1]}")
        # x, y (float)
        if not _approx_eq(o[2], e[2]):
            errors.append(f"{label} planet {pid}: x {o[2]:.6f} vs {e[2]:.6f}")
        if not _approx_eq(o[3], e[3]):
            errors.append(f"{label} planet {pid}: y {o[3]:.6f} vs {e[3]:.6f}")
        # radius (float)
        if not _approx_eq(o[4], e[4]):
            errors.append(f"{label} planet {pid}: radius {o[4]:.6f} vs {e[4]:.6f}")
        # ships (int)
        if o[5] != e[5]:
            errors.append(f"{label} planet {pid}: ships {o[5]} vs {e[5]}")
        # production (int)
        if o[6] != e[6]:
            errors.append(f"{label} planet {pid}: production {o[6]} vs {e[6]}")

    return errors


def _compare_fleets(out_fleets: list, exp_fleets: list, label: str) -> list[str]:
    """Compare fleet lists by fleet ID. Returns list of mismatch strings."""
    errors = []
    exp_map = {f[0]: f for f in exp_fleets}
    out_map = {f[0]: f for f in out_fleets}

    missing = set(exp_map) - set(out_map)
    extra   = set(out_map) - set(exp_map)
    if missing:
        errors.append(f"{label}: missing fleet IDs {sorted(missing)}")
    if extra:
        errors.append(f"{label}: extra fleet IDs {sorted(extra)}")

    for fid in sorted(set(out_map) & set(exp_map)):
        o = out_map[fid]
        e = exp_map[fid]
        if o[1] != e[1]:
            errors.append(f"{label} fleet {fid}: owner {o[1]} vs {e[1]}")
        if not _approx_eq(o[2], e[2]):
            errors.append(f"{label} fleet {fid}: x {o[2]:.6f} vs {e[2]:.6f}")
        if not _approx_eq(o[3], e[3]):
            errors.append(f"{label} fleet {fid}: y {o[3]:.6f} vs {e[3]:.6f}")
        if not _approx_eq(o[4], e[4]):
            errors.append(f"{label} fleet {fid}: angle {o[4]:.6f} vs {e[4]:.6f}")
        if o[5] != e[5]:
            errors.append(f"{label} fleet {fid}: from_planet_id {o[5]} vs {e[5]}")
        if o[6] != e[6]:
            errors.append(f"{label} fleet {fid}: ships {o[6]} vs {e[6]}")

    return errors


def _compare_comet_ids(out: list, exp: list, label: str) -> list[str]:
    if set(out) != set(exp):
        return [f"{label}: comet_planet_ids {sorted(out)} vs {sorted(exp)}"]
    return []


def _compare_next_fleet_id(out: int, exp: int, label: str) -> list[str]:
    if out != exp:
        return [f"{label}: next_fleet_id {out} vs {exp}"]
    return []


def _run_transition(game: dict, tr: dict) -> list[str]:
    """Run one transition and return list of field mismatch strings (empty = pass)."""
    from cwm.interpreter import cwm_apply_joint_action

    t = tr["t"]
    obs_t      = tr["obs_t"]
    joint_act  = tr["joint_action"]
    obs_t1     = tr["obs_t1"]
    episode_seed = game["episode_seed"]
    cfg = game["config"]

    # Build a minimal config-like object
    class _Cfg:
        episodeSteps = cfg["episodeSteps"]
        shipSpeed    = cfg["shipSpeed"]
        cometSpeed   = cfg["cometSpeed"]
        agentCount   = cfg["agentCount"]

    # Reconstruct state_t from obs
    state_t = state_from_obs(obs_t, _Cfg(), cached_num_players=cfg["agentCount"])

    # Build comet spawn RNG (only actually used when (t+1) in COMET_SPAWN_STEPS)
    spawn_rng = random.Random(f"orbit_wars-comet-{episode_seed}-{t + 1}")

    # Apply CWM transition
    out = cwm_apply_joint_action(state_t, joint_act, spawn_rng=spawn_rng)

    # Build expected state from obs_t+1
    exp = state_from_obs(obs_t1, _Cfg(), cached_num_players=cfg["agentCount"])

    label = f"game={game.get('_idx','?')} t={t}"
    errors = []
    errors += _compare_planets(out.planets, exp.planets, label)
    errors += _compare_fleets(out.fleets, exp.fleets, label)
    errors += _compare_comet_ids(out.comet_planet_ids, exp.comet_planet_ids, label)
    errors += _compare_next_fleet_id(out.next_fleet_id, exp.next_fleet_id, label)

    return errors


# ── Parametrised tests ─────────────────────────────────────────────────────────

def _collect_params(subdir: str):
    """Collect (game_dict, transition_dict) pairs for parametrize."""
    games = _load_trajectories(subdir)
    params = []
    for idx, game in enumerate(games):
        game["_idx"] = idx
        for tr in game["transitions"]:
            params.append((game, tr))
    return params


# Collect once at module import (not inside the test function for speed)
_2P_PARAMS = _collect_params("2p")
_4P_PARAMS = _collect_params("4p")


@pytest.mark.parametrize("game,tr", _2P_PARAMS)
def test_transition_2p(game, tr):
    """2p CWM transition must exactly match recorded obs_{t+1} (100% target)."""
    errors = _run_transition(game, tr)
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("game,tr", _4P_PARAMS)
def test_transition_4p(game, tr):
    """4p CWM transition must exactly match recorded obs_{t+1} (100% target)."""
    errors = _run_transition(game, tr)
    assert not errors, "\n".join(errors)


# ── Summary helper (run directly for quick diagnosis) ─────────────────────────

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "2p"
    games = _load_trajectories(mode)
    total = pass_ = fail = 0
    first_errors = []
    for idx, game in enumerate(games):
        game["_idx"] = idx
        for tr in game["transitions"]:
            total += 1
            errs = _run_transition(game, tr)
            if errs:
                fail += 1
                if len(first_errors) < 5:
                    first_errors.extend(errs[:3])
            else:
                pass_ += 1
    print(f"{mode}: {pass_}/{total} passed, {fail} failed")
    if first_errors:
        print("First errors:")
        for e in first_errors:
            print(" ", e)
