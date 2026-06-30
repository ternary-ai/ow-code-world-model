"""
tests/test_arena.py — sanity checks for the evaluation harness (cli/arena.py).

Guards the statistics helpers and the match mechanics that gate every later
strength claim, so a broken arena can't silently report bogus win rates.
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cli.arena import (
    wilson_interval,
    greedy_agent,
    load_start_states,
    play_match,
    head_to_head,
)
from cwm.interpreter import cwm_is_terminal


# ── Wilson interval ────────────────────────────────────────────────────────────

def test_wilson_interval_contains_point_estimate():
    lo, hi = wilson_interval(7, 10)
    assert 0.0 <= lo <= 0.7 <= hi <= 1.0


def test_wilson_interval_shrinks_with_more_samples():
    lo_small, hi_small = wilson_interval(5, 10)
    lo_big, hi_big = wilson_interval(500, 1000)
    assert (hi_big - lo_big) < (hi_small - lo_small)


def test_wilson_interval_empty_is_full_range():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_interval_all_wins_high_near_one():
    lo, hi = wilson_interval(50, 50)
    assert hi == 1.0 or hi > 0.9
    assert lo > 0.5


# ── Match mechanics ────────────────────────────────────────────────────────────

def test_play_match_does_not_mutate_start_state():
    states = load_start_states("2p", n=1)
    start = states[0]
    before_planets = [list(p) for p in start.planets]
    before_step = start.step

    g = greedy_agent()
    play_match([g, g], start, num_players=2, max_steps=20, rng=random.Random(0))

    assert start.step == before_step, "start state step was mutated"
    assert [list(p) for p in start.planets] == before_planets, "planets mutated"


def test_play_match_returns_one_result_per_seat():
    states = load_start_states("4p", n=1)
    g = greedy_agent()
    res = play_match([g, g, g, g], states[0], num_players=4,
                     max_steps=10, rng=random.Random(1))
    assert len(res) == 4
    assert all(r in (0.0, 0.5, 1.0) for r in res)


def test_head_to_head_is_deterministic_for_fixed_seed():
    g1 = greedy_agent("g1")
    g2 = greedy_agent("g2")
    r1 = head_to_head(g1, g2, "2p", games=4, seed=7, max_steps=30)
    r2 = head_to_head(g1, g2, "2p", games=4, seed=7, max_steps=30)
    assert r1.a_score == r2.a_score
    assert r1.a_wins == r2.a_wins
    assert r1.games == 4


def test_head_to_head_scores_are_consistent():
    g1 = greedy_agent("g1")
    g2 = greedy_agent("g2")
    r = head_to_head(g1, g2, "2p", games=5, seed=0, max_steps=30)
    # wins + draws + losses == games; score == wins + 0.5*draws
    assert r.a_wins + r.draws + r.a_losses == r.games
    assert abs(r.a_score - (r.a_wins + 0.5 * r.draws)) < 1e-9
