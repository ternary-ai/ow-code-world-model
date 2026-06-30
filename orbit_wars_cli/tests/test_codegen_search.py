"""
tests/test_codegen_search.py — Tests for cwm/codegen_search.py (Module 9).

Coverage:
  - test_thompson_select_favors_high_pass_rate
  - test_thompson_select_favors_unexplored_when_tied
  - test_run_trajectory_tests_counts_correctly
  - test_refine_loop_terminates_on_full_pass
  - test_refine_loop_terminates_on_max_rounds
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cwm.codegen_search import (
    CodeCandidate,
    Trajectory,
    thompson_select,
    run_trajectory_tests,
    refine_loop,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _candidate(cid: str, passes: int, total: int, attempts: int = 0) -> CodeCandidate:
    return CodeCandidate(
        candidate_id=cid,
        source_code=f"# candidate {cid}",
        test_pass_count=passes,
        test_total=total,
        refinement_attempts=attempts,
    )


def _identity_transition(obs, action):
    """A trivial transition function: always returns the observation unchanged."""
    return obs


def _make_trajectory(length: int = 3) -> Trajectory:
    """Build a simple trajectory where observation and action cycle through ints."""
    steps = []
    for i in range(length):
        steps.append({"obs": i, "action": i + 10, "next_obs": i + 1})
    return Trajectory(steps=steps)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestThompsonSelect:

    def test_thompson_select_favors_high_pass_rate(self):
        """Candidate with 9/10 tests passing is selected more often than 1/10."""
        good = _candidate("good", passes=9, total=10)
        bad = _candidate("bad", passes=1, total=10)
        candidates = [good, bad]

        rng = np.random.default_rng(0)
        counts = {"good": 0, "bad": 0}
        n_trials = 500
        for _ in range(n_trials):
            selected = thompson_select(candidates, rng)
            counts[selected.candidate_id] += 1

        assert counts["good"] > counts["bad"] * 3, (
            f"Expected 'good' to dominate, got counts={counts}"
        )

    def test_thompson_select_favors_unexplored_when_tied(self):
        """When pass rates are equal, the less-refined candidate is selected more."""
        explored = _candidate("explored", passes=5, total=10, attempts=10)
        fresh = _candidate("fresh", passes=5, total=10, attempts=0)
        candidates = [explored, fresh]

        rng = np.random.default_rng(42)
        counts = {"explored": 0, "fresh": 0}
        n_trials = 300
        for _ in range(n_trials):
            selected = thompson_select(candidates, rng)
            counts[selected.candidate_id] += 1

        assert counts["fresh"] > counts["explored"], (
            f"Expected 'fresh' to be selected more, got counts={counts}"
        )

    def test_thompson_select_handles_single_candidate(self):
        """Single candidate is always selected."""
        c = _candidate("only", passes=5, total=10)
        rng = np.random.default_rng(0)
        for _ in range(10):
            assert thompson_select([c], rng).candidate_id == "only"


class TestRunTrajectoryTests:

    def test_run_trajectory_tests_counts_correctly(self):
        """Mock transition with injected mismatch produces expected pass_count."""
        # Trajectory: obs=0 --action=10--> next_obs=1
        # Candidate transition: returns obs+1 (correct)
        correct_code = "def transition(obs, action): return obs + 1"
        c = CodeCandidate(
            candidate_id="c0",
            source_code=correct_code,
            test_pass_count=0,
            test_total=0,
            refinement_attempts=0,
        )
        traj = Trajectory(steps=[
            {"obs": 0, "action": 10, "next_obs": 1},
            {"obs": 1, "action": 11, "next_obs": 2},
            {"obs": 2, "action": 12, "next_obs": 3},  # mismatch: will inject wrong
        ])

        def wrong_transition_fn(obs, action):
            if obs == 2:
                return 99   # wrong for the 3rd step
            return obs + 1

        # Test the full trajectory against the wrong transition
        mismatch_code = "# has mismatch at obs=2"
        c2 = CodeCandidate(
            candidate_id="c2",
            source_code=mismatch_code,
            test_pass_count=0,
            test_total=0,
            refinement_attempts=0,
        )
        result = run_trajectory_tests(c2, [traj],
                                      transition_fn=wrong_transition_fn)
        assert result.test_total == 3   # 3 steps in trajectory
        assert result.test_pass_count == 2  # first 2 steps pass; step 3 fails

    def test_run_trajectory_tests_all_pass(self):
        """Correct transition function passes all steps."""
        c = _candidate("c", passes=0, total=0)
        traj = _make_trajectory(4)

        def correct_fn(obs, action):
            return obs + 1  # matches step["next_obs"] = obs + 1

        result = run_trajectory_tests(c, [traj], transition_fn=correct_fn)
        assert result.test_pass_count == result.test_total == 4

    def test_run_trajectory_tests_multiple_trajectories(self):
        """Multiple trajectories are all tested; totals accumulate."""
        c = _candidate("c", passes=0, total=0)
        trajs = [_make_trajectory(3), _make_trajectory(2)]

        def correct_fn(obs, action):
            return obs + 1

        result = run_trajectory_tests(c, trajs, transition_fn=correct_fn)
        assert result.test_total == 5
        assert result.test_pass_count == 5


class TestRefineLoop:

    def test_refine_loop_terminates_on_full_pass(self):
        """refine_fn returning a perfect candidate causes early termination."""
        seed_candidates = [_candidate("seed", passes=0, total=3)]
        trajs = [_make_trajectory(3)]

        def perfect_refine(c: CodeCandidate) -> CodeCandidate:
            return CodeCandidate(
                candidate_id="perfect",
                source_code="perfect",
                test_pass_count=3,
                test_total=3,
                refinement_attempts=c.refinement_attempts + 1,
            )

        # Perfect candidate passes all tests → loop should stop after 1 round
        # We use a transition_fn that matches the trajectory
        result = refine_loop(
            seed_candidates=seed_candidates,
            trajectories=trajs,
            refine_fn=perfect_refine,
            max_rounds=10,
            transition_fn=lambda obs, action: obs + 1,
        )
        assert result.candidate_id == "perfect"
        assert result.test_pass_count == result.test_total

    def test_refine_loop_terminates_on_max_rounds(self):
        """A refine_fn that never improves causes the loop to stop at max_rounds."""
        seed_candidates = [_candidate("seed", passes=0, total=5)]
        trajs = [_make_trajectory(5)]
        call_count = {"n": 0}

        def bad_refine(c: CodeCandidate) -> CodeCandidate:
            call_count["n"] += 1
            return CodeCandidate(
                candidate_id=f"attempt_{call_count['n']}",
                source_code="bad",
                test_pass_count=0,
                test_total=5,
                refinement_attempts=c.refinement_attempts + 1,
            )

        max_rounds = 5
        result = refine_loop(
            seed_candidates=seed_candidates,
            trajectories=trajs,
            refine_fn=bad_refine,
            max_rounds=max_rounds,
            transition_fn=lambda obs, action: 99,   # always wrong → 0 passes
        )
        assert call_count["n"] == max_rounds


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
