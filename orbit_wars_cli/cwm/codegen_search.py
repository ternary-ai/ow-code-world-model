"""
cwm/codegen_search.py — Thompson-sampling bandit over code candidates (Module 9).

Applies multi-armed bandit logic to selecting which LLM-generated code candidate
to refine next, enabling backtracking over the search tree of code revisions.

Intended for use with components where correctness is approximate by nature
(opponent_model, intercept in pathological cases), not for closed-form physics
code where greedy LLM refinement against unit tests converges quickly.

Algorithm:
  - Each CodeCandidate has a Beta(pass_count+1, fail_count+1) posterior.
  - thompson_select samples from each posterior, weighted by 1/(1+attempts)
    to favor less-explored candidates.
  - refine_loop: select → refine (external LLM call) → re-test → repeat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ── Types ──────────────────────────────────────────────────────────────────────

@dataclass
class CodeCandidate:
    """One candidate implementation with its test history."""
    candidate_id: str
    source_code: str
    test_pass_count: int = 0
    test_total: int = 0
    refinement_attempts: int = 0


@dataclass
class Trajectory:
    """A recorded sequence of (obs, action, next_obs) triples."""
    steps: list  # list of {"obs": ..., "action": ..., "next_obs": ...}


# ── Core functions ─────────────────────────────────────────────────────────────

def thompson_select(
    candidates: list[CodeCandidate],
    rng: np.random.Generator,
) -> CodeCandidate:
    """Select a candidate using Thompson sampling with an exploration bonus.

    For each candidate:
      1. Sample θ from Beta(pass_count+1, fail_count+1).
      2. Scale by the exploration weight 1 / (1 + refinement_attempts).
    Return the candidate with the highest scaled sample.
    """
    best_candidate = candidates[0]
    best_value = -1.0

    for c in candidates:
        fail_count = c.test_total - c.test_pass_count
        # Beta(α, β): α = passes+1, β = fails+1
        theta = float(rng.beta(c.test_pass_count + 1, fail_count + 1))
        # Exploration weight: favour less-refined candidates
        weight = 1.0 / (1.0 + c.refinement_attempts)
        value = theta * weight
        if value > best_value:
            best_value = value
            best_candidate = c

    return best_candidate


def run_trajectory_tests(
    candidate: CodeCandidate,
    trajectories: list[Trajectory],
    transition_fn: Callable | None = None,
) -> CodeCandidate:
    """Replay each trajectory through candidate.source_code's transition function.

    For each step (obs, action, next_obs), calls transition_fn(obs, action) and
    checks whether the result equals next_obs.  Returns an updated CodeCandidate
    with test_pass_count and test_total set.

    The transition_fn parameter allows injecting a mock; when None, the function
    attempts to exec() candidate.source_code and call its `transition` symbol.
    """
    if transition_fn is None:
        ns: dict = {}
        exec(candidate.source_code, ns)  # noqa: S102
        transition_fn = ns.get("transition")
        if transition_fn is None:
            raise ValueError(
                f"candidate {candidate.candidate_id!r} source_code does not define "
                "'transition(obs, action)'"
            )

    passes = 0
    total = 0
    for traj in trajectories:
        for step in traj.steps:
            obs = step["obs"]
            action = step["action"]
            expected = step["next_obs"]
            try:
                result = transition_fn(obs, action)
                if result == expected:
                    passes += 1
            except Exception:
                pass   # exception counts as a failure
            total += 1

    return CodeCandidate(
        candidate_id=candidate.candidate_id,
        source_code=candidate.source_code,
        test_pass_count=passes,
        test_total=total,
        refinement_attempts=candidate.refinement_attempts,
    )


def refine_loop(
    seed_candidates: list[CodeCandidate],
    trajectories: list[Trajectory],
    refine_fn: Callable[[CodeCandidate], CodeCandidate],
    max_rounds: int,
    transition_fn: Callable | None = None,
    rng: np.random.Generator | None = None,
) -> CodeCandidate:
    """Thompson-sampling refinement loop.

    Repeats for up to max_rounds:
      1. Select a candidate via thompson_select.
      2. Call refine_fn(candidate) to produce a revised candidate (the LLM call).
      3. Re-test via run_trajectory_tests.
      4. Append the refined candidate to the pool.
      5. Stop early if the refined candidate passes ALL tests.

    Returns the best candidate found (highest pass_count / test_total ratio).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Test seeds first so we have an initial score baseline
    pool: list[CodeCandidate] = []
    for c in seed_candidates:
        tested = run_trajectory_tests(c, trajectories, transition_fn)
        pool.append(tested)

    for _ in range(max_rounds):
        selected = thompson_select(pool, rng)
        refined = refine_fn(selected)
        refined = CodeCandidate(
            candidate_id=refined.candidate_id,
            source_code=refined.source_code,
            test_pass_count=refined.test_pass_count,
            test_total=refined.test_total,
            refinement_attempts=refined.refinement_attempts,
        )
        tested = run_trajectory_tests(refined, trajectories, transition_fn)
        tested = CodeCandidate(
            candidate_id=tested.candidate_id,
            source_code=tested.source_code,
            test_pass_count=tested.test_pass_count,
            test_total=tested.test_total,
            refinement_attempts=tested.refinement_attempts,
        )
        pool.append(tested)

        # Early termination: perfect candidate
        if tested.test_total > 0 and tested.test_pass_count == tested.test_total:
            return tested

    # Return the candidate with the highest pass rate
    def _score(c: CodeCandidate) -> float:
        return c.test_pass_count / max(c.test_total, 1)

    return max(pool, key=_score)
