"""
cli/train.py — `orbit-wars train` subcommand.

CMA-ES weight/MCTS-parameter tuner via self-play.

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is a gradient-free
optimiser that maintains a multivariate Gaussian over the parameter space and
adapts its covariance matrix towards regions of high reward.  It finds better
weights than random search with the same evaluation budget because it exploits
the correlation structure of the objective.

All 24 parameters are normalised to [0, 1] so that sigma=0.3 is a meaningful
initial step size (~30% of each parameter's range).  Discrete parameters
(max_depth, opp_aggregation, …) are encoded as continuous values and rounded
back to the nearest option at decode time.

The ``--iters`` flag sets the total number of candidate evaluations (maxfevals
for CMA-ES).  CMA-ES internally uses a population of λ ≈ 4+3·ln(n) ≈ 14
solutions per generation for n=24 parameters; 40 iters ≈ 3 generations, 200
iters ≈ 14 generations.  For meaningful search, aim for ≥ 100 iters.

Usage (via main.py):
    orbit-wars train [--mode 2p|4p|both] [--iters N] [--games-per-eval N]
                     [--seed N] [--out-dir PATH] [--workers N]
                     [--self-play] [--pool-size N]

Defaults:
    --mode          both
    --iters         40       (total candidate evaluations)
    --games-per-eval 4
    --seed          0
    --workers       1
    --out-dir       <repo>/mcts/
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from mcts.value_fn import DEFAULT_WEIGHTS
from cli.arena import greedy_agent, mcts_agent, head_to_head, head_to_head_field


_MCTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "mcts")

# Evaluation truncation per mode.  4p games need more turns to reach stable
# outcomes; 120 steps (the old shared default) cuts off too early.
_MAX_STEPS = {"2p": 120, "4p": 250}

# mcts_budget_s training range per player count.
# 4p lower-bound is 0.20 so candidates are tested at a budget close to the
# 0.60 s competition cap, not 0.07 s where the agent was previously trained.
_BUDGET_RANGE = {2: (0.05, 0.25), 4: (0.20, 0.45)}


# ── Parameter space ────────────────────────────────────────────────────────────
#
# Each entry: (name, kind, spec) where
#   kind="continuous" → spec=(lo, hi)
#   kind="discrete"   → spec=[option0, option1, ...]
#
# All parameters are normalised to [0, 1] in the CMA-ES vector.
# Discrete: encoded as a float; round(v * (n-1)) gives the option index.
#
# right_size is held at True (proven win; not sampled).
# w_anti_leader is forced to 0.0 for 2p at decode time.

_PARAMS: list[tuple] = [
    # ── value-function weights ─────────────────────────────────────────────────
    ("w_material",        "continuous", (0.05, 1.0)),
    ("w_production",      "continuous", (0.05, 1.0)),
    ("w_control",         "continuous", (0.05, 1.0)),
    ("w_offense",         "continuous", (0.05, 1.0)),
    ("w_cohesion",        "continuous", (0.0,  0.6)),
    ("w_centrality",      "continuous", (0.0,  0.6)),
    ("w_threat",          "continuous", (0.0,  0.6)),
    ("w_anti_leader",     "continuous", (0.0,  0.6)),  # zeroed for 2p in _decode
    ("w_neutral_access",  "continuous", (0.0,  0.6)),
    ("w_incoming_threat", "continuous", (0.0,  0.6)),
    ("w_time_fleet",      "continuous", (0.0,  0.6)),
    ("w_prod_density",    "continuous", (0.0,  0.6)),
    ("w_phase_material",  "continuous", (0.0,  0.6)),
    ("w_event_fleet",     "continuous", (0.0,  0.6)),  # Module 5; was held at 0
    ("target_weakness",   "continuous", (0.0,  1.0)),
    ("mcts_budget_s",     "continuous", (0.05, 0.25)),
    # ── search hyper-parameters ────────────────────────────────────────────────
    ("opp_aggregation",   "discrete",   ["sum", "max", "mean"]),
    ("max_depth",         "discrete",   [2, 3, 4]),
    ("pw_c",              "discrete",   [2.0, 4.0, 8.0]),
    ("pw_alpha",          "discrete",   [0.3, 0.5, 0.7]),
    ("k_targets",         "discrete",   [3, 4, 5]),
    ("n_active_planets",  "discrete",   [2, 3, 4]),
    ("k_reinforce",       "discrete",   [0, 1, 2]),
    ("fine_fractions",    "discrete",   [False, True]),
]
_N = len(_PARAMS)  # 24


def _budget_range(num_players: int) -> tuple[float, float]:
    return _BUDGET_RANGE.get(num_players, _BUDGET_RANGE[2])


def _encode(weights: dict, num_players: int) -> list[float]:
    """Encode a weights dict → normalised [0, 1]^_N CMA-ES vector."""
    x = []
    for name, kind, spec in _PARAMS:
        val = weights.get(name, DEFAULT_WEIGHTS.get(name, 0.0))
        if kind == "continuous":
            lo, hi = spec if name != "mcts_budget_s" else _budget_range(num_players)
            x.append(float(np.clip((val - lo) / (hi - lo), 0.0, 1.0)))
        else:
            options = spec
            try:
                idx = options.index(val)
            except ValueError:
                idx = 0
            x.append(idx / max(1, len(options) - 1))
    return x


def _decode(x: list[float], num_players: int) -> dict:
    """Decode a normalised CMA-ES vector → weights dict ready for the agent."""
    out: dict = {"right_size": True}
    for i, (name, kind, spec) in enumerate(_PARAMS):
        v = float(np.clip(x[i], 0.0, 1.0))
        if kind == "continuous":
            lo, hi = spec if name != "mcts_budget_s" else _budget_range(num_players)
            out[name] = round(lo + v * (hi - lo), 4)
        else:
            options = spec
            idx = int(round(v * (len(options) - 1)))
            out[name] = options[max(0, min(len(options) - 1, idx))]
    if num_players < 4:
        out["w_anti_leader"] = 0.0
    return out


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def _evaluate(
    candidate: dict,
    mode: str,
    n_games: int,
    seed: int,
    max_steps: int | None = None,
) -> float:
    """Win-rate of *candidate* vs greedy baseline; in [0, 1]."""
    if max_steps is None:
        max_steps = _MAX_STEPS[mode]
    res = head_to_head(
        mcts_agent(candidate, name="candidate"),
        greedy_agent(),
        mode,
        games=n_games,
        seed=seed,
        max_steps=max_steps,
    )
    return res.a_rate


def _eval_worker(task: tuple) -> tuple[int, float]:
    """Top-level picklable worker: score one candidate vs greedy."""
    index, candidate, mode, n_games, seed, max_steps = task
    return index, _evaluate(candidate, mode, n_games, seed, max_steps)


def _pool_eval_worker(task: tuple) -> tuple[int, int, float, int]:
    """Top-level picklable worker: score one candidate against one opponent spec.

    For 4p, opp_spec may be a list of specs — the candidate then plays a
    heterogeneous field drawn from the full pool rather than 3 clones of
    the same agent.
    """
    ci, oi, candidate, opp_spec, mode, n_games, seed, max_steps = task
    cand_agent = mcts_agent(candidate, name="candidate")

    if isinstance(opp_spec, list):
        # 4p heterogeneous field: build one agent per pool member
        field = [
            greedy_agent() if s == "greedy" else mcts_agent(s, name=f"pool{i}")
            for i, s in enumerate(opp_spec)
        ]
        res = head_to_head_field(cand_agent, field, mode,
                                 games=n_games, seed=seed, max_steps=max_steps)
    else:
        opp = greedy_agent() if opp_spec == "greedy" else mcts_agent(opp_spec, name="pool")
        res = head_to_head(cand_agent, opp, mode,
                           games=n_games, seed=seed, max_steps=max_steps)
    return ci, oi, res.a_score, res.games


# ── CMA-ES core ────────────────────────────────────────────────────────────────

def _cmaes_options(n_fevals: int, rng_seed: int) -> dict:
    return {
        "maxfevals": n_fevals,
        "seed":      rng_seed % (2**32 - 1),
        "verbose":   -9,           # suppress CMA stdout; we log ourselves
        "bounds":    [[0.0] * _N, [1.0] * _N],
        "tolx":      1e-4,
        "tolfun":    1e-4,
    }


def _eval_population(
    solutions: list,
    candidates: list[dict],
    mode: str,
    n_games: int,
    rng: random.Random,
    workers: int,
    max_steps: int,
) -> list[float]:
    """Evaluate a CMA-ES population against greedy; return win-rates."""
    seeds = [rng.randint(0, 1 << 30) for _ in candidates]
    if workers <= 1:
        return [_evaluate(c, mode, n_games, s, max_steps) for c, s in zip(candidates, seeds)]
    tasks = [(i, candidates[i], mode, n_games, seeds[i], max_steps) for i in range(len(candidates))]
    scores = [0.0] * len(candidates)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for idx, score in pool.map(_eval_worker, tasks):
            scores[idx] = score
    return scores


def _pool_score_population(
    candidates: list[dict],
    pool: list,
    mode: str,
    n_games: int,
    base_seed: int,
    workers: int,
    max_steps: int,
) -> list[float]:
    """Pooled win-rate for each candidate vs the pool.

    2p: one task per (candidate, pool-member) pair; scores are averaged.
    4p: one task per candidate; the candidate plays a heterogeneous field
        drawn from the entire pool — no clones, more realistic.
    """
    if mode == "4p":
        # Single task per candidate: entire pool becomes the field
        tasks = [
            (ci, 0, cand, list(pool), mode, n_games, base_seed + ci * 1_000_003, max_steps)
            for ci, cand in enumerate(candidates)
        ]
        scores = [0.0] * len(candidates)
        if workers <= 1:
            for t in tasks:
                ci, _, a_score, games = _pool_eval_worker(t)
                scores[ci] = a_score / games if games else 0.0
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for ci, _, a_score, games in ex.map(_pool_eval_worker, tasks):
                    scores[ci] = a_score / games if games else 0.0
        return scores
    else:
        # 2p: score against each pool member individually, then average
        tasks = []
        for ci, cand in enumerate(candidates):
            for oi, opp in enumerate(pool):
                seed = base_seed + ci * 1_000_003 + oi * 7_919
                tasks.append((ci, oi, cand, opp, mode, n_games, seed, max_steps))
        agg = [[0.0, 0] for _ in candidates]
        if workers <= 1:
            for t in tasks:
                ci, oi, a_score, games = _pool_eval_worker(t)
                agg[ci][0] += a_score
                agg[ci][1] += games
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for ci, oi, a_score, games in ex.map(_pool_eval_worker, tasks):
                    agg[ci][0] += a_score
                    agg[ci][1] += games
        return [(s / g if g else 0.0) for s, g in agg]


# ── Training loops ─────────────────────────────────────────────────────────────

def _train_mode_cmaes(
    mode: str,
    n_iters: int,
    n_games: int,
    rng: random.Random,
    out_dir: str,
    workers: int = 1,
    max_steps: int | None = None,
) -> dict:
    """CMA-ES optimisation vs greedy baseline. Returns the best weights found."""
    import cma as _cma

    num_players = 2 if mode == "2p" else 4
    if max_steps is None:
        max_steps = _MAX_STEPS[mode]
    out_path = os.path.join(out_dir, f"best_weights_{mode}.json")

    if os.path.exists(out_path):
        with open(out_path) as fh:
            best = json.load(fh)
        print(f"  [{mode}] Warm-starting CMA-ES from {out_path}")
    else:
        best = dict(DEFAULT_WEIGHTS)
        best["mcts_budget_s"] = 0.10

    x0 = _encode(best, num_players)
    es = _cma.CMAEvolutionStrategy(x0, 0.3, _cmaes_options(n_iters, rng.randint(1, 2**31 - 1)))

    best_score = _evaluate(best, mode, n_games, rng.randint(0, 1 << 30), max_steps)
    print(f"  [{mode}] incumbent win_rate vs greedy = {best_score:.3f}  "
          f"(CMA-ES λ≈{es.popsize}  n={_N}  budget={n_iters} evals  max_steps={max_steps})")

    total_evals = 0
    gen = 0

    while not es.stop() and total_evals < n_iters:
        solutions = es.ask()
        candidates = [_decode(sol, num_players) for sol in solutions]
        scores = _eval_population(solutions, candidates, mode, n_games, rng, workers, max_steps)

        # CMA-ES minimises: pass negated win-rates
        es.tell(solutions, [-s for s in scores])

        for i, (cand, score) in enumerate(zip(candidates, scores)):
            total_evals += 1
            is_new_best = score > best_score
            if is_new_best:
                best, best_score = cand, score
                _save(best, out_path)
            print(
                f"  [{mode}] gen={gen:3d}  eval={total_evals:3d}/{n_iters}  "
                f"win_rate={score:.3f}  best={best_score:.3f}"
                + ("  *NEW BEST*" if is_new_best else "")
            )
        gen += 1

    _save(best, out_path)
    print(f"  [{mode}] Done — best win_rate vs greedy = {best_score:.3f}  "
          f"saved to {out_path}")
    return best


def _train_mode_cmaes_selfplay(
    mode: str,
    n_iters: int,
    n_games: int,
    rng: random.Random,
    out_dir: str,
    workers: int = 1,
    pool_size: int = 4,
    max_steps: int | None = None,
) -> dict:
    """CMA-ES against a growing self-play pool. Returns the best weights found."""
    import cma as _cma

    num_players = 2 if mode == "2p" else 4
    if max_steps is None:
        max_steps = _MAX_STEPS[mode]
    out_path = os.path.join(out_dir, f"best_weights_{mode}.json")

    if os.path.exists(out_path):
        with open(out_path) as fh:
            best = json.load(fh)
        print(f"  [{mode}] Warm-starting CMA-ES from {out_path}")
    else:
        best = dict(DEFAULT_WEIGHTS)
        best["mcts_budget_s"] = 0.10

    pool: list = ["greedy", dict(best)]

    x0 = _encode(best, num_players)
    es = _cma.CMAEvolutionStrategy(x0, 0.3, _cmaes_options(n_iters, rng.randint(1, 2**31 - 1)))

    best_score = _pool_score_population([best], pool, mode, n_games,
                                        rng.randint(0, 1 << 30), workers, max_steps)[0]
    print(f"  [{mode}] incumbent pooled win_rate = {best_score:.3f}  "
          f"(pool: greedy + {len(pool) - 1} snapshot  "
          f"CMA-ES λ≈{es.popsize}  n={_N}  budget={n_iters} evals  max_steps={max_steps})")

    total_evals = 0
    gen = 0

    while not es.stop() and total_evals < n_iters:
        solutions = es.ask()
        candidates = [_decode(sol, num_players) for sol in solutions]
        scores = _pool_score_population(candidates, pool, mode, n_games,
                                        rng.randint(0, 1 << 30), workers, max_steps)

        es.tell(solutions, [-s for s in scores])

        # Accept the best candidate in this generation if it beats the incumbent
        best_in_gen = max(range(len(scores)), key=lambda k: scores[k])
        if scores[best_in_gen] > best_score:
            best = candidates[best_in_gen]
            best_score = scores[best_in_gen]
            _save(best, out_path)
            pool.append(dict(best))
            if len(pool) > pool_size + 1:
                pool = [pool[0]] + pool[-pool_size:]
            # Re-baseline against updated pool so next generation faces a harder bar
            best_score = _pool_score_population([best], pool, mode, n_games,
                                                rng.randint(0, 1 << 30), workers, max_steps)[0]
            print(f"  [{mode}] >>> accepted; pool now greedy + {len(pool) - 1} snapshot(s)  "
                  f"re-baselined pooled win_rate = {best_score:.3f}")

        for i, (cand, score) in enumerate(zip(candidates, scores)):
            total_evals += 1
            print(
                f"  [{mode}] gen={gen:3d}  eval={total_evals:3d}/{n_iters}  "
                f"pooled={score:.3f}  best={best_score:.3f}"
            )
        gen += 1

    _save(best, out_path)
    print(f"  [{mode}] Done — best pooled win_rate = {best_score:.3f}  "
          f"saved to {out_path}")
    return best


def _save(weights: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(weights, fh, indent=2)


# ── Public entry point ─────────────────────────────────────────────────────────

def run(
    mode: str = "both",
    iters: int = 40,
    games_per_eval: int = 4,
    seed: int = 0,
    out_dir: str | None = None,
    workers: int = 1,
    self_play: bool = False,
    pool_size: int = 4,
    max_steps: int | None = None,
) -> int:
    """Run CMA-ES training. Returns 0 on success."""
    if out_dir is None:
        out_dir = _MCTS_DIR

    modes = ["2p", "4p"] if mode == "both" else [mode]
    rng = random.Random(seed)
    method = "cmaes-selfplay" if self_play else "cmaes-vs-greedy"

    print(
        f"orbit-wars train  modes={modes}  iters={iters}  "
        f"games_per_eval={games_per_eval}  seed={seed}  "
        f"workers={workers}  method={method}  max_steps={max_steps or 'default'}"
    )
    t0 = time.monotonic()

    for m in modes:
        print(f"\n=== Training {m} ===")
        if self_play:
            _train_mode_cmaes_selfplay(
                m, iters, games_per_eval, rng, out_dir,
                workers=workers, pool_size=pool_size, max_steps=max_steps,
            )
        else:
            _train_mode_cmaes(m, iters, games_per_eval, rng, out_dir,
                              workers=workers, max_steps=max_steps)

    print(f"\nDone in {time.monotonic() - t0:.1f}s")
    return 0
