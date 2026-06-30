"""
cli/submit.py — `orbit-wars submit` subcommand.

Produces a single self-contained agent file suitable for Kaggle submission
by inlining all CWM and MCTS modules into main.py, then smoke-tests it.

Steps
-----
1. Read each source module in dependency order.
2. Strip intra-package imports (`from cwm.* import …`, `from mcts.* import …`).
3. Strip redundant `from __future__ import annotations` (keep one at top).
4. Concatenate into <out_dir>/main.py with a two-parameter `agent(obs, config)`
   entry point that caches state at step 0 and calls joint_action_mcts.
5. Load best_weights_{2p,4p}.json from mcts/ if present.
6. Smoke-test: exec() the generated file, feed it a real step-0 obs from a
   saved trajectory, call agent(obs, config), assert a valid action is returned.

Usage (via main.py):
    orbit-wars submit [--out-dir PATH] [--no-smoke-test]
"""
from __future__ import annotations

import importlib
import json
import os
import re
import sys
import types

# Ensure orbit_wars_cli/ is on sys.path when run directly.
_CLI_DIR  = os.path.dirname(__file__)
_REPO_DIR = os.path.join(_CLI_DIR, "..")
sys.path.insert(0, _REPO_DIR)

_MCTS_DIR = os.path.join(_REPO_DIR, "mcts")
_DATA_DIR  = os.path.join(_REPO_DIR, "data", "trajectories")

# ── Source files in dependency order ──────────────────────────────────────────
# Each tuple: (label, relative_path_from_repo_dir)
_SOURCE_FILES = [
    ("cwm/state",          "cwm/state.py"),
    ("cwm/geometry",       "cwm/geometry.py"),
    # New CWM modules (Modules 2–7); must precede the files that use them.
    ("cwm/masking",        "cwm/masking.py"),        # needs: state, geometry
    ("cwm/intercept",      "cwm/intercept.py"),       # needs: state, geometry
    ("cwm/action_space",   "cwm/action_space.py"),    # needs: state, geometry, masking, intercept
    ("cwm/event_graph",    "cwm/event_graph.py"),     # needs: state, geometry
    ("cwm/symmetry",       "cwm/symmetry.py"),        # needs: state
    ("cwm/opponent_model", "cwm/opponent_model.py"),  # needs: state, masking
    # Core game engine (unchanged order).
    ("cwm/comets",         "cwm/comets.py"),
    ("cwm/combat",         "cwm/combat.py"),
    ("cwm/interpreter",    "cwm/interpreter.py"),
    # MCTS stack.
    ("mcts/value_fn",      "mcts/value_fn.py"),       # needs: state, geometry, event_graph
    ("mcts/actions",       "mcts/actions.py"),        # needs: state, geometry, masking, intercept, action_space
    ("mcts/search",        "mcts/search.py"),         # needs: all above
]

# Patterns for imports that will be satisfied by inlining (strip these lines).
_INTERNAL_IMPORT_RE = re.compile(
    r"^\s*from\s+(cwm|mcts)\.\S+\s+import\b"
)

_FUTURE_IMPORT = "from __future__ import annotations\n"


def _process_file(path: str) -> str:
    """Read *path*, strip internal imports and module docstring, return body."""
    with open(path, encoding="utf-8") as fh:
        source = fh.read()

    lines = source.splitlines(keepends=True)
    out   = []
    in_module_docstring  = False
    docstring_done       = False
    in_internal_import   = False   # inside a multi-line internal import block
    i                    = 0

    while i < len(lines):
        line = lines[i]

        # ── Skip continuation of a multi-line internal import ─────────────────
        if in_internal_import:
            if ")" in line:
                in_internal_import = False
            i += 1
            continue

        # ── Strip leading module docstring ────────────────────────────────────
        if not docstring_done:
            stripped = line.strip()
            if not in_module_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
                quote = stripped[:3]
                in_module_docstring = True
                rest = stripped[3:]
                if rest.endswith(quote) and len(rest) >= 3:
                    docstring_done = True
                    in_module_docstring = False
                    i += 1
                    continue
                i += 1
                continue
            elif in_module_docstring:
                if '"""' in line or "'''" in line:
                    in_module_docstring = False
                    docstring_done = True
                i += 1
                continue
            else:
                docstring_done = True   # first non-docstring line hit

        # ── Strip `from __future__ import annotations` ────────────────────────
        if line.strip() == "from __future__ import annotations":
            i += 1
            continue

        # ── Strip internal package imports (single- and multi-line) ───────────
        if _INTERNAL_IMPORT_RE.match(line):
            # Multi-line import (opening paren without closing on same line)?
            if "(" in line and ")" not in line:
                in_internal_import = True
            i += 1
            continue

        out.append(line)
        i += 1

    return "".join(out).strip() + "\n"


# ── Agent weights loader (embedded into generated main.py) ───────────────────

_AGENT_HEADER = '''\
# =============================================================================
# orbit-wars agent entry point  (auto-generated by `orbit-wars submit`)
# =============================================================================
import json as _json
import os as _os
import random as _random
import time as _time

_OW_STATE_CACHE: dict = {}          # keyed by obs["player"]
_OW_WEIGHTS_CACHE: dict = {}        # "2p" / "4p" -> weights dict

_OW_DEFAULT_WEIGHTS = dict(DEFAULT_WEIGHTS)
_OW_DEFAULT_WEIGHTS.setdefault("mcts_budget_s", 0.10)

_OW_WEIGHTS_JSON_2P: dict | None = {WEIGHTS_2P}
_OW_WEIGHTS_JSON_4P: dict | None = {WEIGHTS_4P}


def _ow_load_weights(num_players: int) -> dict:
    key = "2p" if num_players == 2 else "4p"
    if key in _OW_WEIGHTS_CACHE:
        return _OW_WEIGHTS_CACHE[key]
    raw = _OW_WEIGHTS_JSON_2P if num_players == 2 else _OW_WEIGHTS_JSON_4P
    if raw:
        w = dict(raw)
    else:
        w = dict(_OW_DEFAULT_WEIGHTS)
    _OW_WEIGHTS_CACHE[key] = w
    return w


# Opening book: steps before which we skip MCTS and use greedy expansion.
# The correct early-game move is almost always "capture the nearest rich neutral"
# — MCTS wastes its entire budget rediscovering this. Skipping it banks the
# overage for the midgame where search actually matters.
_OPENING_STEPS = {2: 15, 4: 25}


def _opening_book(state, player_id):
    """Greedy opening: capture the closest neutrals first.

    In 1v1, flight time dominates — a nearby planet captured in 5 steps beats
    a richer one 30 steps away that the opponent can contest.  Distance² penalty
    strongly biases toward the nearest targets; production breaks ties only.
    """
    own = [p for p in state.planets if p[1] == player_id and p[5] > 0]
    if not own:
        return ()
    own.sort(key=lambda p: p[5], reverse=True)
    src = own[0]                              # largest garrison as launch pad
    neutrals = [p for p in state.planets if p[1] == -1]
    if not neutrals:
        return ()
    sx, sy = _current_pos(src, state)
    def _ob_score(t):
        tx, ty = _current_pos(t, state)
        dist2 = (tx - sx) ** 2 + (ty - sy) ** 2 + 0.01
        return t[4] / dist2                   # production / distance²
    targets = sorted(neutrals, key=_ob_score, reverse=True)[:2]
    return tuple((src[0], t[0], _RIGHT_SIZE_FIT) for t in targets)


def agent(obs, config):
    """Kaggle submission entry point."""
    t_start = _time.monotonic()
    player_id = obs["player"] if isinstance(obs, dict) else obs.player

    # --- detect num_players and cache at step 0 ---
    if player_id not in _OW_STATE_CACHE:
        _OW_STATE_CACHE[player_id] = {"num_players": None, "rng": _random.Random(player_id)}

    cache = _OW_STATE_CACHE[player_id]
    rng   = cache["rng"]

    state = state_from_obs(obs, config, cached_num_players=cache["num_players"])

    if cache["num_players"] is None:
        cache["num_players"] = state.num_players

    # Opening book: bypass MCTS entirely for early turns and bank the budget.
    if state.step < _OPENING_STEPS.get(state.num_players, 15):
        act = _opening_book(state, player_id)
        if act:
            return abstracted_to_concrete(state, player_id, act)

    weights = _ow_load_weights(state.num_players)
    budget  = weights.get("mcts_budget_s", 0.10)
    rov     = (obs.get("remainingOverageTime", 0.0) if isinstance(obs, dict)
               else getattr(obs, "remainingOverageTime", 0.0))
    # Per-turn wall-clock budget. The deadline is anchored to t_start (the true
    # start of this turn) so that state construction and candidate generation
    # are counted against the budget -- guaranteeing total turn time stays under
    # the competition's ~1 s limit even on slower hardware. The 0.60 cap leaves
    # ~0.40 s of headroom: right-sizing makes each simulation heavier, so the
    # loop's final iteration can overshoot the deadline by one full (deep)
    # simulation (~0.15 s measured); the cap plus that tail stays well under
    # 1 s, and rare single-turn spikes are absorbed by the overage bank.
    # Draw 30% of remaining overage so early-game turns (full ~60s bank) get
    # meaningfully more search without burning the bank too fast.
    budget   = min(budget + rov * 0.3, 0.60)
    deadline = t_start + budget

    act = joint_action_mcts(
        state, player_id,
        weights=weights,
        num_players=state.num_players,
        rng=rng,
        deadline=deadline,
    )
    return abstracted_to_concrete(state, player_id, act)
'''


def _load_weights_json(mode: str) -> str:
    """Return a Python-literal string of the weights for embedding in the agent.

    The weights file is JSON, whose `true`/`false`/`null` tokens are not valid
    Python. Parse it and emit a Python `repr()` so booleans (e.g. the Phase 4
    `fine_fractions` flag) become `True`/`False` rather than bare `true`/`false`.
    """
    path = os.path.join(_MCTS_DIR, f"best_weights_{mode}.json")
    if os.path.exists(path):
        with open(path) as fh:
            return repr(json.load(fh))
    return "None"


def generate(out_path: str) -> None:
    """Write the inlined agent to *out_path*."""
    sections = [_FUTURE_IMPORT, "\n"]

    for label, rel_path in _SOURCE_FILES:
        abs_path = os.path.join(_REPO_DIR, rel_path)
        body = _process_file(abs_path)
        sections.append(f"\n# ── {label} {'─' * (55 - len(label))}\n\n")
        sections.append(body)
        sections.append("\n")

    # Embed best weights (or None) into the agent header.
    w2p = _load_weights_json("2p")
    w4p = _load_weights_json("4p")
    header = _AGENT_HEADER.replace("{WEIGHTS_2P}", w2p).replace("{WEIGHTS_4P}", w4p)
    sections.append("\n")
    sections.append(header)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("".join(sections))

    size_kb = os.path.getsize(out_path) / 1024
    print(f"  Generated {out_path}  ({size_kb:.1f} KB)")


def smoke_test(out_path: str) -> bool:
    """Exec the generated file and call agent() on a real trajectory obs.

    Returns True if the smoke test passes.
    """
    print("  Smoke-testing agent …")

    # Load a real step-0 obs from the first 2p trajectory.
    traj_2p = os.path.join(_DATA_DIR, "2p")
    game_file = sorted(f for f in os.listdir(traj_2p) if f.endswith(".json"))[0]
    with open(os.path.join(traj_2p, game_file)) as fh:
        game = json.load(fh)

    obs_dict    = game["transitions"][5]["obs_t"]   # mid-game enough to have ships
    config_dict = game["config"]

    # Build SimpleNamespace-style objects matching what kaggle passes.
    from types import SimpleNamespace

    def _to_ns(obj):
        # Kaggle only shallowly converts the top-level obs dict to a
        # SimpleNamespace; nested dicts (e.g. comet groups) stay as dicts.
        # Matching that behaviour here avoids false failures in the smoke test.
        if isinstance(obj, dict):
            return SimpleNamespace(**obj)
        return obj

    obs_ns    = _to_ns(obs_dict)
    config_ns = _to_ns(config_dict)
    # Also set obs.player (it's per-agent in kaggle, player 0 here).
    obs_ns.player = 0

    # Exec the file in a fresh module namespace.
    with open(out_path, encoding="utf-8") as fh:
        src = fh.read()

    ns: dict = {}
    try:
        exec(compile(src, out_path, "exec"), ns)  # noqa: S102
    except Exception as exc:
        print(f"  FAIL: exec() raised {exc}", file=sys.stderr)
        return False

    agent_fn = ns.get("agent")
    if agent_fn is None:
        print("  FAIL: `agent` function not found in generated file", file=sys.stderr)
        return False

    try:
        result = agent_fn(obs_ns, config_ns)
    except Exception as exc:
        print(f"  FAIL: agent() raised {exc}", file=sys.stderr)
        return False

    # Validate result: must be a list (possibly empty).
    if not isinstance(result, list):
        print(f"  FAIL: agent() returned {type(result)}, expected list", file=sys.stderr)
        return False

    # If non-empty, each move must be a list of length 3.
    for move in result:
        if not (isinstance(move, list) and len(move) == 3):
            print(f"  FAIL: invalid move {move}", file=sys.stderr)
            return False

    print(f"  PASS: agent() returned {len(result)} move(s): {result}")
    return True


# ── Public entry point ─────────────────────────────────────────────────────────

def run(out_dir: str | None = None, no_smoke_test: bool = False) -> int:
    """Generate main.py and optionally smoke-test it. Returns 0 on success."""
    if out_dir is None:
        out_dir = os.path.join(_REPO_DIR, "..")   # workspace root
    out_path = os.path.join(out_dir, "main.py")

    print(f"orbit-wars submit  out={out_path}")

    generate(out_path)

    if no_smoke_test:
        print("  Smoke test skipped.")
        return 0

    ok = smoke_test(out_path)
    return 0 if ok else 1
