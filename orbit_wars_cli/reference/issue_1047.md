# Orbit Wars: Critical Game Logic Issues Report

Source: https://github.com/Kaggle/kaggle-environments/issues/1047
Author: @hassenhamdi, opened May 1

This document outlines verified bugs in `orbit_wars.py` that impact the logical
integrity, competitive fairness, and physical consistency of the simulation.

---

## 1. The "Black Hole" Fleet Destruction (Race Condition)

**Impact**: High (Permanent loss of assets)
**Location**: `interpreter()` lines 511, 593, and 606-607.

### Description

A race condition exists between fleet collision, comet expiration, and combat
resolution.

1. A fleet hits a comet scheduled to expire this turn and is added to
   `combat_lists[comet_id]`.
2. The comet is then removed from `obs0.planets` because it reached the end
   of its path.
3. Combat resolution attempts to find the planet by ID to apply damage. Since
   the planet is gone, the logic `if not planet: continue` triggers.

### Logical Impact

- **Points Deletion**: Ships in the attacking fleet are removed from the `fleets`
  list but never added to a planet or subtracted from a defender. They vanish
  from the "Total Score" entirely.
- **Paradox**: Strategically, it is better to miss an expiring comet than to hit it.
  A "miss" allows the player to keep their ships in the `fleets` list for the
  turn's final score tally, while a "hit" deletes them.
- **Unfair Defense**: The current owner of the comet is protected; their garrison
  is never fought, while the attacker's force is erased.

---

## 2. Geometric Tunneling (List-Order Collision Bias)

**Impact**: Medium (Violation of Physics)
**Location**: `interpreter()` lines 505-514.

### Description

The collision engine iterates through the `planets` list and `break`s at the
first match.

```python
for planet in obs0.planets:
    if point_to_segment_distance(...) < planet[4]:
        ...
        break
```

### Logical Impact

- **Physics Violation**: If two planets lie on the same trajectory, the fleet will
  always hit the one with the lower internal ID, even if the other planet is
  physically closer to the fleet's starting point.
- **Counter-Intuitive Play**: Players cannot "shield" a planet by placing a comet
  or another planet in front of it unless the shield has a lower list-index than
  the target.

---

## 4. Multi-Attacker Asset Erasure (N-Way Combat)

**Impact**: Medium (Strategic Narrowing)
**Location**: `interpreter()` lines 618-626.

### Description

The combat logic only processes the top two attacking forces.

```python
survivor_ships = top_ships - second_ships
```

### Logical Impact

- **Force Deletion**: Ships from a 3rd or 4th player involved in a battle are
  deleted from the simulation without dealing damage to the winner or the garrison.
- **Diplomatic Failure**: In a 4-player game, a 3rd player cannot "help" a 2nd
  player take down a dominant leader's planet by weakening the leader's incoming
  reinforcement; their ships simply disappear upon collision.

---

## 5. Temporal Desync (Inconsistent Step Defaults)

**Impact**: Medium (Simulation Consistency)
**Location**: `interpreter()` lines 402 and 524.

### Description

The `step` variable (the turn counter) is retrieved with different defaults in
the same function:

- Comet spawn check: `get(obs0, "step", 0)`
- Planet rotation: `get(obs0, "step", 1)`

### Logical Impact

- **Phase Shift**: If the environment runner doesn't provide a `step` key, comets
  and planets will operate on different timelines. A planet's position on "Turn 10"
  will correspond to the comet's position on "Turn 11," making unified trajectory
  projection impossible for agents.

---

## 6. Premature Termination (Off-by-One Error)

**Impact**: Low (Length Inconsistency)
**Location**: `interpreter()` line 645.

### Description

```python
if step >= configuration.episodeSteps - 2:
    terminated = True
```

### Logical Impact

- **Turn Loss**: In a 500-step game (steps 0-499), the game triggers termination
  at step 498. This prevents the final turn (Step 499) from resolving its movement
  and combat phases, truncating the game to 499 turns.

---

## 7. Coordinate Axis Discrepancy (X/Y Swap)

**Impact**: Critical for Agent Developers (Interface Bug)
**Location**: `generate_planets()` and `Planet` namedtuple.

### Description

The internal data structure stores coordinates as `[id, owner, Y, X]`, but the
`namedtuple` used by agents labels them as `Planet(id, owner, x, y, ...)`.

### Logical Impact

- **Trig Inversion**: Standard trigonometric functions (`math.cos`, `math.sin`)
  used by agents will be inverted. `Angle 0` (intended to be Right) actually
  moves the fleet Down. This contradicts the documentation and the visualizer's
  expected behavior, requiring every agent to implement a "manual swap" to function.
