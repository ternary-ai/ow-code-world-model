# Issue #1047 — Status vs. orbit_wars_original.py
# Verified against: reference/orbit_wars_original.py (fetched 2025-06, master branch)
# Verification method: direct source inspection of each claim.

---

## Item 1 — "Black Hole" Fleet Destruction (Race Condition)

**Status: STILL PRESENT**

Source evidence (lines ~590-640):
- During the fleet-movement phase, `group["path_index"] += 1` is done for all comets.
- If the new index >= len(path), the comet is added to `expired_comet_pids` and a
  placeholder entry is written to `planet_paths` (old_pos, old_pos, check=True).
  The comet is NOT yet removed from `obs0.planets` at this point.
- Fleets are then swept against this still-present comet, and hits are added to
  `combat_lists[comet_pid]`.
- Immediately AFTER fleet movement and BEFORE combat resolution, the code does:
    ```python
    if expired_comet_pids:
        obs0.planets = [p for p in obs0.planets if p[0] not in expired_set]
    ```
  This removes the comet from obs0.planets.
- In combat resolution: `planet = next((p for p in obs0.planets if p[0] == pid), None)`
  → returns None for the expired comet.
- `if not planet or not planet_fleets: continue` → attacking ships vanish silently.

**CWM decision**: Replicate source behavior faithfully (ships vanish). Noted in
interpreter.py. This matches trajectory recordings and keeps CWM consistent with
the actual environment used for scoring.

---

## Item 2 — Geometric Tunneling (List-Order Collision Bias)

**Status: STILL PRESENT**

Source evidence (fleet movement loop, ~lines 560-585):
```python
hit_planet = False
for planet in obs0.planets:
    path = planet_paths.get(planet[0])
    if path is None or not path[2]:
        continue
    p_old, p_new, _ = path
    if swept_pair_hit(old_pos, new_pos, p_old, p_new, planet[4]):
        combat_lists[planet[0]].append(fleet)
        fleets_to_remove.append(fleet)
        hit_planet = True
        break    # ← first-match wins regardless of distance
```
The source upgraded from `point_to_segment_distance` to `swept_pair_hit` (continuous
swept-pair test), but retained the `break`-on-first-match iteration order.

**CWM decision**: Replicate source behavior (first planet in list wins). Noted in
interpreter.py. Required for trajectory fidelity.

---

## Item 4 — Multi-Attacker Asset Erasure (N-Way Combat)

**Status: STILL PRESENT**

Source evidence (combat resolution, ~lines 618-640):
```python
sorted_players = sorted(
    player_ships.items(), key=lambda item: item[1], reverse=True
)
top_player, top_ships = sorted_players[0]

if len(sorted_players) > 1:
    second_ships = sorted_players[1][1]
    survivor_ships = top_ships - second_ships
    ...
```
Only the top-2 FLEET owners are evaluated. Groups 3 and 4 contribute to
`player_ships` but are never accessed — their ships vanish without effect on the
garrison or on the top-2 outcome.

IMPORTANT NUANCE: The garrison (planet[5], planet[1]) is NOT included in
`player_ships`. It is a SEPARATE subsequent fight:
```python
if survivor_ships > 0:
    if planet[1] == survivor_owner:
        planet[5] += survivor_ships
    else:
        planet[5] -= survivor_ships
        if planet[5] < 0:
            planet[1] = survivor_owner
            planet[5] = abs(planet[5])
```
So the actual combat model is:
  1. Fleet-only battle: top-2 fleet owners fight; 3rd+ fleet owners' ships vanish.
  2. Fleet winner vs. garrison: standard attacker/defender logic.
  3. Tie in fleet battle (top-2 equal): survivor_ships=0, garrison untouched.

This is "OPTION A (strict top-2, fleet-only)" in combat.py terms.

**CWM decision**: Implement exactly this. Documented in combat.py with source
line reference.

---

## Item 5 — Temporal Desync (Inconsistent Step Defaults)

**Status: STILL PRESENT**

Source evidence:
- Line ~402: `step = get(obs0, "step", 0)` (used for comet spawn check)
- Line ~524: `step = get(obs0, "step", 1)` (used for planet rotation)

If `step` is missing from obs (which doesn't happen in normal kaggle_environments
execution, as `step` is always populated), comets would operate on step 0 while
planets rotate as if on step 1. In practice, `step` is always present, so this
has no real-world gameplay impact.

**CWM decision**: CWM always passes step explicitly; `state_from_obs` populates
it from obs. The inconsistency is moot for our use. Noted for completeness.

---

## Item 6 — Premature Termination (Off-by-One Error)

**Status: STILL PRESENT**

Source evidence (line ~645):
```python
if step >= configuration.episodeSteps - 2:
    terminated = True
```
With `episodeSteps = 500` (default), terminates at step >= 498. The game therefore
runs steps 0-498 (499 turns), not 500. Step 499 never executes.

**CWM decision**: `cwm_is_terminal` uses `step >= config.episodeSteps - 2` to
exactly replicate source behavior. This is the termination condition recorded
in trajectories and used for scoring. Documented in interpreter.py.

---

## Item 7 — Coordinate Axis Discrepancy (X/Y Swap)

**Status: STILL PRESENT (but self-consistent; not a simulation correctness bug)**

Source evidence (generate_planets, lines ~103-110):
```python
x = CENTER + orbital_r * math.cos(angle)  # horizontal
y = CENTER + orbital_r * math.sin(angle)  # vertical
temp_planets = [
    [id_counter, -1, y, x, r, ships, prod],  # ← y stored at position 2 (namedtuple .x)
    [id_counter + 1, -1, BOARD_SIZE - x, y, ...],
    [id_counter + 2, -1, x, BOARD_SIZE - y, ...],
    [id_counter + 3, -1, BOARD_SIZE - y, BOARD_SIZE - x, ...],
]
```
For Q1 planets, field [2] (labeled `Planet.x`) stores the value computed as `y`
(sin-component) in polar generation, and field [3] (labeled `Planet.y`) stores
the `x` (cos-component). The swap is INTENTIONAL — it is required to achieve
4-fold 90°-rotational symmetry across all four quadrant copies while using the
same path array.

The entire interpreter uses planet[2] and planet[3] directly (not .x/.y names),
and fleet movement uses `math.cos(angle)` for delta[2] and `math.sin(angle)` for
delta[3]. The simulation is internally self-consistent.

README states `direction_angle: 0 = right, pi/2 = down` which describes the
internal coordinate convention (cos → [2], sin → [3]), not screen x/y. Agents
using `math.atan2(p2[3] - p1[3], p2[2] - p1[2])` (i.e., atan2(.y - .y, .x - .x))
will get correct directions IN THE SIMULATION'S COORDINATE SYSTEM.

**CWM decision**: Use planet[2]/planet[3] (or .x/.y from namedtuple) consistently
throughout all CWM modules. Never introduce a coordinate swap. This matches the
source and ensures trajectory reproducibility.
