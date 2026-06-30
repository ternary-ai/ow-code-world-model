"""
mcts/value_fn.py — Heuristic value function for Orbit Wars MCTS.

cwm_value_function(state, player_id, weights, num_players) -> float

Returns a value in [0, 1] where 1.0 = winning heavily, 0.0 = losing.

Features (each a [0, 1] ratio of own vs aggregated-opponent quantity)
--------------------------------------------------------------------
material    : own_ships     / (own_ships     + agg_opp_ships)
              total ships including garrisons AND in-transit fleets.
production  : own_production / (own_production + agg_opp_production)
              economic engine — ships generated per turn from owned planets.
control     : own_planets   / (own_planets   + agg_opp_planets)
              raw board presence / resilience (planet count, neutrals excluded).
offense     : own_fleet_ships / (own_fleet_ships + agg_opp_fleet_ships)
              initiative — ships already committed in transit toward targets.

Strategic-guidance features (DEFAULT OFF, weight 0 — only active when tuned in):
cohesion    : own-planet compactness ratio vs opponents (clustered planets =
              short reinforcement paths; Strategic Principle 2).
centrality  : own production weighted toward the board centre, ratio vs opp
              (map control / strategic position; Principles 1-2).
threat      : fraction of own production NOT exposed to a nearby enemy planet
              (safe economic core; Principles 2 & 7). 1 = safe, 0 = exposed.
anti_leader : multiplayer (num_players > 2) anti-snowball term. The visible
              material leader gets focus-fired in 4p, so being the premature
              leader is dangerous. 1 = not the leader (safe); drops toward 0 as
              our material lead over the strongest opponent grows. The penalty
              fades to ~0 over the course of the game (an early lead is risky, a
              lead near step 499 is just winning), and is neutral (0.5) in 2p.
neutral_access : expansion-opportunity advantage. For each unclaimed non-comet
              planet, measure access as production/(1+distance) for us vs
              opponents. Ratio of our access to opponents'. High = we are
              better placed to grab neutral planets.
incoming_threat : directed fleet-balance feature. Own fleet ships whose target
              planet (nearest planet by angle) is enemy-owned, vs. enemy fleet
              ships targeting our planets. High = we are attacking more than
              being attacked (distinct from offense, which is undirected).
time_fleet  : time-discounted in-transit ships. Like offense but fleet ships are
              weighted by 1/(1+steps_to_arrival) so nearby fleets count more
              than distant ones.
prod_density : production per total ship (own_prod / (own_ships+1)) ratio vs
              opponents. Captures economic leverage: more production-per-ship
              means better long-term growth even when behind on raw count.
phase_material : material advantage weighted by game phase (step/episode_steps).
              Neutral 0.5 early-game; equals material in the final turn.
              Lets the agent emphasise position early, material late.

The final value is a weight-normalized blend:
    value = Σ wᵢ·fᵢ / Σ wᵢ
so adding features never pushes the result outside [0, 1].

opp_aggregation controls how opponent values are combined:
  'sum'  (default) — total across all opponents  (2p: equivalent to zero-sum)
  'max'            — most dangerous single opponent
  'mean'           — average opponent

Terminal shortcut: own_ships / sum_all_ships (or 0.5 if nobody has ships).

DEFAULT_WEIGHTS
  'opp_aggregation':   'sum'
  'w_material':         0.45
  'w_production':       0.20
  'w_control':          0.25
  'w_offense':          0.10
  'w_cohesion':         0.0   (strategic-guidance, default off)
  'w_centrality':       0.0   (strategic-guidance, default off)
  'w_threat':           0.0   (strategic-guidance, default off)
  'w_anti_leader':      0.0   (multiplayer anti-snowball, default off)
  'w_neutral_access':   0.0   (expansion opportunity, default off)
  'w_incoming_threat':  0.0   (directed fleet balance, default off)
  'w_time_fleet':       0.0   (time-discounted fleets, default off)
  'w_prod_density':     0.0   (economic leverage, default off)
  'w_phase_material':   0.0   (late-game material emphasis, default off)

Backward compatibility: any weight absent from the supplied dict falls back to
its DEFAULT_WEIGHTS value, so older 2-feature weight dicts transparently gain
the new features at their default strength.
"""
from __future__ import annotations

import math

from cwm.state import State, CENTER
from cwm.geometry import fleet_speed
from cwm.event_graph import extract_events, encode_events

DEFAULT_WEIGHTS: dict = {
    "opp_aggregation": "sum",
    "w_material":       0.45,
    "w_production":     0.20,
    "w_control":        0.25,
    "w_offense":        0.10,
    # Strategic-guidance features; default OFF (weight 0) so existing tuned
    # weight dicts are unaffected (the weight-normalised blend ignores them).
    "w_cohesion":        0.0,
    "w_centrality":      0.0,
    "w_threat":          0.0,
    "w_anti_leader":     0.0,
    "w_neutral_access":  0.0,
    "w_incoming_threat": 0.0,
    "w_time_fleet":      0.0,
    "w_prod_density":    0.0,
    "w_phase_material":  0.0,
    "w_event_fleet":     0.0,   # fleet arrival balance (Module 5); default OFF
}


def _agg(values: list[float], method: str) -> float:
    """Aggregate a list of per-opponent values."""
    if not values:
        return 0.0
    if method == "max":
        return max(values)
    if method == "mean":
        return sum(values) / len(values)
    return sum(values)   # "sum" (default)


def _ratio(own: float, agg_opp: float) -> float:
    """own / (own + agg_opp), guarding the degenerate 0/0 case as neutral 0.5."""
    denom = own + agg_opp
    if denom <= 0.0:
        return 0.5
    return own / denom


def cwm_value_function(
    state: State,
    player_id: int,
    weights: dict | None = None,
    num_players: int | None = None,
) -> float:
    """Evaluate *state* from *player_id*'s perspective.

    Parameters
    ----------
    state : State
        Game state to evaluate.
    player_id : int
        The player we are computing value for.
    weights : dict | None
        Evaluation weights; falls back to DEFAULT_WEIGHTS.
        Keys: 'opp_aggregation', 'w_material', 'w_production',
        'w_control', 'w_offense'.
    num_players : int | None
        Override if different from state.num_players.

    Returns
    -------
    float in [0, 1].
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if num_players is None:
        num_players = state.num_players

    opp_ids = [i for i in range(num_players) if i != player_id]
    agg_method = weights.get("opp_aggregation", "sum")

    # ── Single-pass per-player accumulation ──────────────────────────────────
    n = num_players
    ships   = [0.0] * n   # garrisons + fleets (total military)
    prod    = [0.0] * n   # production from owned planets
    planets = [0.0] * n   # owned planet count (board presence)
    fleets  = [0.0] * n   # ships in transit (initiative)
    # Accumulators for the strategic-guidance features.
    cnt     = [0.0] * n   # owned planet count (float, for variance means)
    sum_x   = [0.0] * n   # Σ x  (per-player centroid)
    sum_y   = [0.0] * n   # Σ y
    sum_x2  = [0.0] * n   # Σ x² (for spread / cohesion)
    sum_y2  = [0.0] * n   # Σ y²
    centr   = [0.0] * n   # Σ production · closeness-to-board-centre (centrality)

    for p in state.planets:
        owner = p[1]
        if 0 <= owner < n:
            px, py, pships, pprod = p[2], p[3], p[5], p[6]
            ships[owner]   += pships
            prod[owner]    += pprod
            planets[owner] += 1.0
            cnt[owner]     += 1.0
            sum_x[owner]   += px
            sum_y[owner]   += py
            sum_x2[owner]  += px * px
            sum_y2[owner]  += py * py
            dc = math.hypot(px - CENTER, py - CENTER)
            centr[owner]   += pprod / (1.0 + dc)
    for f in state.fleets:
        owner = f[1]
        if 0 <= owner < n:
            ships[owner]  += f[6]
            fleets[owner] += f[6]

    own_ships = ships[player_id]
    sum_all   = sum(ships)

    if sum_all <= 0.0:
        return 0.5          # degenerate: nobody has ships

    # Terminal-ish shortcut: if all opponents are wiped, value is decisive.
    agg_opp_ships = _agg([ships[i] for i in opp_ids], agg_method)
    if agg_opp_ships <= 0.0:
        return 1.0

    material   = _ratio(own_ships, agg_opp_ships)
    production = _ratio(prod[player_id],
                        _agg([prod[i] for i in opp_ids], agg_method))
    control    = _ratio(planets[player_id],
                        _agg([planets[i] for i in opp_ids], agg_method))
    offense    = _ratio(fleets[player_id],
                        _agg([fleets[i] for i in opp_ids], agg_method))

    w_mat  = weights.get("w_material",   DEFAULT_WEIGHTS["w_material"])
    w_prod = weights.get("w_production", DEFAULT_WEIGHTS["w_production"])
    w_ctrl = weights.get("w_control",    DEFAULT_WEIGHTS["w_control"])
    w_off  = weights.get("w_offense",    DEFAULT_WEIGHTS["w_offense"])
    w_coh  = weights.get("w_cohesion",   DEFAULT_WEIGHTS["w_cohesion"])
    w_cen  = weights.get("w_centrality", DEFAULT_WEIGHTS["w_centrality"])
    w_thr  = weights.get("w_threat",     DEFAULT_WEIGHTS["w_threat"])
    w_anti = weights.get("w_anti_leader", DEFAULT_WEIGHTS["w_anti_leader"])

    # ── Strategic-guidance features (only computed when weighted) ─────────────
    # cohesion: clustered own planets (short reinforcement paths). Compactness
    # = 1 / (1 + RMS spread about own centroid); compared as a ratio vs opp.
    cohesion = 0.5
    if w_coh != 0.0:
        def _compact(i: int) -> float:
            c = cnt[i]
            if c <= 0.0:
                return 0.0
            var = (sum_x2[i] - sum_x[i] * sum_x[i] / c
                   + sum_y2[i] - sum_y[i] * sum_y[i] / c) / c
            if var < 0.0:
                var = 0.0
            return 1.0 / (1.0 + math.sqrt(var))
        cohesion = _ratio(_compact(player_id),
                          _agg([_compact(i) for i in opp_ids], agg_method))

    # centrality: own production weighted toward the board centre (map control).
    centrality = 0.5
    if w_cen != 0.0:
        centrality = _ratio(centr[player_id],
                            _agg([centr[i] for i in opp_ids], agg_method))

    # threat: fraction of own production that is NOT exposed to a nearby enemy
    # (1 = safe core, 0 = production sitting next to enemies). O(P²), so only
    # paid when the tuner selects this feature.
    threat = 0.5
    if w_thr != 0.0:
        own_pl = [p for p in state.planets if p[1] == player_id]
        enemy_pl = [p for p in state.planets if 0 <= p[1] < n and p[1] != player_id]
        own_prod_total = prod[player_id]
        if own_pl and enemy_pl and own_prod_total > 0.0:
            exposed = 0.0
            for p in own_pl:
                d_min = min(math.hypot(p[2] - e[2], p[3] - e[3]) for e in enemy_pl)
                exposure = 1.0 / (1.0 + d_min)        # near enemy → ~1
                exposed += p[6] * exposure
            threat = 1.0 - min(1.0, exposed / own_prod_total)
        else:
            threat = 1.0                              # no enemies → fully safe

    # anti_leader: in multiplayer, penalise being the visible material leader
    # (which draws focus-fire). 1 = not the strongest; falls toward 0 as our
    # lead over the strongest opponent grows, with the penalty fading to ~0 by
    # game end (a late lead is just winning). Neutral 0.5 in 2p (where leading
    # IS the objective and there is no third party to gang up).
    anti_leader = 0.5
    if w_anti != 0.0:
        if n <= 2:
            # Two-player: there is no third party to gang up, so anti-snowball
            # is meaningless. Mirror material so a non-zero weight is harmless
            # redundancy (just reweights material) rather than a damping 0.5.
            anti_leader = material
        else:
            max_opp = max((ships[i] for i in opp_ids), default=0.0)
            if own_ships > max_opp and (own_ships + max_opp) > 0.0:
                lead = (own_ships - max_opp) / (own_ships + max_opp)
                horizon = state.episode_steps if state.episode_steps > 0 else 500
                remaining = (horizon - state.step) / horizon
                if remaining < 0.0:
                    remaining = 0.0
                elif remaining > 1.0:
                    remaining = 1.0
                anti_leader = 1.0 - lead * remaining
            else:
                anti_leader = 1.0                     # not the leader → safe

    w_neu = weights.get("w_neutral_access",  DEFAULT_WEIGHTS["w_neutral_access"])
    w_inc = weights.get("w_incoming_threat", DEFAULT_WEIGHTS["w_incoming_threat"])
    w_tfl = weights.get("w_time_fleet",      DEFAULT_WEIGHTS["w_time_fleet"])
    w_prd = weights.get("w_prod_density",    DEFAULT_WEIGHTS["w_prod_density"])
    w_phm = weights.get("w_phase_material",  DEFAULT_WEIGHTS["w_phase_material"])

    # ── neutral_access: expansion-opportunity advantage ───────────────────────
    # Sum production/(1+distance) to each unclaimed non-comet planet for own
    # nearest planet vs each opponent's nearest planet, then compare as ratio.
    neutral_access = 0.5
    if w_neu != 0.0:
        comet_ids   = state.comet_planet_ids
        own_pos     = [(p[2], p[3]) for p in state.planets if p[1] == player_id]
        opp_pos     = [[(p[2], p[3]) for p in state.planets if p[1] == i]
                       for i in range(n)]
        neu_own = 0.0
        neu_opp = [0.0] * n
        for p in state.planets:
            if p[1] != -1 or p[0] in comet_ids:
                continue
            pprod, px, py = p[6], p[2], p[3]
            d_own = (min(math.hypot(px - ox, py - oy) for ox, oy in own_pos)
                     if own_pos else 1e9)
            neu_own += pprod / (1.0 + d_own)
            for i in opp_ids:
                d_i = (min(math.hypot(px - ox, py - oy) for ox, oy in opp_pos[i])
                        if opp_pos[i] else 1e9)
                neu_opp[i] += pprod / (1.0 + d_i)
        neutral_access = _ratio(neu_own, _agg([neu_opp[i] for i in opp_ids], agg_method))

    # ── incoming_threat + time_fleet: per-fleet directed analysis ─────────────
    # For each fleet, find the planet whose bearing most closely matches the
    # fleet's angle (proxy target). Use a 45° cone — beyond that, ignore.
    # incoming_threat : own fleets targeting enemy planets vs enemy fleets targeting ours.
    # time_fleet      : same as offense but ships discounted by steps-to-arrival.
    _ANGLE_THR = math.pi / 4.0
    own_directed   = 0.0
    opp_directed   = [0.0] * n
    own_discounted = 0.0
    opp_discounted = [0.0] * n

    if w_inc != 0.0 or w_tfl != 0.0:
        non_comet_pl = [p for p in state.planets if p[0] not in state.comet_planet_ids]
        for f in state.fleets:
            fowner = f[1]
            if not (0 <= fowner < n):
                continue
            fx, fy, fangle, fships = f[2], f[3], f[4], f[6]
            best_p   = None
            best_da  = float("inf")
            for p in non_comet_pl:
                ang_to_p = math.atan2(p[3] - fy, p[2] - fx)
                da = abs(math.atan2(math.sin(fangle - ang_to_p),
                                    math.cos(fangle - ang_to_p)))
                if da < best_da:
                    best_da = da
                    best_p  = p
            if best_p is None or best_da > _ANGLE_THR:
                continue
            tgt_owner = best_p[1]
            if w_inc != 0.0:
                if fowner == player_id and tgt_owner != player_id:
                    own_directed += fships
                elif fowner != player_id and tgt_owner == player_id:
                    opp_directed[fowner] += fships
            if w_tfl != 0.0:
                dist  = math.hypot(best_p[2] - fx, best_p[3] - fy)
                steps = dist / max(fleet_speed(fships), 0.1)
                disc  = fships / (1.0 + steps)
                if fowner == player_id:
                    own_discounted += disc
                else:
                    opp_discounted[fowner] += disc

    incoming_threat = 0.5
    if w_inc != 0.0:
        incoming_threat = _ratio(own_directed,
                                 _agg([opp_directed[i] for i in opp_ids], agg_method))

    time_fleet = 0.5
    if w_tfl != 0.0:
        time_fleet = _ratio(own_discounted,
                            _agg([opp_discounted[i] for i in opp_ids], agg_method))

    # ── prod_density: production per total ship (economic leverage) ───────────
    prod_density = 0.5
    if w_prd != 0.0:
        own_eff  = prod[player_id] / (ships[player_id] + 1.0)
        opp_effs = [prod[i] / (ships[i] + 1.0) for i in opp_ids]
        prod_density = _ratio(own_eff, _agg(opp_effs, agg_method))

    # ── phase_material: material weighted by game progress ────────────────────
    # Neutral (0.5) in the first turn; equals material at the final turn.
    phase_material = 0.5
    if w_phm != 0.0:
        ep    = state.episode_steps if state.episode_steps > 0 else 500
        phase = min(1.0, max(0.0, state.step / ep))
        phase_material = 0.5 * (1.0 - phase) + material * phase

    # ── event_fleet: net incoming fleet advantage over a short horizon ────────
    # Encodes in-transit fleets as a signed ship-delta (positive = our ships
    # arriving, negative = enemy ships arriving).  Captures imminent threats and
    # commitments that raw offense misses (Module 5: cwm.event_graph).
    w_evf = weights.get("w_event_fleet", DEFAULT_WEIGHTS["w_event_fleet"])
    event_fleet = 0.5
    if w_evf != 0.0:
        _horizon = 20
        _n_pl    = len(state.planets)
        _events  = extract_events(state, _horizon)
        _arr     = encode_events(_events, _n_pl, _horizon,
                                 observing_player=player_id, base_turn=state.step)
        _flat = _arr[:, :, 0]              # signed ship deltas
        _our  = float(_flat[_flat > 0].sum()) if (_flat > 0).any() else 0.0
        _opp  = float((-_flat)[_flat < 0].sum()) if (_flat < 0).any() else 0.0
        event_fleet = _ratio(_our, _opp)

    w_sum = (w_mat + w_prod + w_ctrl + w_off + w_coh + w_cen + w_thr + w_anti
             + w_neu + w_inc + w_tfl + w_prd + w_phm + w_evf)
    if w_sum <= 0.0:
        return material     # degenerate weights: fall back to material

    return (w_mat * material + w_prod * production
            + w_ctrl * control + w_off * offense
            + w_coh * cohesion + w_cen * centrality + w_thr * threat
            + w_anti * anti_leader + w_neu * neutral_access
            + w_inc * incoming_threat + w_tfl * time_fleet
            + w_prd * prod_density + w_phm * phase_material
            + w_evf * event_fleet) / w_sum

