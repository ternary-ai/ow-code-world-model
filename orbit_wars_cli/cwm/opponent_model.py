"""
cwm/opponent_model.py — Archetype-biased ISMCTS determinization (Module 7).

Clusters opponent replay logs into k behavioral archetypes (e.g., aggressive,
passive, rush) and uses the fitted model to bias the ISMCTS determinization
sampler toward plausible opponent actions rather than uniform random.

Behavioral features extracted per replay:
  - first_attack_turn    : turn of the opponent's first non-empty action
  - avg_ships_per_fleet  : mean fleet size across all launches
  - attack_rate          : fraction of turns with at least one launch
  - avg_fleet_ships_early: mean fleet size in the first 20 turns

Clustering: sklearn KMeans on the 4-dimensional feature vector.
Classification: nearest-centroid (1-NN to cluster centres), returning a
  softmax probability distribution.  With no history → uniform prior.
Sampling: draw archetype from archetype_probs, then sample a (from_planet,
  angle, ships) move from that archetype's empirical distribution, conditioned
  on the state's legal actions.
"""

from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass, field

import numpy as np

from cwm.state import State, CENTER, SUN_RADIUS
from cwm.masking import sun_blocks_path


# ── Types ──────────────────────────────────────────────────────────────────────

@dataclass
class ReplayLog:
    """A recorded game's action sequence for one player."""
    player_id: int
    actions: list    # list of moves per turn: each entry is [] or [from_id, angle, ships]
    num_turns: int


@dataclass
class ArchetypeModel:
    """Fitted archetype model: k cluster centres + per-cluster statistics."""
    k: int
    cluster_centres: np.ndarray   # shape (k, n_features)
    scaler: object                # fitted StandardScaler
    # Per-cluster empirical action distribution:
    # ships_mean[i], ships_std[i], attack_rate[i]
    ships_mean: np.ndarray        # shape (k,)
    ships_std: np.ndarray         # shape (k,)
    attack_rate: np.ndarray       # shape (k,)  fraction of turns with a launch


# ── Feature extraction ─────────────────────────────────────────────────────────

def _extract_features(log: ReplayLog) -> np.ndarray:
    """Extract 4-dimensional behavioral feature vector from a ReplayLog."""
    actions = log.actions
    n = len(actions)

    launches = [a for a in actions if a]   # non-empty moves
    first_attack = next(
        (i for i, a in enumerate(actions) if a), n
    )
    avg_ships = (
        float(np.mean([a[2] for a in launches if len(a) >= 3]))
        if launches else 0.0
    )
    attack_rate = len(launches) / max(n, 1)
    early = [a[2] for a in launches[:20] if len(a) >= 3]
    avg_ships_early = float(np.mean(early)) if early else 0.0

    return np.array([first_attack, avg_ships, attack_rate, avg_ships_early],
                    dtype=np.float32)


def _per_cluster_stats(
    logs: list[ReplayLog], labels: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-cluster empirical action-distribution statistics."""
    ships_mean = np.zeros(k, dtype=np.float32)
    ships_std = np.ones(k, dtype=np.float32)
    attack_rate = np.zeros(k, dtype=np.float32)

    for cluster_id in range(k):
        cluster_logs = [l for l, lab in zip(logs, labels) if lab == cluster_id]
        if not cluster_logs:
            continue

        all_ships = []
        total_turns = 0
        attack_turns = 0
        for log in cluster_logs:
            for a in log.actions:
                total_turns += 1
                if a:
                    attack_turns += 1
                    if len(a) >= 3:
                        all_ships.append(a[2])

        if all_ships:
            ships_mean[cluster_id] = float(np.mean(all_ships))
            ships_std[cluster_id] = max(1.0, float(np.std(all_ships)))
        attack_rate[cluster_id] = attack_turns / max(total_turns, 1)

    return ships_mean, ships_std, attack_rate


# ── Public API ─────────────────────────────────────────────────────────────────

def cluster_archetypes(replay_logs: list[ReplayLog], k: int) -> ArchetypeModel:
    """Extract behavioral features and fit a k-cluster model.

    Returns a fitted ArchetypeModel with k archetypes.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    features = np.stack([_extract_features(log) for log in replay_logs])

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    km.fit(X)
    labels = km.labels_

    ships_mean, ships_std, attack_rate = _per_cluster_stats(replay_logs, labels, k)

    return ArchetypeModel(
        k=k,
        cluster_centres=km.cluster_centers_,  # in scaled space
        scaler=scaler,
        ships_mean=ships_mean,
        ships_std=ships_std,
        attack_rate=attack_rate,
    )


def classify_opponent(
    observed_history: list,
    model: ArchetypeModel,
) -> dict[int, float]:
    """Return probability distribution over archetypes given observed actions.

    With empty history: uniform prior.  Otherwise, compute soft nearest-centroid
    (inverse distance) in the feature space.
    """
    if not observed_history:
        uniform = 1.0 / model.k
        return {i: uniform for i in range(model.k)}

    # Build a partial ReplayLog from the observed history and extract features
    partial = ReplayLog(player_id=-1, actions=observed_history,
                        num_turns=len(observed_history))
    feat = _extract_features(partial).reshape(1, -1)
    feat_scaled = model.scaler.transform(feat)

    # Soft assignment: softmax of negative distances to each centroid
    dists = np.linalg.norm(model.cluster_centres - feat_scaled, axis=1)
    # Softmax over negative distances (temperature=1.0)
    neg_d = -dists
    neg_d -= neg_d.max()   # numerical stability
    exp_d = np.exp(neg_d)
    probs = exp_d / exp_d.sum()

    return {i: float(p) for i, p in enumerate(probs)}


def _legal_moves(state: State, player_id: int) -> list[tuple[int, int]]:
    """Return list of legal (from_planet_id, to_planet_id) pairs for player_id."""
    own_planets = [p for p in state.planets if p[1] == player_id and p[5] > 0]
    other_planets = [p for p in state.planets if p[0] not in {p2[0] for p2 in own_planets}]
    sun_pos = (CENTER, CENTER)

    pairs = []
    for src in own_planets:
        for tgt in other_planets:
            if not sun_blocks_path((src[2], src[3]), (tgt[2], tgt[3]),
                                   sun_pos, SUN_RADIUS):
                pairs.append((src[0], tgt[0]))
    return pairs


def sample_opponent_action(
    state: State,
    archetype_probs: dict[int, float],
    model: ArchetypeModel,
    rng: np.random.Generator,
) -> list:
    """Sample one plausible action for the opponent given archetype_probs.

    1. Sample archetype_id from archetype_probs.
    2. Sample ship count from that archetype's empirical distribution.
    3. Pick a random legal (source, target) pair; compute angle.
    4. Clamp ships to the legal garrison.

    Returns a list of moves [[from_id, angle, ships], ...] (may be empty).
    """
    # Resolve which player is the opponent (first player not 0 in state)
    opp_players = list({p[1] for p in state.planets if p[1] not in (-1, 0)})
    player_id = opp_players[0] if opp_players else 1

    # Sample archetype
    ids = list(archetype_probs.keys())
    probs = np.array([archetype_probs[i] for i in ids], dtype=np.float64)
    probs /= probs.sum()
    chosen_id = ids[int(rng.choice(len(ids), p=probs))]

    archetype_rate = float(model.attack_rate[chosen_id])
    # Decide whether to launch this turn based on archetype's attack rate
    if rng.random() > max(archetype_rate, 0.1):
        return []   # passive turn

    legal_pairs = _legal_moves(state, player_id)
    if not legal_pairs:
        return []

    # Sample ship count from the archetype's distribution (clamped to garrison)
    mean_ships = float(model.ships_mean[chosen_id])
    std_ships = float(model.ships_std[chosen_id])
    raw_ships = int(rng.normal(mean_ships, std_ships))

    # Pick a random legal pair
    pair_idx = int(rng.integers(len(legal_pairs)))
    from_id, to_id = legal_pairs[pair_idx]

    src = next(p for p in state.planets if p[0] == from_id)
    tgt = next(p for p in state.planets if p[0] == to_id)

    num_ships = max(1, min(raw_ships, src[5]))
    angle = math.atan2(tgt[3] - src[3], tgt[2] - src[2])

    return [[from_id, angle, num_ships]]
