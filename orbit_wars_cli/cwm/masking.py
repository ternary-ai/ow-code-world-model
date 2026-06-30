"""
cwm/masking.py — Sun-occlusion legality precomputation (Module 4).

Determines which source→target planet pairs are blocked by the sun's exclusion
zone. Used by action_space.generate_candidates to filter out illegal launches.
"""

from __future__ import annotations

import numpy as np

from cwm.geometry import segment_circle_collision
from cwm.state import State, CENTER, SUN_RADIUS


def sun_blocks_path(
    source_pos: tuple[float, float],
    target_pos: tuple[float, float],
    sun_pos: tuple[float, float],
    sun_radius: float,
) -> bool:
    """True if the line segment from source_pos to target_pos passes within
    sun_radius of sun_pos at any point along the segment (not just endpoints).

    Uses strict distance < sun_radius, matching the game engine's sun-crossing
    check in cwm/geometry.py segment_circle_collision.
    """
    return segment_circle_collision(source_pos, target_pos, sun_pos, sun_radius)


def legal_pair_mask(state: State) -> np.ndarray:
    """Return an [num_planets, num_planets] boolean matrix.

    mask[i, j] is True where a straight path from planet i to planet j does
    not cross the sun's exclusion radius, and i != j.  The diagonal is always
    False (a planet cannot target itself).

    Uses the planets' current stored positions (state.planets[*][2], [3]).
    """
    planets = state.planets
    n = len(planets)
    mask = np.zeros((n, n), dtype=bool)
    sun_pos = (CENTER, CENTER)

    positions = [(p[2], p[3]) for p in planets]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if not sun_blocks_path(positions[i], positions[j], sun_pos, SUN_RADIUS):
                mask[i, j] = True

    return mask
