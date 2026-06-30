"""
tests/test_combat.py — Tests for cwm/combat.py (resolve_combat) and
    cwm/interpreter.py (apply_fleet_launches via cwm_apply_joint_action).

OPTION A (strict top-2, fleet-only) confirmed against source.

3-attacker derivation (OPTION A):
  A=100, B=60, C=40 ships arriving at planet owned by D (garrison=30).
  Fleet battle (OPTION A):
    sorted: [(A,100), (B,60), (C,40)]
    top=A(100), second=B(60), survivor=40
    C's 40 ships VANISH — not subtracted from A, not added to D's defense.
  Fleet winner vs. garrison:
    survivor_owner=A ≠ planet.owner=D → garrison=30 -= 40 = -10 < 0
    → planet.owner=A, garrison=10
  Result: A captures with 10 ships. B and C lose all ships.
  This CONFIRMS OPTION A: if OPTION B (iterative) were correct, C would
  reduce A by 40 first (A=100-40=60), then B(60) ties A(60) → garrison
  untouched, D keeps planet. The correct source behavior is OPTION A.

4-attacker derivation (4p scenario, OPTION A):
  A=80, B=70, C=50, D=30 arriving at neutral planet (owner=-1, garrison=0).
  Fleet battle: sorted [(A,80),(B,70),(C,50),(D,30)]
    survivor = 80-70 = 10, survivor_owner = A
    C's 50 and D's 30 VANISH.
  Fleet winner vs. garrison: neutral (garrison=0), survivor_owner=A ≠ -1
    garrison=0 -= 10 = -10 < 0 → planet.owner=A, garrison=10
  Result: A captures with 10 ships.
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cwm.state import State, CENTER
from cwm.combat import resolve_combat
from cwm.interpreter import cwm_apply_joint_action, cwm_is_terminal, cwm_get_rewards


# ── Helpers ────────────────────────────────────────────────────────────────────

def _minimal_state(planets=None, fleets=None, num_players=2, step=0) -> State:
    planets = [list(p) for p in (planets or [])]
    return State(
        planets=planets,
        fleets=list(fleets or []),
        initial_planets=[list(p) for p in planets],
        comets=[],
        comet_planet_ids=[],
        step=step,
        next_fleet_id=10,
        angular_velocity=0.03,
        num_players=num_players,
        episode_steps=500,
        ship_speed=6.0,
        comet_speed=4.0,
    )


def _planet(pid, owner, ships, x=70.0, y=50.0, radius=2.0, prod=2):
    return [pid, owner, x, y, radius, ships, prod]


def _resolve(planets, combat_queue, num_players=2):
    """Shortcut: build state, call resolve_combat, return planet_map."""
    s = _minimal_state(planets=planets, num_players=num_players)
    out = resolve_combat(s, combat_queue)
    return {p[0]: p for p in out.planets}


# ── 2-attacker scenarios ───────────────────────────────────────────────────────

class TestTwoAttacker:

    def test_single_fleet_owner_conquers_neutral(self):
        """Single arriving fleet owner conquers a neutral planet."""
        planets = [_planet(0, -1, 0)]   # neutral, 0 garrison
        # A sends 50 ships
        q = {0: [(0, 50)]}
        pm = _resolve(planets, q)
        p = pm[0]
        assert p[1] == 0, "planet should be owned by player 0"
        assert p[5] == 50, "garrison should be 50"

    def test_single_fleet_owner_attacks_garrison_and_loses(self):
        """Single fleet owner attacks but garrison holds."""
        planets = [_planet(0, 1, 100)]  # player 1 owns, garrison=100
        q = {0: [(0, 30)]}              # player 0 attacks with 30
        pm = _resolve(planets, q)
        p = pm[0]
        assert p[1] == 1, "original owner should keep planet"
        assert p[5] == 70, "garrison should be reduced by 30"

    def test_single_fleet_owner_attacks_and_captures(self):
        """Single fleet overcomes garrison, captures planet."""
        planets = [_planet(0, 1, 20)]   # owner=1, garrison=20
        q = {0: [(0, 50)]}              # player 0 attacks with 50
        pm = _resolve(planets, q)
        p = pm[0]
        assert p[1] == 0, "player 0 should capture planet"
        assert p[5] == 30, "garrison should be 50-20=30"

    def test_reinforcement_same_owner(self):
        """Fleet from same owner adds to garrison."""
        planets = [_planet(0, 2, 40)]   # owner=2, garrison=40
        q = {0: [(2, 60)]}              # player 2 reinforces with 60
        pm = _resolve(planets, q)
        p = pm[0]
        assert p[1] == 2, "owner unchanged"
        assert p[5] == 100, "garrison should be 40+60=100"

    def test_two_fleet_owners_attacker_wins(self):
        """Two fleet owners: top wins, garrison fight after."""
        # A=100, B=40 arriving at neutral (garrison=0)
        # Fleet battle: survivor = 100-40 = 60, owner=A=player 0
        # vs garrison=0: 0-60=-60 < 0 → capture by player 0 with 60
        planets = [_planet(0, -1, 0)]
        q = {0: [(0, 100), (1, 40)]}
        pm = _resolve(planets, q)
        p = pm[0]
        assert p[1] == 0, "player 0 (top fleet) captures"
        assert p[5] == 60, "survivor=60 vs garrison=0 → captured with 60"

    def test_two_fleet_owners_exact_tie(self):
        """Top two fleet owners tie exactly: all fleet ships destroyed, garrison untouched."""
        # A=50, B=50 at planet owned by player 1 (garrison=30)
        # Fleet battle: tie → survivor=0
        # Garrison untouched.
        planets = [_planet(0, 1, 30)]
        q = {0: [(0, 50), (1, 50)]}
        pm = _resolve(planets, q)
        p = pm[0]
        assert p[1] == 1, "owner unchanged after fleet tie"
        assert p[5] == 30, "garrison untouched after fleet tie"

    def test_two_fleet_owners_loser_outright_loses(self):
        """Two fleet owners: loser's ships all gone, winner takes garrison fight."""
        # A=80, B=30 at planet owned by B (garrison=60)
        # Fleet battle: survivor=80-30=50, survivor_owner=A
        # vs garrison=60 (B's): 60-50=10 → garrison holds at 10, owner=B
        planets = [_planet(0, 1, 60)]
        q = {0: [(0, 80), (1, 30)]}
        pm = _resolve(planets, q)
        p = pm[0]
        assert p[1] == 1, "player 1 garrison holds"
        assert p[5] == 10, "garrison=60-50=10"

    def test_attacker_exact_match_garrison(self):
        """Fleet winner exactly matches garrison: result is 0 garrison, owner unchanged."""
        # A=70, B=20 at planet owned by C (garrison=50)
        # Fleet: survivor=70-20=50, owner=A
        # vs garrison=50: 50-50=0 → garrison=0, owner unchanged (C)
        planets = [_planet(0, 2, 50)]
        q = {0: [(0, 70), (1, 20)]}
        pm = _resolve(planets, q)
        p = pm[0]
        assert p[1] == 2, "owner unchanged when attacker exactly matches garrison"
        assert p[5] == 0, "garrison=0"

    def test_no_fleets_no_change(self):
        """Planet with no arriving fleets is unchanged."""
        planets = [_planet(0, 1, 50)]
        q = {0: []}
        pm = _resolve(planets, q)
        p = pm[0]
        assert p[1] == 1
        assert p[5] == 50

    def test_multiple_planets_resolved_independently(self):
        """Two planets each receive different fleet combats; both resolved correctly."""
        planets = [_planet(0, -1, 0), _planet(1, -1, 0)]
        # Planet 0: A=60 → captures with 60
        # Planet 1: A=40, B=40 → tie, garrison=0 unchanged (neutral stays neutral)
        q = {
            0: [(0, 60)],
            1: [(0, 40), (1, 40)],
        }
        pm = _resolve(planets, q)
        assert pm[0][1] == 0 and pm[0][5] == 60
        # After tie at planet 1: survivor=0, garrison=0 unchanged, owner remains -1
        assert pm[1][1] == -1 and pm[1][5] == 0


# ── 3-attacker test (OPTION A confirmation) ────────────────────────────────────

class TestThreeAttacker:

    def test_three_attackers_option_a_confirmed(self):
        """
        OPTION A CONFIRMED: strict top-2 fleet-only battle.

        Setup: A=100, B=60, C=40 arriving at planet owned by D (player 3)
               with garrison=30.

        Expected (OPTION A):
          Fleet battle: sorted [(A=0,100),(B=1,60),(C=2,40)]
            survivor = 100-60 = 40, owner=player 0
            C's 40 ships VANISH without affecting the outcome.
          vs garrison: 30 -= 40 = -10 < 0 → player 0 captures with 10 ships.

        If OPTION B (iterative) were correct:
          Round 1: A(100) vs C(40) → A=60
          Round 2: A(60) vs B(60) → TIE → survivor=0, garrison untouched.
          → Player 3 (D) keeps planet with 30 ships.
          (Different result — this test pins which option the source implements.)
        """
        planets = [_planet(0, 3, 30)]   # owner=player 3, garrison=30
        q = {0: [(0, 100), (1, 60), (2, 40)]}
        pm = _resolve(planets, q, num_players=4)
        p = pm[0]
        # OPTION A result:
        assert p[1] == 0, (
            f"OPTION A: player 0 (A=100) should capture. Got owner={p[1]}. "
            "If owner==3, OPTION B is implemented instead — check combat.py."
        )
        assert p[5] == 10, (
            f"OPTION A: garrison should be 100-60-30=10. Got {p[5]}."
        )

    def test_three_attackers_tie_in_top_two(self):
        """Top two fleet owners tie; 3rd attacker's ships still vanish."""
        # A=50, B=50, C=30 at neutral planet (garrison=0)
        # Fleet battle: A and B tie → survivor=0
        # C's 30 ships vanish
        # Garrison: untouched (0)
        planets = [_planet(0, -1, 0)]
        q = {0: [(0, 50), (1, 50), (2, 30)]}
        pm = _resolve(planets, q, num_players=4)
        p = pm[0]
        assert p[1] == -1, "still neutral after top-2 tie"
        assert p[5] == 0, "garrison unchanged"

    def test_three_attackers_third_largest_is_lost(self):
        """Verify 3rd attacker's ships don't reduce the winner's survivors."""
        # A=100, B=50, C=99 at neutral (garrison=0)
        # OPTION A: sorted [(A,100),(C,99),(B,50)]
        #   survivor = 100-99 = 1, owner=A
        #   B's 50 ships vanish
        # vs garrison=0: capture with 1 ship
        planets = [_planet(0, -1, 0)]
        q = {0: [(0, 100), (1, 50), (2, 99)]}
        pm = _resolve(planets, q, num_players=4)
        p = pm[0]
        assert p[1] == 0, "player 0 (A) captures"
        assert p[5] == 1, "survivor=100-99=1 (B's 50 ships had no effect)"


# ── 4-attacker test (4p scenario) ─────────────────────────────────────────────

class TestFourAttacker:

    def test_four_attackers_all_different_sizes(self):
        """
        4-player scenario: all 4 players' fleets collide at one neutral planet.

        Setup: A=80, B=70, C=50, D=30 at neutral (garrison=0).

        Expected (OPTION A):
          Fleet battle: sorted [(A,80),(B,70),(C,50),(D,30)]
            survivor = 80-70 = 10, owner=player 0 (A)
            C's 50 and D's 30 VANISH.
          vs garrison=0: capture with 10 ships.
        """
        planets = [_planet(0, -1, 0)]
        q = {0: [(0, 80), (1, 70), (2, 50), (3, 30)]}
        pm = _resolve(planets, q, num_players=4)
        p = pm[0]
        assert p[1] == 0, f"Player 0 (A=80) should capture. Got owner={p[1]}"
        assert p[5] == 10, f"survivor=80-70=10. Got garrison={p[5]}"

    def test_four_attackers_top_two_tie(self):
        """4p top-2 tie: all fleet ships lost, C and D ships also vanish, garrison untouched."""
        # A=60, B=60, C=40, D=20 at owned planet (garrison=25)
        # Fleet tie: survivor=0; C,D vanish; garrison untouched.
        planets = [_planet(0, 1, 25)]
        q = {0: [(0, 60), (1, 60), (2, 40), (3, 20)]}
        pm = _resolve(planets, q, num_players=4)
        p = pm[0]
        assert p[1] == 1, "owner unchanged after top-2 tie"
        assert p[5] == 25, "garrison untouched"

    def test_four_attackers_same_owner_reinforcement(self):
        """If all 4 fleets belong to the same owner: they reinforce (sum all)."""
        planets = [_planet(0, 2, 10)]   # owner=player 2, garrison=10
        q = {0: [(2, 20), (2, 30), (2, 15), (2, 5)]}
        pm = _resolve(planets, q, num_players=4)
        p = pm[0]
        assert p[1] == 2, "owner unchanged"
        assert p[5] == 80, "garrison = 10 + (20+30+15+5) = 80"

    def test_black_hole_planet_removed(self):
        """Fleet hits a planet that is subsequently removed from state (black hole).
        Arriving ships vanish — combat_queue entry is skipped when planet not found."""
        planets = [_planet(99, -1, 0)]
        # Put combat queue entry for a pid that is NOT in planets
        s = _minimal_state(planets=planets)
        # Override state's planet_map so planet 99 is absent during resolve
        removed_state = _minimal_state(planets=[])   # no planets
        q = {99: [(0, 50)]}                          # fleet targeted dead planet
        out = resolve_combat(removed_state, q)
        # Ships should vanish (no planet to apply to)
        assert len(out.planets) == 0


# ── Invalid launch params (via interpreter) ────────────────────────────────────

class TestInvalidLaunchParams:
    """Verify apply_fleet_launches silently drops all invalid moves.
    Tested via cwm_apply_joint_action since apply_fleet_launches is internal."""

    def _base_state(self):
        """State with one planet owned by player 0 with 50 ships."""
        return _minimal_state(
            planets=[_planet(0, 0, 50, x=70.0, y=50.0)],
            num_players=2,
        )

    def test_over_garrison_dropped(self):
        """Launching more ships than available is silently ignored."""
        s = self._base_state()
        action = [[0, 0.0, 100]]   # requests 100, only 50 available
        out = cwm_apply_joint_action(s, [action, []])
        p = next(p for p in out.planets if p[0] == 0)
        assert p[5] >= 0, "garrison should not go negative"
        # Fleet should NOT have been created (no valid launch)
        assert len(out.fleets) == 0

    def test_zero_ships_dropped(self):
        """Launching 0 ships is silently ignored."""
        s = self._base_state()
        out = cwm_apply_joint_action(s, [[[0, 0.0, 0]], []])
        assert len(out.fleets) == 0

    def test_negative_ships_dropped(self):
        """Launching negative ships is silently ignored."""
        s = self._base_state()
        out = cwm_apply_joint_action(s, [[[0, 0.0, -5]], []])
        assert len(out.fleets) == 0

    def test_unowned_planet_dropped(self):
        """Launching from a planet not owned by the player is ignored."""
        s = _minimal_state(planets=[_planet(0, 1, 50)])  # owned by player 1
        out = cwm_apply_joint_action(s, [[[0, 0.0, 25]], []])  # player 0 tries
        assert len(out.fleets) == 0

    def test_nonexistent_planet_id_dropped(self):
        """Launching from a planet ID that doesn't exist is ignored."""
        s = self._base_state()
        out = cwm_apply_joint_action(s, [[[999, 0.0, 10]], []])
        assert len(out.fleets) == 0

    def test_malformed_move_wrong_length_dropped(self):
        """Move with wrong element count is ignored."""
        s = self._base_state()
        out = cwm_apply_joint_action(s, [[[0, 0.0]], []])   # len 2, needs 3
        assert len(out.fleets) == 0

    def test_none_action_no_crash(self):
        """None action for a player does not raise."""
        s = self._base_state()
        out = cwm_apply_joint_action(s, [None, None])
        assert out is not None

    def test_empty_action_no_crash(self):
        """Empty list for all players does not raise."""
        s = self._base_state()
        out = cwm_apply_joint_action(s, [[], []])
        assert out is not None

    def test_valid_launch_creates_fleet(self):
        """Control: a valid launch DOES create a fleet and deducts garrison."""
        s = self._base_state()
        out = cwm_apply_joint_action(s, [[[0, 0.0, 25]], []])
        assert len(out.fleets) == 1, "valid launch should create fleet"
        p = next(p for p in out.planets if p[0] == 0)
        # garrison was 50, launched 25, production added (prod=2 for radius=2:1+ln2≈1.69→prod=2?)
        # Actually production in _planet is prod=2, so garrison after launch+production:
        # 50 - 25 + 2 = 27
        assert p[5] == 27, f"garrison should be 50-25+2=27, got {p[5]}"

    def test_exact_garrison_launch_allowed(self):
        """Launching exactly all ships (== garrison) is valid."""
        s = self._base_state()
        out = cwm_apply_joint_action(s, [[[0, 0.0, 50]], []])
        assert len(out.fleets) == 1, "exact garrison launch should be valid"


# ── cwm_is_terminal ────────────────────────────────────────────────────────────

class TestIsTerminal:

    def test_not_terminal_early_game(self):
        s = _minimal_state(
            planets=[_planet(0, 0, 50), _planet(1, 1, 50)], step=10
        )
        assert not cwm_is_terminal(s)

    def test_terminal_at_episode_steps_minus_2(self):
        """Terminates at step >= episodeSteps - 2 = 498 (not 500 or 499)."""
        s = _minimal_state(
            planets=[_planet(0, 0, 50), _planet(1, 1, 50)], step=498
        )
        s.episode_steps = 500
        assert cwm_is_terminal(s)

    def test_not_terminal_at_497(self):
        s = _minimal_state(
            planets=[_planet(0, 0, 50), _planet(1, 1, 50)], step=497
        )
        s.episode_steps = 500
        assert not cwm_is_terminal(s)

    def test_terminal_elimination_one_player_left(self):
        """Only player 0 has planets; player 1 eliminated → terminal."""
        s = _minimal_state(
            planets=[_planet(0, 0, 50), _planet(1, 0, 30)], step=10
        )
        assert cwm_is_terminal(s)

    def test_terminal_all_eliminated(self):
        """No player has planets or fleets (e.g., mutual destruction) → terminal."""
        s = _minimal_state(planets=[], step=10)
        s.fleets = []
        assert cwm_is_terminal(s)

    def test_not_terminal_two_players_active(self):
        """Two players have planets → not terminal."""
        s = _minimal_state(
            planets=[_planet(0, 0, 50), _planet(1, 1, 50)], step=10
        )
        assert not cwm_is_terminal(s)


# ── cwm_get_rewards ────────────────────────────────────────────────────────────

class TestGetRewards:

    def test_two_player_rewards(self):
        planets = [_planet(0, 0, 40), _planet(1, 1, 60)]
        s = _minimal_state(planets=planets, num_players=2)
        rewards = cwm_get_rewards(s)
        assert len(rewards) == 2
        assert rewards[0] == 40.0
        assert rewards[1] == 60.0

    def test_fleet_ships_included(self):
        planets = [_planet(0, 0, 30)]
        s = _minimal_state(planets=planets, num_players=2)
        s.fleets = [[0, 0, 75.0, 50.0, 0.0, 0, 20],   # player 0 fleet
                    [1, 1, 25.0, 50.0, 0.0, 0, 15]]    # player 1 fleet
        rewards = cwm_get_rewards(s)
        assert rewards[0] == 50.0   # 30 planet + 20 fleet
        assert rewards[1] == 15.0   # 15 fleet

    def test_neutral_planets_not_counted(self):
        planets = [_planet(0, -1, 99), _planet(1, 0, 20)]
        s = _minimal_state(planets=planets, num_players=2)
        rewards = cwm_get_rewards(s)
        assert rewards[0] == 20.0   # only owned planet
        assert rewards[1] == 0.0

    def test_four_player_rewards(self):
        planets = [
            _planet(0, 0, 10), _planet(1, 1, 20),
            _planet(2, 2, 30), _planet(3, 3, 40),
        ]
        s = _minimal_state(planets=planets, num_players=4)
        rewards = cwm_get_rewards(s)
        assert rewards == [10.0, 20.0, 30.0, 40.0]


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
