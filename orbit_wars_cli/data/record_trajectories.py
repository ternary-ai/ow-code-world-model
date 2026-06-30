"""
data/record_trajectories.py — Record ground-truth transitions from kaggle_environments.

Usage:
    python data/record_trajectories.py --num-players 2 [--games 15]
    python data/record_trajectories.py --num-players 4 [--games 15]

Saves data/trajectories/{2p,4p}/game_NNN.json.
Each file contains:
  {
    "episode_seed": int,           # for comet-spawn RNG reconstruction
    "num_players":  int,
    "config":       {key: value},  # episodeSteps, shipSpeed, cometSpeed, agentCount
    "transitions":  [
      {
        "t":           int,
        "obs_t":       dict,        # obs at step t  (obs.step == t)
        "joint_action": [[move,..], ..],  # per-player action lists
        "obs_t1":      dict,        # obs at step t+1 (obs.step == t+1)
      }, ...
    ]
  }

NOTE: comet spawn transitions (where (t+1) in [50,150,250,350,450]) are included.
      The episode_seed is saved so test_transitions.py can reconstruct the exact
      comet path RNG: random.Random(f"orbit_wars-comet-{episode_seed}-{t+1}").
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running from project root or from data/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _to_python(obj):
    """Recursively convert kaggle_environments Struct / nested objects to plain Python."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    # Dict-like (includes kaggle_environments.utils.Struct)
    try:
        keys = list(obj.keys())
        return {k: _to_python(obj[k]) for k in keys}
    except (AttributeError, TypeError):
        pass
    # Fallback: try __dict__
    if hasattr(obj, "__dict__"):
        return {k: _to_python(v) for k, v in obj.__dict__.items()
                if not k.startswith("_")}
    return obj


def record(num_players: int, num_games: int, out_dir: str) -> None:
    import kaggle_environments as ke
    from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent

    os.makedirs(out_dir, exist_ok=True)

    agents = [random_agent] * num_players

    for game_idx in range(num_games):
        env = ke.make(
            "orbit_wars",
            configuration={"agentCount": num_players},
        )
        steps = env.run(agents)

        episode_seed = env.info.get("seed", 0) if env.info else 0
        cfg = env.configuration
        config_dict = {
            "episodeSteps": int(cfg.episodeSteps),
            "shipSpeed":    float(cfg.shipSpeed),
            "cometSpeed":   float(cfg.cometSpeed),
            "agentCount":   int(cfg.agentCount),
        }

        # Action indexing: steps[t].action is the action that was executed by the
        # interpreter to PRODUCE steps[t] from steps[t-1].  So the correct triple
        # is: obs_t = steps[t].obs, joint_action = steps[t+1].action, obs_t1 = steps[t+1].obs.
        transitions = []
        for t in range(len(steps) - 1):
            obs_t  = _to_python(steps[t][0].observation)
            obs_t1 = _to_python(steps[t + 1][0].observation)
            joint_action = [
                _to_python(steps[t + 1][i].action) for i in range(num_players)
            ]
            transitions.append({
                "t":            t,
                "obs_t":        obs_t,
                "joint_action": joint_action,
                "obs_t1":       obs_t1,
            })

        record_data = {
            "episode_seed": episode_seed,
            "num_players":  num_players,
            "config":       config_dict,
            "transitions":  transitions,
        }

        out_path = os.path.join(out_dir, f"game_{game_idx:03d}.json")
        with open(out_path, "w") as f:
            json.dump(record_data, f, separators=(",", ":"))

        print(f"  game {game_idx:3d}: {len(transitions)} transitions → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Record Orbit Wars trajectories")
    parser.add_argument("--num-players", type=int, choices=[2, 4], required=True)
    parser.add_argument("--games", type=int, default=15)
    args = parser.parse_args()

    subdir = "2p" if args.num_players == 2 else "4p"
    out_dir = os.path.join(
        os.path.dirname(__file__), "trajectories", subdir
    )
    print(f"Recording {args.games} {args.num_players}p games → {out_dir}")
    record(args.num_players, args.games, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
