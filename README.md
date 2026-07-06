# Ternary Capital — Code World Model

**Agent-native finance. Build for markets.**

A high-fidelity code world model for the Orbit Wars environment — a real-time strategy benchmark used to develop and evaluate Ternary's planning and search-based agent architectures.

---

## What this is

A hand-written Python simulator of the Orbit Wars game engine, verified step-for-step against recorded replays. It serves as the planning substrate for tree search algorithms (MCTS, CMA-ES), enabling agents to reason about future states without interacting with the live environment.

This is Ternary's implementation of the Code World Model architecture introduced by DeepMind — where an LLM or hand-authored simulator acts as the world model, and a classical search algorithm handles decision-making.

## Why a Code World Model

| Property | Neural World Model | Code World Model |
|---|---|---|
| Determinism | No — probabilistic | Yes — `f(state, action) → state` |
| Speed | Slow (model forward pass) | Fast (~0.5ms vs 500ms API) |
| Debuggability | Opaque | Readable Python |
| Hallucination | Possible | Impossible |

At a 300ms search budget, a 1000x speed advantage translates from 0.6 simulations/turn to 600 — enabling multi-turn planning, timing attacks, and position evaluation at depth.

## Architecture

- `main.py` — agent entry point; MCTS + CMA-ES loop
- `orbit_wars_cli/` — CLI interface for local game execution
- `orbit_wars_cli/reference/` — full game specification and observation reference

## Results

In internal arena play, the CWM-backed agent achieves a 100% win rate against random and greedy baselines, and a 72% win rate over the same MCTS architecture with untuned weights — isolating the contribution of CMA-ES weight discovery.

---

Part of [github.com/ternary-ai](https://github.com/ternary-ai) — open-source agents, models, synthetic data, prompt engineering, and scaffolding.

[contact@ternary.capital](mailto:contact@ternary.capital) · US · Japan · Est. 2020
