# Tarok Project — Copilot Instructions

## SCORING — DO NOT CHANGE

The scoring model is final. Do not suggest, discuss, or implement:
- Zero-sum scoring
- Opponent score multiplication (e.g. declarer wins ×3 from 3 opponents)
- Any "real Tarok" scoring variant

Current rules:
- Normal games: declarer team gets +X or -X, opponents get 0
- Klop: each player scores individually
- This is intentional and correct

## Metrics

- **Avg Score** in the arena = total score / number of sessions (NOT per game)
- A session = N games (default 50). This is the unit players care about.
- Score history chart plots cumulative score per session.

## Architecture

- Single game engine path: always use `run_self_play` for arena games (supports all seat types: nn, bot_v5, bot_v6, bot_m6)
- Do NOT add `run_arena_games` as an alternative path. One implementation only.

## Clean Architecture Guardrails

- `tarok.use_cases` is the application core. Keep it free of direct infrastructure/data imports.
- Do NOT import `json`, `csv`, `pickle`, `numpy`, `pandas`, or anything from `tarok.adapters` inside `tarok.use_cases`.
- When new behavior needs I/O, serialization, storage, or integration work:
	- define/extend a Port in `tarok.ports`
	- implement the Port in `tarok.adapters`
	- inject the adapter through the port interface into the use case
- Keep business orchestration and game rules in use cases; keep framework/library details in adapters.

## Git

- NEVER use `--no-verify` when committing. Fix failing tests instead.
- The pre-commit hook runs `make lint-architecture` and the backend test suite. All must pass before committing.
