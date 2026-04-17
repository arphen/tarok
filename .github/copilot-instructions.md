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
