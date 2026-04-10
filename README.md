# Slovenian Tarok

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![React 19](https://img.shields.io/badge/react-19-cyan.svg)
![FastAPI](https://img.shields.io/badge/fastapi-0.115+-green.svg)

A full-stack 4-player Slovenian Tarok card game with AI agents that learn by
self-play using deep reinforcement learning (PPO). Play against trained AI
opponents in real time, watch AI-vs-AI spectator games, train new agents from
the browser dashboard, or run hyperparameter evolution and behavioral breeding
experiments.

## Features

- **Play vs AI** — Human-vs-3-AI games with per-opponent model selection,
  adjustable AI speed, and a live game log
- **Spectator mode** — Watch 4 AI agents play each other with full hand
  visibility and step-by-step playback
- **Training dashboard** — Start/stop PPO training runs, monitor win rate,
  reward, loss, bid distributions, and per-contract stats in real-time charts
- **Hyperparameter evolution** — DEAP-based population search over learning
  rate, entropy coefficient, discount factor, etc.
- **Behavioral breeding** — Genetic selection for play-style specialisation
  (aggressive bidder, solo specialist, …)
- **Tournament bracket** — Round-robin or bracket tournaments between any mix
  of RL, lookahead, and random agents
- **High-performance Rust engine** — Optional PyO3-based engine for legal-move
  generation, trick evaluation, and state encoding (10–100× faster than pure
  Python)

## Architecture (Clean Architecture)

```
backend/src/tarok/
├── entities/          # Domain entities — Card, GameState, Scoring
│   ├── card.py        # Card, Suit, Deck (54 cards)
│   ├── game_state.py  # GameState, Phase, Contract, Trick
│   └── scoring.py     # Point counting, game scoring
├── engine/            # Rules engine — legal moves, conditions
├── ports/             # Interfaces (dependency inversion)
│   ├── player_port.py # PlayerPort — any player (human/AI/random)
│   └── observer_port.py # GameObserverPort — event notifications
├── use_cases/         # Application business rules
│   ├── game_loop.py   # Orchestrates full game flow
│   ├── deal.py        # Card dealing
│   ├── bid.py         # Bidding
│   ├── call_king.py   # King calling (2v2 partner selection)
│   ├── exchange_talon.py # Talon exchange
│   └── play_trick.py  # Trick play & resolution
└── adapters/          # Infrastructure
    ├── ai/            # RL agent, neural network, PPO trainer,
    │   │              #   lookahead (Monte Carlo), StockŠkis heuristic,
    │   │              #   random baseline, evo optimizer, breeding
    │   ├── agent.py
    │   ├── network.py
    │   ├── trainer.py
    │   ├── encoding.py
    │   ├── lookahead_agent.py
    │   ├── stockskis_player.py
    │   ├── random_agent.py
    │   ├── evo_optimizer.py
    │   └── breeding.py
    └── api/           # FastAPI REST + WebSocket
        ├── server.py
        ├── human_player.py
        ├── ws_observer.py
        └── schemas.py
```

**Dependency rule**: entities ← ports ← use_cases ← adapters. Inner layers
never import outer layers.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, TypeScript 5.6, Vite 6, Recharts |
| Backend | Python 3.12, FastAPI, Uvicorn, WebSockets |
| AI / ML | PyTorch 2.4+, PPO (actor-critic), DEAP |
| Engine | Rust (PyO3), optional high-perf game logic |
| Testing | Vitest + Testing Library (frontend unit), Playwright (E2E), pytest + pytest-bdd + Hypothesis (backend) |
| Dev | uv (pkg mgr), ruff (lint), pre-commit coverage gate |

## Quick Start

```bash
# One-time setup (macOS — installs Homebrew, Python, Node, uv, all deps)
make setup

# Start backend (port 8000) + frontend (port 3000)
make run
```

Open http://localhost:3000

## Testing

```bash
make test              # All tests (backend + frontend typecheck + frontend unit)
make test-backend      # Backend pytest suite
make test-frontend     # TypeScript type-check
make test-frontend-unit # Vitest component tests
make test-e2e          # Playwright end-to-end tests
make test-coverage     # Backend coverage report
make test-quick        # Backend, fail-fast
```

## Training

```bash
make train             # 100 sessions × 100 games (PPO self-play)
make evolve            # DEAP hyperparameter search
make train-evolved     # Continue training with best evolved hparams
make breed             # Behavioral specialisation breeding
make train-bred        # Continue with best bred model
```

Or use the in-browser Training Dashboard (click "Train AI" on the home page).

## Game Rules (Slovenian Tarok, 4-player)

- **54 cards**: 22 Taroks (I–XXI + Škis) + 32 suit cards (4 suits × 8)
- **4 players**, each dealt 12 cards, 6 go to the talon
- **Bidding**: Players bid contracts (Three → Two → One → Solo Three → Solo Two → Solo One → Solo → Berač)
- **King Calling**: Declarer calls a king — the holder becomes their secret 2v2 partner
- **Talon Exchange**: Declarer picks talon cards and discards equal number
- **12 Tricks**: Must follow suit; must play tarok if void in led suit
- **Scoring**: Cards counted in groups of 3. Declarer team needs ≥36 of 70 points

## AI Agent

- **PPO (Proximal Policy Optimization)** with actor-critic architecture
- **179-feature state encoding**: hand (54), played cards (54), current trick (54), talon, contract, position
- **Masked action space**: 54-card output head filtered to legal moves
- **4-agent self-play**: all agents share one network
- **Mixed training**: optionally pit RL agents against StockŠkis heuristic or Lookahead (Monte Carlo) opponents

## Makefile Reference

| Target | Description |
|--------|-------------|
| `make setup` | One-time bootstrap (Homebrew, Python, Node, deps, git hooks) |
| `make install` | Install dependencies only |
| `make run` | Start backend + frontend |
| `make stop` | Kill running servers |
| `make test` | Run all tests |
| `make test-backend` | Backend pytest |
| `make test-frontend` | Frontend typecheck |
| `make test-frontend-unit` | Frontend Vitest unit tests |
| `make test-e2e` | Playwright E2E |
| `make test-coverage` | Backend coverage report |
| `make build` | Production frontend build |
| `make train` | Train RL agents |
| `make evolve` | Hyperparameter evolution |
| `make breed` | Behavioral breeding |
| `make clean` | Remove caches, venvs, node_modules |

## License

This project is licensed under the [MIT License](LICENSE).


## Checkpoints

### Ema
The `hof_Ema_Mlakar_age316_d3f9ae44.pt` checkpoint is included in the repository.
- **Instruct + ~500k self-play games**: This model was initially trained using imitation learning on a dataset of expert games, then fine-tuned through approximately 500,000 games of self-play.
- **Winrates**: Capable of achieving over 90% winrate vs V1 and ~65% vs V3 of the StockŠkis heuristic bot.
