# Slovenian Tarok

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![React](https://img.shields.io/badge/react-18-cyan.svg)
![FastAPI](https://img.shields.io/badge/fastapi-0.110+-green.svg)

A 4-player Slovenian Tarok card game with AI agents that learn by self-play using deep reinforcement learning (PPO).

Developed with clean architecture principles, separating the domain logic completely from frameworks like FastAPI and React.

## Architecture (Clean Architecture)

```
backend/src/tarok/
├── entities/          # Domain entities — Card, GameState, Scoring
│   ├── card.py        # Card, Suit, Deck (54 cards)
│   ├── game_state.py  # GameState, Phase, Contract, Trick
│   └── scoring.py     # Point counting, game scoring
├── ports/             # Interfaces (dependency inversion)
│   ├── player_port.py # PlayerPort — any player (human/AI/random)
│   ├── observer_port.py # GameObserverPort — event notifications
│   └── game_repo_port.py # GameRepoPort — persistence
├── use_cases/         # Application business rules
│   ├── game_loop.py   # Orchestrates full game flow
│   ├── deal.py        # Card dealing
│   ├── bid.py         # Bidding
│   ├── call_king.py   # King calling (2v2 partner selection)
│   ├── exchange_talon.py # Talon exchange (plugin system)
│   └── play_trick.py  # Trick play & resolution
└── adapters/          # Infrastructure
    ├── ai/            # RL Agent, neural network, PPO trainer
    │   ├── agent.py   # RLAgent implements PlayerPort
    │   ├── network.py # Actor-Critic neural network
    │   ├── trainer.py # PPO self-play training loop
    │   ├── encoding.py # State → tensor encoding
    │   └── random_agent.py # Random baseline
    └── api/           # FastAPI REST + WebSocket
        ├── server.py  # API endpoints
        ├── human_player.py # HumanPlayer implements PlayerPort
        ├── ws_observer.py  # WebSocket game event broadcaster
        └── schemas.py # Pydantic schemas
```

**Dependency rule**: entities ← ports ← use_cases ← adapters. Inner layers never import outer layers.

## Quick Start

### Backend

```bash
cd backend
uv sync
uv run pytest                    # Run tests
uv run python -m tarok train 500 # Train agents (500 games)
uv run python -m tarok           # Start API server
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

## Game Rules (Slovenian Tarok, 4-player)

- **54 cards**: 22 Taroks (I–XXI + Škis) + 32 suit cards (4 suits × 8)
- **4 players**, each dealt 12 cards, 6 go to the talon
- **Bidding**: Players bid contracts (Three, Two, One, Solo)
- **King Calling**: Declarer calls a king — the holder is their secret 2v2 partner
- **Talon Exchange**: Declarer picks talon cards and discards equal number
- **12 Tricks**: Must follow suit, must play tarok if can't follow
- **Scoring**: Cards counted in groups of 3. Declarer team needs ≥36 of 70 points

## AI Agent

- **PPO (Proximal Policy Optimization)** with actor-critic architecture
- State encoding: 179 features (hand, played cards, trick, contract, position)
- Action space: 54 cards (masked to legal moves)
- Self-play: 4 agents sharing one network play against each other
- Training dashboard shows win rate, reward, loss curves in real-time

## License

This project is licensed under the [MIT License](LICENSE).

