# Slovenian Tarok

[![CI](https://github.com/arphen/tarok/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/arphen/tarok/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/arphen/tarok/branch/main/graph/badge.svg)](https://codecov.io/gh/arphen/tarok)
![License: AGPL v3+](https://img.shields.io/badge/License-AGPL%20v3%2B-blue.svg)

A full-stack Slovenian Tarok platform with a modern web UI, a high-performance Rust game engine, and reinforcement-learning agents trained through self-play.

## What You Can Do

- Play human vs AI with configurable opponents
- Spectate AI vs AI games with full game state visibility
- Run tournament brackets between model checkpoints and bots
- Run large Bot Arena simulations and inspect aggregated results
- Train and benchmark models with the training-lab pipeline

## Project Highlights

- **Rust-first game engine** for game flow, legal moves, trick resolution, and scoring
- **FastAPI backend** with REST + WebSocket endpoints
- **React + TypeScript frontend** focused on gameplay and diagnostics
- **PPO-based training stack** with modular adapters in `training-lab/`
- **CI + coverage reporting** on push and pull request

## Repository Layout

```text
backend/        FastAPI app, AI adapters, tests
engine-rs/      Rust Tarok engine (PyO3 extension)
frontend/       React + TypeScript client
model/          Shared model/encoding package
training-lab/   Training orchestration and PPO pipeline
docs/           Design docs and roadmap notes
```

## Quick Start

### 1. One-time setup

```bash
make setup
```

### 2. Run backend + frontend

```bash
make run
```

Frontend: `http://localhost:3000`  
Backend: `http://localhost:8000`

## Development Commands

```bash
make test                # Backend + frontend checks
make test-backend        # Backend pytest suite
make test-lab            # Training-lab tests
make test-frontend       # TypeScript check
make test-frontend-unit  # Vitest suite
make test-e2e            # Playwright tests
make build-engine        # Build/install Rust extension
make ensure-engine       # Verify/build Rust extension if missing
```

## Training

Basic training entrypoints:

```bash
make train-new
make train-iterate MODEL=path/to/checkpoint.pt
```

For more advanced training flows and configs, see `training-lab/` and `docs/training_lab.md`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, workflow, and testing expectations.

## License

This project is licensed under the GNU Affero General Public License v3.0 or later. See [LICENSE](LICENSE).
