# Slovenian Tarok

[![CI](https://github.com/arphen/tarok/actions/workflows/ci.yml/badge.svg?branch=gpu)](https://github.com/arphen/tarok/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/arphen/tarok/branch/gpu/graph/badge.svg)](https://codecov.io/gh/arphen/tarok)
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
- **Arena checkpoint leaderboard** — Dedicated page that ranks RL checkpoints
  using persisted Bot Arena results (survives refresh and browser restart)
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

## Arena Persistence and Leaderboard

Bot Arena saves completed runs to backend storage at `backend/data/arena_results.json`.
Saved runs are available after a frontend refresh or browser restart.

- `GET /api/arena/history` — returns persisted arena runs
- `GET /api/arena/history?checkpoint=<filename>` — filter runs by checkpoint
- `GET /api/arena/leaderboard/checkpoints` — aggregated, checkpoint-focused leaderboard

From the home screen, open **Arena Leaderboard** to view checkpoint rankings in a
separate page (not inside Bot Arena tabs).

## Training

The training system lives in `training-lab/` and uses a high-performance Rust
engine for fast game simulation. Models are trained with PPO (Proximal Policy
Optimization) using self-play and/or rule-based bot opponents.

### Prerequisites

```bash
make setup     # one-time: installs Python, Node, Rust, all deps, builds engine
```

If you already ran `make setup`, make sure the Rust engine is built:

```bash
make ensure-engine
```

### Train a New Model from Scratch

```bash
make train-new
```

This creates a brand-new randomly-initialized model with a random Slovenian
name (e.g. `Klara_Oblak`, `Maja_Horvat`) and trains it for 10 iterations of
10,000 games each. Checkpoints are saved to `checkpoints/<Name>/`.

```bash
# Use a different training config
make train-new CONFIG=self-play

# More iterations, custom hyperparameters
make train-new CONFIG=vs-3-bots EXTRA="--iterations 50 --games 20000 --lr 0.0001"
```

### Continue Training an Existing Model

```bash
make train-iterate MODEL=path/to/model.pt
```

The default config is `vs-3-bots` (1 neural network vs 3 rule-based StockŠkis
v5 bots). Override with `CONFIG=`:

```bash
# Continue training Katja (the current strongest model)
make train-iterate MODEL=backend/checkpoints/hall_of_fame/pinned/hof_Katja_Vidmar_age59_014692c0.pt

# Train against stronger v6 bots
make train-iterate CONFIG=vs-3-v6 MODEL=path/to/model.pt

# Full self-play (4 neural networks, no bots)
make train-iterate CONFIG=self-play MODEL=path/to/model.pt

# 50 iterations of 20k games with a lower learning rate
make train-iterate MODEL=path/to/model.pt EXTRA="--iterations 50 --games 20000 --lr 0.0001"
```

### Training Configs

Configs live in `training-lab/configs/`. Each YAML file sets the seat layout,
hyperparameters, and benchmark settings:

| Config | Seats | Description |
|--------|-------|-------------|
| `vs-3-bots` | NN vs 3× bot_v5 | **Default.** Easiest. Good starting point for new models. |
| `vs-2-bots` | 2× NN vs 2× bot_v5 | NN gets a partner that plays like itself. |
| `vs-1-bot` | 3× NN vs 1× bot_v5 | Late-stage training when NN already beats bots. |
| `vs-3-v6` | NN vs 3× bot_v6 | Harder opponents. Use when v5 is too easy. |
| `self-play` | 4× NN | Pure self-play. Use once NN has surpassed all bots. |
| `ec2-g5-1h` | 4× NN + league bots | 1-hour CUDA-oriented self-play run for EC2 g5.2xlarge. |

You can override any YAML setting via the `EXTRA` variable:

```bash
make train-iterate EXTRA="--seats nn,bot_v6,bot_v5,bot_v5 --explore-rate 0.15"
```

### Remote Training on EC2 Spot

The repo includes a deployment script plus Makefile wrappers for launching a
`g5.2xlarge` spot instance, syncing the repo, bootstrapping the backend, and
starting training inside a remote `tmux` session.

You can run this in any AWS region where all of the following are true:

1. `g5` instances are offered in that region and in the subnet/AZ you choose.
2. Your account has enough vCPU / GPU quota for `g5.2xlarge` spot requests there.
3. You pick a region-valid Ubuntu GPU AMI.
4. Your key pair, security group, subnet, and instance profile exist in that same region.

For this workload, `g5.2xlarge` is a good default. It gives you 1× A10G with 24 GB VRAM,
which is enough for the `ec2-g5-1h` config and materially faster than M3/MPS training.
It is not the only valid choice, but it is a sensible first target because the script and
config are tuned around it.

One-time setup on your Mac:

```bash
brew install awscli
aws configure
aws s3 mb s3://my-tarok-checkpoints
```

If you already use AWS CLI profiles, prefer putting `AWS_PROFILE` and
`AWS_DEFAULT_REGION` in a local `.env` file instead of storing raw access keys.
The EC2 instance itself should use an IAM instance profile, so your long-lived AWS
secrets do not need to be copied to the machine.

Example `.env`:

```bash
AWS_PROFILE=default
AWS_DEFAULT_REGION=eu-central-1

EC2_KEY=my-keypair
EC2_SG=sg-0123456789abcdef0
EC2_BUCKET=s3://my-tarok-checkpoints
EC2_SUBNET=subnet-0123456789abcdef0
EC2_INSTANCE_PROFILE=ec2-s3-tarok
EC2_INSTANCE_TYPE=g5.2xlarge
EC2_SPOT_PRICE=1.20
TAROK_EC2_AMI=auto

MODEL=checkpoints/Petra_Novak/iter_090.pt
CONFIG=ec2-g5-1h
```

`.env` is ignored by git, and the repo includes `.env.example` as a template.

You also need:

1. An EC2 keypair whose PEM file is in `~/.ssh/<key>.pem`
2. A security group that allows SSH from your IP
3. An instance profile named `ec2-s3-tarok` with S3 read/write access to your bucket

Launch a 1-hour self-play run:

```bash
make ec2-train \
  EC2_KEY=my-keypair \
  EC2_SG=sg-0123456789abcdef0 \
  EC2_BUCKET=s3://my-tarok-checkpoints \
  MODEL=checkpoints/Petra_Novak/iter_090.pt \
  CONFIG=ec2-g5-1h
```

Or, if you filled out `.env`, just run:

```bash
make ec2-train
```

The script prints the instance ID and public IP when training starts. Use those
with the monitoring targets below.

### Where To Get Each Value In AWS UI

1. `AWS_DEFAULT_REGION`: top-right region selector in the AWS console.
2. `EC2_KEY`: EC2 → Network & Security → Key Pairs. Create or import one; use the key pair name.
3. `EC2_SG`: EC2 → Network & Security → Security Groups. Create one with inbound TCP 22 from your public IP; use the security group ID.
4. `EC2_BUCKET`: S3 → Buckets. Create a bucket; use `s3://bucket-name`.
5. `EC2_SUBNET`: VPC → Subnets. Choose a public subnet in an AZ where `g5` has capacity, or leave it unset to use the default VPC path.
6. `EC2_INSTANCE_PROFILE`: IAM → Roles. Create a role for EC2 with S3 read/write permissions, then use that role / instance profile name.
7. `TAROK_EC2_AMI`: EC2 → AMIs. You can leave this as `auto`, or explicitly pick the latest Deep Learning GPU Ubuntu AMI for your region.
8. `MODEL`: local checkpoint path on your machine.

Two practical rules:

1. Keep the S3 bucket in the same region as the instance unless you have a reason not to.
2. Cheapest region only helps if `g5` spot capacity is actually available there when you launch.

### EC2 Monitoring Targets

```bash
# Attach to the remote tmux session for full live output
make ec2-attach EC2_KEY=my-keypair EC2_IP=1.2.3.4

# Follow the tee'd training log without attaching to tmux
make ec2-logs EC2_KEY=my-keypair EC2_IP=1.2.3.4

# Pull checkpoints synced to S3 back to your machine
make ec2-pull EC2_BUCKET=s3://my-tarok-checkpoints

# Terminate the instance when the run is finished
make ec2-terminate INSTANCE_ID=i-0123456789abcdef0
```

Operational notes:

1. Training runs inside `tmux`, so detaching with `Ctrl-b d` does not stop the job.
2. The deploy script writes `~/tarok/train.log` on the instance via `tee`.
3. A cron job syncs `~/tarok/checkpoints/` to `s3://.../checkpoints/` every 5 minutes.
4. `training-lab/configs/ec2-g5-1h.yaml` is tuned for the A10G GPU on `g5.2xlarge`.

### Training Output

Each iteration prints a progress bar with:
1. **Self-play** — N games, collects training experiences
2. **PPO update** — gradient steps on the collected data
3. **Benchmark** — N games greedy (no exploration), measures avg session placement

```
─── Iteration 3/10  ████████░░░░░░░░  20%  ETA 4m12s ───
  [1/3] Self-play: 10000 games (nn,bot_v5,bot_v5,bot_v5, explore=0.1)... 38291 exps in 28s
  [2/3] PPO update (6 epochs, batch=8192)... loss=0.6234 (p=0.0312 v=0.5922 ent=0.0421) in 3s
  [3/3] Benchmark: 10000 games (greedy)... placement=1.850 in 24s
  → placement 2.100 → 1.850  (-0.250 ▲ better!)
```

At the end, a summary shows placement trend across iterations and saves the
best model:

```
  Best model saved to checkpoints/training_run/best.pt
```

### Recommended Training Path

1. **Start with bots**: `make train-new` — this trains against v5 bots until
   the model is competitive
2. **Harder bots**: `make train-iterate CONFIG=vs-3-v6 MODEL=checkpoints/<Name>/best.pt`
3. **Self-play**: `make train-iterate CONFIG=self-play MODEL=checkpoints/<Name>/best.pt`
   — once the model beats all bots, pure self-play refines strategy further

### Advanced: In-Browser Training Dashboard

The web UI includes a real-time training dashboard with live charts for win
rate, loss, bid rate, per-contract stats, and more:

```bash
make run    # start backend + frontend
```

Open http://localhost:3000 and click **Train AI**.

### Advanced: Legacy Training Commands

These are older training entrypoints that go through the backend directly:

```bash
make train             # 100 sessions × 100 games (basic PPO)
make evolve            # DEAP hyperparameter search
make train-evolved     # Continue with best evolved hparams
make breed             # Behavioral specialisation breeding
make train-bred        # Continue with best bred model
make pipeline          # Full 3-phase: imitation → PPO vs bots → self-play
```

### Hall of Fame

The `backend/checkpoints/hall_of_fame/pinned/` directory contains the strongest
models trained so far. These are tracked in git and should not be deleted.

| Model | Description |
|-------|-------------|
| `hof_Ema_Mlakar_age316` | Imitation + ~500k self-play games. >90% WR vs v1, ~65% vs v3. |
| `hof_Katja_Vidmar_age59` | Current strongest model. |

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
| **Testing** | |
| `make test` | Run all tests |
| `make test-backend` | Backend pytest |
| `make test-frontend` | Frontend typecheck |
| `make test-frontend-unit` | Frontend Vitest unit tests |
| `make test-e2e` | Playwright E2E |
| `make test-coverage` | Backend coverage report |
| **Training** | |
| `make train-new` | Train a brand-new model from scratch (random Slovenian name) |
| `make train-iterate` | Continue training an existing model (`MODEL=path/to/model.pt`) |
| `make pipeline` | Full 3-phase pipeline: imitation → PPO vs bots → self-play |
| `make train` | Legacy: basic 100×100 PPO training |
| `make evolve` | Legacy: DEAP hyperparameter search |
| `make breed` | Legacy: behavioral breeding |
| **Build** | |
| `make build-engine` | Compile Rust engine (PyO3) into the Python venv |
| `make build` | Production frontend build |
| `make clean` | Remove caches, venvs, node_modules |

## Rust Engine (optional but recommended)

The `engine-rs/` directory contains a PyO3-based Rust extension (`tarok_engine`)
that accelerates game simulation 10–100× over the pure-Python implementation.
It's used for training, batch game running, and the imitation learning pipeline.

```bash
# One-time: install Rust (if not already present)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build and install into the backend venv
make build-engine
```

`make setup` handles both steps automatically. If the extension is not installed,
the backend falls back to the pure-Python game loop — training will be slower but
still functional.

To verify the extension is installed:

```bash
cd backend && PYTHONPATH=src uv run python -c "import tarok_engine; print('OK')"
```

## License

This project is licensed under the [MIT License](LICENSE).
