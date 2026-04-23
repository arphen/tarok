ifneq (,$(wildcard .env))
include .env
export
endif


.PHONY: run backend frontend test train clean install test-e2e setup setup-hooks \
	test-backend test-lab test-frontend test-frontend-unit test-coverage test-coverage-backend test-coverage-lab check-coverage test-lookahead \
	lint-architecture pipeline imitation-pretrain generate-expert-data build-engine ensure-engine kill stop \
	train-with-humans cartpole cartpole-ppo ec2-train ec2-attach ec2-logs ec2-pull ec2-terminate

UV_RUN = cd backend && PYTHONPATH=src:../model/src uv run --default-index https://pypi.org/simple
UV_RUN_LAB = cd backend && PYTHONPATH=src:../model/src:../training-lab uv run --default-index https://pypi.org/simple
EC2_KEY ?=
EC2_SG ?=
EC2_BUCKET ?=
EC2_IP ?=
INSTANCE_ID ?=
EC2_SUBNET ?=
EC2_INSTANCE_TYPE ?=
EC2_SPOT_PRICE ?=
EC2_INSTANCE_PROFILE ?=

# ──────────────────────────────────────────────
# Bootstrap — run this once on a fresh Mac
# ──────────────────────────────────────────────
setup:
	@echo "==> Checking Homebrew…"
	@command -v brew >/dev/null 2>&1 || \
		/bin/bash -c "$$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
	@echo "==> Checking Python 3.12+…"
	@command -v python3 >/dev/null 2>&1 || brew install python@3.12
	@echo "==> Checking Node.js…"
	@command -v node >/dev/null 2>&1 || brew install node
	@echo "==> Checking uv…"
	@command -v uv >/dev/null 2>&1 || brew install uv
	@echo "==> Checking Rust…"
	@command -v rustc >/dev/null 2>&1 || \
		(curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
		. "$$HOME/.cargo/env")
	@echo "==> Installing backend deps…"
	cd backend && uv sync --default-index https://pypi.org/simple --extra dev
	@echo "==> Writing .env defaults for Rust engine…"
	@TORCH_LIB_DIR=$$(cd backend && uv run --default-index https://pypi.org/simple python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")'); \
	if [ ! -f .env ]; then \
		echo "LIBTORCH_USE_PYTORCH=1" > .env; \
		echo "DYLD_FALLBACK_LIBRARY_PATH=$$TORCH_LIB_DIR" >> .env; \
		echo "Created .env with Rust engine defaults."; \
	else \
		if ! grep -q '^LIBTORCH_USE_PYTORCH=' .env; then \
			echo "LIBTORCH_USE_PYTORCH=1" >> .env; \
			echo "Added LIBTORCH_USE_PYTORCH to .env."; \
		fi; \
		if ! grep -q '^DYLD_FALLBACK_LIBRARY_PATH=' .env; then \
			echo "DYLD_FALLBACK_LIBRARY_PATH=$$TORCH_LIB_DIR" >> .env; \
			echo "Added DYLD_FALLBACK_LIBRARY_PATH to .env."; \
		fi; \
	fi
	@echo "==> Installing frontend deps…"
	cd frontend && npm install
	@echo "==> Building Rust engine…"
	$(MAKE) build-engine
	@echo "==> Installing pre-commit hook…"
	$(MAKE) setup-hooks
	@echo ""
	@echo "✅  All done. Run 'make run' to start the app."

# ──────────────────────────────────────────────
# Install deps (skip system tools)
# ──────────────────────────────────────────────
install:
	cd backend && uv sync --default-index https://pypi.org/simple --extra dev
	cd frontend && npm install
	$(MAKE) build-engine

# ──────────────────────────────────────────────
# Git hooks
# ──────────────────────────────────────────────
setup-hooks:
	@mkdir -p .git/hooks
	@cp scripts/pre-commit .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "Pre-commit hook installed."

# ──────────────────────────────────────────────
# Run full stack (backend + frontend)
# ──────────────────────────────────────────────
run: ensure-engine backend frontend

backend:
	@TORCH_LIB_DIR=$$(cd backend && uv run --default-index https://pypi.org/simple python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")'); \
	cd backend && DYLD_FALLBACK_LIBRARY_PATH="$$TORCH_LIB_DIR:$$DYLD_FALLBACK_LIBRARY_PATH" PYTHONPATH=src:../model/src uv run --default-index https://pypi.org/simple uvicorn tarok.adapters.api.server:app --reload --host 0.0.0.0 --port 8000 &

frontend:
	cd frontend && npm run dev &

# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────
test: test-backend test-lab test-frontend test-frontend-unit

test-backend:
	$(MAKE) lint-architecture
	$(UV_RUN) python -m pytest tests/ -v

lint-architecture:
	@echo "==> Running Clean Architecture import contracts..."
	@cd backend && PYTHONPATH=src:../model/src uv run --default-index https://pypi.org/simple --extra dev lint-imports || { \
		echo ""; \
		echo "Clean Architecture guard failed."; \
		echo "Rule: tarok.use_cases is application logic and must not import infrastructure/data libraries (json/csv/pickle/numpy/pandas) or tarok.adapters."; \
		echo "Teaching mode: build bounded contexts, not adapter wastelands."; \
		echo "Design guide:"; \
		echo "  - define/extend a Port in tarok.ports first"; \
		echo "  - implement adapters in focused domain modules (not catch-all files)"; \
		echo "  - keep orchestration and rules in tarok.use_cases"; \
		echo "Naming guide:"; \
		echo "  - prefer explicit names like <tech>_<purpose>_adapter.py"; \
		echo "  - avoid vague files like misc.py, helpers.py, utils.py in adapters"; \
		echo "Definition of done:"; \
		echo "  1) Port contract added/updated"; \
		echo "  2) Adapter implemented in a cohesive domain module"; \
		echo "  3) Use case depends only on Port"; \
		exit 1; \
	}
	@cd training-lab && PYTHONPATH=../backend/src:../model/src:. ../backend/.venv/bin/lint-imports || { \
		echo ""; \
		echo "Clean Architecture guard failed (training-lab)."; \
		echo "Rule: training.use_cases must not import infrastructure/data libraries or training.adapters."; \
		echo "Teaching mode: leave architecture cleaner than you found it."; \
		echo "Current adapter structure is domain-driven:"; \
		echo "  - self_play, evaluation, modeling, configuration, presentation, persistence, policies"; \
		echo "Placement guide:"; \
		echo "  - add behavior to the right domain module; do not create flat adapter dumps"; \
		echo "  - keep file names explicit (e.g. rust_self_play_adapter.py, json_league_state_persistence.py)"; \
		echo "  - add/extend a Port in training.ports before wiring adapter code"; \
		echo "Definition of done:"; \
		echo "  1) Port contract added/updated"; \
		echo "  2) Adapter implemented in the correct domain package"; \
		echo "  3) Use case imports only ports/entities"; \
		exit 1; \
	}


	$(UV_RUN_LAB) python -m pytest ../training-lab/tests/ -v

test-quick:
	$(UV_RUN) python -m pytest tests/ -v -x --no-header

test-lookahead:
	$(UV_RUN) python -m pytest tests/test_lookahead.py -v

test-frontend:
	cd frontend && npx tsc -b --noEmit

test-frontend-unit:
	cd frontend && npx vitest run

test-e2e:
	cd frontend && npx playwright test

# ──────────────────────────────────────────────
# Coverage
# ──────────────────────────────────────────────
COVERAGE_BASELINE_FILE = .coverage-baseline

# Deterministic flags — pin randomness so coverage doesn't fluctuate between runs
COV_PYTEST = PYTHONHASHSEED=0 $(UV_RUN) python -m pytest tests/ -p no:randomly --hypothesis-seed=0

test-coverage: test-coverage-backend test-coverage-lab

test-coverage-backend:
	$(COV_PYTEST) --cov=tarok --cov-report=term-missing --cov-report=json:coverage.json -v

test-coverage-lab: ensure-engine
	PYTHONHASHSEED=0 $(UV_RUN_LAB) python -m pytest ../training-lab/tests/ -p no:randomly --hypothesis-seed=0 --cov=training --cov-report=term-missing --cov-report=json:../training-lab/coverage.json -v

check-coverage:
	@echo "==> Running tests with coverage…"
	@$(COV_PYTEST) -q --cov=tarok --cov-report=json:coverage.json
	@NEW_PCT=$$(python3 -c "import json; print(round(json.load(open('backend/coverage.json'))['totals']['percent_covered'], 2))"); \
	echo "Current coverage: $${NEW_PCT}%"; \
	if [ -f $(COVERAGE_BASELINE_FILE) ]; then \
		OLD_PCT=$$(cat $(COVERAGE_BASELINE_FILE)); \
		echo "Baseline coverage: $${OLD_PCT}%"; \
		python3 -c "import sys; sys.exit(0 if float('$${NEW_PCT}') >= float('$${OLD_PCT}') - 0.5 else 1)" || \
			{ echo "❌  Coverage dropped from $${OLD_PCT}% to $${NEW_PCT}% (>0.5% regression). Commit rejected."; exit 1; }; \
		echo "✅  Coverage did not decrease."; \
	else \
		echo "No baseline found — saving $${NEW_PCT}% as baseline."; \
		echo "$${NEW_PCT}" > $(COVERAGE_BASELINE_FILE); \
	fi

update-coverage-baseline:
	@$(COV_PYTEST) -q --cov=tarok --cov-report=json:coverage.json
	@python3 -c "import json; print(round(json.load(open('backend/coverage.json'))['totals']['percent_covered'], 2))" > $(COVERAGE_BASELINE_FILE)
	@echo "Baseline updated to $$(cat $(COVERAGE_BASELINE_FILE))%"

# ──────────────────────────────────────────────
# Train agents
# ──────────────────────────────────────────────
train:
	$(UV_RUN) python -m tarok train 100 100

# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# 3-Phase Training Pipeline
# ──────────────────────────────────────────────

# Build the Rust engine (required before training).
# maturin must be in the backend venv — install dev deps first (see pyproject.toml).
build-engine:
	cd backend && uv sync --default-index https://pypi.org/simple --extra dev --quiet && \
		LIBTORCH_USE_PYTORCH=1 RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup" \
		uv run --default-index https://pypi.org/simple maturin develop --release --manifest-path ../engine-rs/Cargo.toml

# Auto-install Rust + build engine if the .so is missing
ensure-engine:
	@TORCH_LIB_DIR=$$(cd backend && uv run --default-index https://pypi.org/simple python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")'); \
	export DYLD_FALLBACK_LIBRARY_PATH="$$TORCH_LIB_DIR:$$DYLD_FALLBACK_LIBRARY_PATH"; \
	if (cd backend && uv run --default-index https://pypi.org/simple python -c 'import tarok_engine as te; assert hasattr(te, "compute_gae"); assert hasattr(te, "CONTRACT_OFFSET")') 2>/dev/null; then \
		echo "✅  Rust engine already installed."; \
	else \
		echo "==> Rust engine missing/outdated, building…"; \
		if ! command -v rustc >/dev/null 2>&1; then \
			echo "==> Installing Rust toolchain…"; \
			curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
			. "$$HOME/.cargo/env"; \
		fi; \
		$(MAKE) build-engine; \
	fi

# Generate expert data from StockŠkis bots (standalone benchmark)
generate-expert-data:
	$(UV_RUN) python -m tarok generate-expert-data --games 1000000

# Imitation pre-train from StockŠkis expert games (policy + value)
imitation-pretrain:
	$(UV_RUN) python -m tarok imitation-pretrain --games 1000000

# Generate DD-solved training data (slow: ~1-5ms per position)
generate-dd-data:
	$(UV_RUN) python -m tarok generate-dd-data --games 1000

# DD pre-training: supervised learning from perfect-play labels
dd-pretrain:
	$(UV_RUN) python -m tarok dd-pretrain --games 10000 --epochs 20 --oracle

# Full 3-phase pipeline: imitation → StockŠkis PPO → self-play
# Phase 1: Supervised pre-training from 1M StockŠkis expert games
# Phase 2: PPO fine-tuning vs StockŠkis bots (auto plateau detection)
# Phase 3: Fictitious self-play for GTO convergence
pipeline:
	$(UV_RUN) python -m tarok pipeline

# Pipeline with custom parameters (example with larger budget)
pipeline-large:
	$(UV_RUN) python -m tarok pipeline \
		--p1-games 5000000 \
		--p2-sessions 500 --p2-games 100 \
		--p3-sessions 1000 --p3-games 100

# Train iteratively with progress bar + benchmark placement tracking
#
# Usage:
#   make train-iterate                                          # default: vs-3-bots config
#   make train-iterate CONFIG=self-play                         # full self-play, no bots
#   make train-iterate CONFIG=vs-3-v6                           # vs stronger v6 bots
#   make train-iterate CONFIG=vs-3-bots MODEL=path/to/model.pt  # custom checkpoint
#
# Available configs (training-lab/configs/):
#   vs-3-bots   — NN vs 3 rule-based bots (easiest, default)
#   vs-2-bots   — 2 NNs vs 2 bots
#   vs-1-bot    — 3 NNs vs 1 bot
#   self-play   — 4 NNs, no bots
#   vs-3-v6     — NN vs 3 stronger v6 bots
##   with-human-data — NN vs bots plus all saved human-vs-AI decisions every iteration
#
# All settings live in the YAML file. CLI overrides still work:
#   make train-iterate CONFIG=vs-3-bots EXTRA="--iterations 20 --games 5000"
MODEL  ?= backend/checkpoints/tarok_agent_latest.pt
CONFIG ?= vs-3-bots
EXTRA  ?=
train-iterate: ensure-engine
	source backend/.venv/bin/activate && \
		TORCH_LIB_DIR=$$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")') && \
		export DYLD_FALLBACK_LIBRARY_PATH="$$TORCH_LIB_DIR:$$DYLD_FALLBACK_LIBRARY_PATH" && \
		PYTHONPATH=backend/src:model/src python training-lab/train_and_evaluate.py \
		--config training-lab/configs/$(CONFIG).yaml \
		--checkpoint $(MODEL) \
		$(EXTRA)

# Fine-tune a model with all accumulated human-vs-AI game experiences.
# Human games from backend/data/human_experiences/ are replayed every iteration —
# they are never discarded, so the model keeps learning from every human game ever played.
#
# Usage:
#   make train-with-humans MODEL=checkpoints/YourModel/best.pt
#   make train-with-humans MODEL=checkpoints/YourModel/best.pt EXTRA="--iterations 20"
train-with-humans: ensure-engine
	source backend/.venv/bin/activate && \
		TORCH_LIB_DIR=$$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")') && \
		export DYLD_FALLBACK_LIBRARY_PATH="$$TORCH_LIB_DIR:$$DYLD_FALLBACK_LIBRARY_PATH" && \
		PYTHONPATH=backend/src:model/src python training-lab/train_and_evaluate.py \
		--config training-lab/configs/with-human-data.yaml \
		--checkpoint $(MODEL) \
		$(EXTRA)

# Train a brand-new randomly-named model from scratch
#   make train-new
#   make train-new CONFIG=self-play EXTRA="--iterations 20"
train-new: ensure-engine
	source backend/.venv/bin/activate && \
		TORCH_LIB_DIR=$$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")') && \
		export DYLD_FALLBACK_LIBRARY_PATH="$$TORCH_LIB_DIR:$$DYLD_FALLBACK_LIBRARY_PATH" && \
		PYTHONPATH=backend/src:model/src python training-lab/train_and_evaluate.py \
		--config training-lab/configs/$(CONFIG).yaml \
		--new \
		$(EXTRA)

# CartPole sanity check for validating custom optimization loops via Gymnasium.
# Keeps Tarok core untouched by running as an adapter-level script.
# Usage:
#   make cartpole
#   make cartpole EXTRA="--episodes 800 --strict"
cartpole:
	source backend/.venv/bin/activate && \
		PYTHONPATH=backend/src:model/src:training-lab python \
		training-lab/training/adapters/evaluation/gymnasium_cartpole_adapter.py \
		$(EXTRA)

# Full PPO math sanity check on CartPole (GAE + clip + multi-epoch updates).
# This mirrors key warm-up RL math knobs while excluding Tarok-specific IL/concurrency.
# Usage:
#   make cartpole-ppo
#   make cartpole-ppo EXTRA="--episodes 1000 --strict"
cartpole-ppo:
	$(MAKE) cartpole EXTRA="--ppo-epochs 4 --clip-epsilon 0.2 --value-clip-epsilon 0.2 --gae-lambda 0.98 --value-coef 0.5 --entropy-coef 0.001 --lr 2e-4 --episodes 800 --window 50 --target-avg-reward 200 $(EXTRA)"

# Launch a g5.2xlarge spot instance, deploy the repo, and start tmux-based training.
# Required vars:
#   EC2_KEY=my-keypair EC2_SG=sg-... EC2_BUCKET=s3://... MODEL=checkpoints/...pt
# Optional vars:
#   CONFIG=ec2-g5-1h AWS_DEFAULT_REGION=us-east-1 EC2_SUBNET=subnet-... \
#   EC2_INSTANCE_TYPE=g5.2xlarge EC2_SPOT_PRICE=1.20 EC2_INSTANCE_PROFILE=ec2-s3-tarok
ec2-train:
	@test -n "$(EC2_KEY)" || (echo "EC2_KEY is required" && exit 1)
	@test -n "$(EC2_SG)" || (echo "EC2_SG is required" && exit 1)
	@test -n "$(EC2_BUCKET)" || (echo "EC2_BUCKET is required" && exit 1)
	@test -n "$(MODEL)" || (echo "MODEL is required" && exit 1)
	./scripts/ec2-train.sh \
		--key "$(EC2_KEY)" \
		--sg "$(EC2_SG)" \
		--bucket "$(EC2_BUCKET)" \
		--model "$(MODEL)" \
		--config "$(CONFIG)" \
		$(if $(EC2_SUBNET),--subnet "$(EC2_SUBNET)") \
		$(if $(EC2_INSTANCE_TYPE),--instance-type "$(EC2_INSTANCE_TYPE)") \
		$(if $(EC2_SPOT_PRICE),--spot-price "$(EC2_SPOT_PRICE)") \
		$(if $(EC2_INSTANCE_PROFILE),--instance-profile "$(EC2_INSTANCE_PROFILE)")

# Attach to the remote tmux session for live training output.
ec2-attach:
	@test -n "$(EC2_KEY)" || (echo "EC2_KEY is required" && exit 1)
	@test -n "$(EC2_IP)" || (echo "EC2_IP is required" && exit 1)
	ssh -i ~/.ssh/$(EC2_KEY).pem ubuntu@$(EC2_IP) -t "tmux attach -t train"

# Follow the remote tee'd training log without attaching to tmux.
ec2-logs:
	@test -n "$(EC2_KEY)" || (echo "EC2_KEY is required" && exit 1)
	@test -n "$(EC2_IP)" || (echo "EC2_IP is required" && exit 1)
	ssh -i ~/.ssh/$(EC2_KEY).pem ubuntu@$(EC2_IP) "tail -f ~/tarok/train.log"

# Pull remote checkpoints from S3 back to the local checkpoints folder.
ec2-pull:
	@test -n "$(EC2_BUCKET)" || (echo "EC2_BUCKET is required" && exit 1)
	aws s3 sync "$(EC2_BUCKET)/checkpoints/" checkpoints/ec2-run/

# Terminate the EC2 instance created by ec2-train.
ec2-terminate:
	@test -n "$(INSTANCE_ID)" || (echo "INSTANCE_ID is required" && exit 1)
	./scripts/ec2-train.sh --terminate "$(INSTANCE_ID)"

# ──────────────────────────────────────────────
# Build
# ──────────────────────────────────────────────
build:
	cd frontend && npm run build

# ──────────────────────────────────────────────
# Housekeeping
# ──────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf backend/.venv frontend/node_modules frontend/dist
	rm -f backend/coverage.json

kill:
	@echo "==> Killing dev servers on ports 8000 and 3000..."
	@PIDS="$$(lsof -ti tcp:8000 -ti tcp:3000 2>/dev/null | sort -u)"; \
	if [ -n "$$PIDS" ]; then \
		echo "Killing PIDs: $$PIDS"; \
		kill $$PIDS 2>/dev/null || true; \
		sleep 1; \
		STILL_RUNNING="$$(lsof -ti tcp:8000 -ti tcp:3000 2>/dev/null | sort -u)"; \
		if [ -n "$$STILL_RUNNING" ]; then \
			echo "Force killing PIDs: $$STILL_RUNNING"; \
			kill -9 $$STILL_RUNNING 2>/dev/null || true; \
		fi; \
	else \
		echo "No dev servers found on ports 8000 or 3000."; \
	fi
	@pkill -f "uvicorn tarok.adapters.api.server:app" 2>/dev/null || true
	@pkill -f "npm run dev" 2>/dev/null || true
	@pkill -f "vite" 2>/dev/null || true

stop: kill
