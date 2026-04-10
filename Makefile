.PHONY: run backend frontend test train clean install test-e2e setup setup-hooks \
	test-backend test-frontend test-frontend-unit test-coverage check-coverage test-lookahead \
	pipeline imitation-pretrain generate-expert-data build-engine kill stop

UV_RUN = cd backend && PYTHONPATH=src uv run --default-index https://pypi.org/simple

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
run: backend frontend

backend:
	cd backend && PYTHONPATH=src uv run --default-index https://pypi.org/simple uvicorn tarok.adapters.api.server:app --reload --host 0.0.0.0 --port 8000 &

frontend:
	cd frontend && npm run dev &

# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────
test: test-backend test-frontend test-frontend-unit

test-backend:
	$(UV_RUN) python -m pytest tests/ -v

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

test-coverage:
	$(COV_PYTEST) --cov=tarok --cov-report=term-missing --cov-report=json:coverage.json -v

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

evolve:
	$(UV_RUN) python -m tarok evolve --pop 12 --gens 10 --eval-sessions 20 --eval-games 10

train-evolved:
	$(UV_RUN) python -m tarok train-evolved

breed:
	$(UV_RUN) python -m tarok breed --warmup 50 --pop 12 --gens 5 --cycles 3 --eval-games 100 --refine 30

train-bred:
	$(UV_RUN) python -m tarok train-bred

# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# 3-Phase Training Pipeline
# ──────────────────────────────────────────────

# Build the Rust engine (required before training)
build-engine:
	cd backend && uv run --default-index https://pypi.org/simple maturin develop --release --manifest-path ../engine-rs/Cargo.toml

# Generate expert data from StockŠkis bots (standalone benchmark)
generate-expert-data:
	$(UV_RUN) python -m tarok generate-expert-data --games 1000000

# Imitation pre-train from StockŠkis expert games (policy + value)
imitation-pretrain:
	$(UV_RUN) python -m tarok imitation-pretrain --games 1000000

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
