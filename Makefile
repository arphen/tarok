.PHONY: run backend frontend test train clean install test-e2e setup setup-hooks \
       test-backend test-frontend test-coverage check-coverage

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
	@echo "==> Installing backend deps…"
	cd backend && uv sync --default-index https://pypi.org/simple --extra dev
	@echo "==> Installing frontend deps…"
	cd frontend && npm install
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
test: test-backend test-frontend

test-backend:
	$(UV_RUN) python -m pytest tests/ -v

test-quick:
	$(UV_RUN) python -m pytest tests/ -v -x --no-header

test-frontend:
	cd frontend && npx tsc -b --noEmit

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
# Housekeeping
# ──────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf backend/.venv frontend/node_modules
	rm -f backend/coverage.json

stop:
	-pkill -f "uvicorn tarok" 2>/dev/null || true
	-pkill -f "vite" 2>/dev/null || true
