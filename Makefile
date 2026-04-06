.PHONY: run backend frontend test train clean install test-e2e

# --- Install dependencies ---
install:
	cd backend && uv sync --default-index https://pypi.org/simple
	cd frontend && npm install

# --- Run full stack (backend + frontend) ---
run: backend frontend

backend:
	cd backend && uv run --default-index https://pypi.org/simple uvicorn tarok.adapters.api.server:app --reload --host 0.0.0.0 --port 8000 &

frontend:
	cd frontend && npm run dev &

# --- Test ---
test:
	cd backend && uv run --default-index https://pypi.org/simple python -m pytest tests/ -v

test-quick:
	cd backend && uv run --default-index https://pypi.org/simple python -m pytest tests/ -v -x --no-header

test-e2e:
	cd frontend && npx playwright test

# --- Train agents ---
train:
	cd backend && uv run --default-index https://pypi.org/simple python -m tarok train 100 100

evolve:
	cd backend && uv run --default-index https://pypi.org/simple python -m tarok evolve --pop 12 --gens 10 --eval-sessions 20 --eval-games 10

train-evolved:
	cd backend && uv run --default-index https://pypi.org/simple python -m tarok train-evolved

breed:
	cd backend && uv run --default-index https://pypi.org/simple python -m tarok breed --warmup 50 --pop 12 --gens 5 --cycles 3 --eval-games 100 --refine 30

train-bred:
	cd backend && uv run --default-index https://pypi.org/simple python -m tarok train-bred

# --- Clean ---
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf backend/.venv frontend/node_modules

# --- Stop background processes ---
stop:
	-pkill -f "uvicorn tarok" 2>/dev/null || true
	-pkill -f "vite" 2>/dev/null || true
