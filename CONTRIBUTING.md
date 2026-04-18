# Contributing to Slovenian Tarok

First off, thank you for considering contributing to Slovenian Tarok!

## Quick Start (macOS)

One command sets up everything — Homebrew, Python, Node.js, uv, all dependencies, and the pre-commit hook:

```bash
make setup
```

Then start the app:

```bash
make run        # starts backend (port 8000) + frontend (port 3000)
make stop       # kills both
```

### What `make setup` does

| Step | What it installs | Skipped if already present? |
|------|------------------|-----------------------------|
| 1 | [Homebrew](https://brew.sh) | Yes |
| 2 | Python 3.12+ | Yes |
| 3 | Node.js | Yes |
| 4 | [uv](https://docs.astral.sh/uv/) (Python package manager) | Yes |
| 5 | [Rust](https://www.rust-lang.org/) (via rustup) | Yes |
| 6 | Backend Python deps (including dev/test) | — |
| 7 | Frontend npm deps | — |
| 8 | Rust engine (`tarok_engine` PyO3 extension) | — |
| 9 | Git pre-commit hook | — |

### If you already have the tools

```bash
make install       # just install Python + Node deps
make setup-hooks   # just install the pre-commit hook
```

## Running Tests

```bash
make test               # backend pytest + frontend type-check + frontend unit tests
make test-backend       # backend only
make test-frontend      # frontend type-check only
make test-frontend-unit # frontend Vitest component tests
make test-quick         # backend, stop on first failure
make test-e2e           # Playwright end-to-end tests
make test-coverage      # backend tests with full coverage report
```

## Coverage Policy

A **pre-commit hook** runs automatically on every `git commit`. It:

1. Runs frontend quality checks (`format:check` / `lint` if configured)
2. Type-checks frontend (`tsc --noEmit`) and runs frontend unit tests (`vitest`)
3. Runs backend formatting + lint checks (`ruff format --check`, `ruff check`)
4. Runs optional backend static typing (`ty`) when enabled (`ENABLE_TY=1`) and installed
5. Runs backend tests with coverage measurement
6. Compares coverage to the saved baseline (`.coverage-baseline`)
7. **Rejects the commit** if coverage decreased

If coverage improves, the baseline is auto-updated. If you intentionally remove dead code and coverage drops, update the baseline manually:

```bash
make update-coverage-baseline
```

## Makefile Cheat Sheet

| Command | Description |
|---------|-------------|
| `make setup` | Full bootstrap (fresh Mac) |
| `make install` | Install deps only |
| `make run` | Start backend + frontend |
| `make stop` | Stop background servers |
| `make test` | Run all tests |
| `make test-frontend-unit` | Frontend component tests (Vitest) |
| `make test-coverage` | Tests with coverage report |
| `make check-coverage` | Test + assert no regression |
| `make update-coverage-baseline` | Reset coverage baseline |
| `make clean` | Remove caches & venvs |

## How Can I Contribute?

### Reporting Bugs
* Use the GitHub issue search. Check if the issue has already been reported.
* Check if the issue has been fixed and try to reproduce it using the latest `main` branch.

### Suggesting Enhancements
* Make sure you are using the latest version.
* Read the documentation carefully and find out if the functionality is already covered.

### Pull Requests
* Fill in the required template.
* Do not include issue numbers in the PR title.
* Include screenshots and animated GIFs in your pull request whenever possible.
* End all files with a newline.
