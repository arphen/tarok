#!/usr/bin/env bash
# Focused Rust test: PIMC vs alpha-mu on the same endgame positions.
#
# Bypasses m6 / lustrek heuristics entirely — tricks 1..=8 are played with
# uniform-random legal moves (identical between both branches), then tricks
# 9..=12 are played twice from the same snapshot, with the declarer seat
# using PIMC in one branch and alpha-mu in the other.  Opponents play
# random-legal in both (seeded identically at the snapshot).
#
# Tunable env vars:
#   ENDGAME_TRIALS       — number of trials (default 120)
#   ENDGAME_WORLDS       — PIMC / alpha-mu world samples (default 100)
#   ENDGAME_ALPHA_DEPTH  — alpha-mu iterative-deepening depth (default 2)
#   ENDGAME_SEED         — base RNG seed (default 0xC0FFEE)
#
# Example:
#   ENDGAME_TRIALS=400 scripts/bench_endgame_solvers.sh

set -euo pipefail
cd "$(dirname "$0")/.."

source backend/.venv/bin/activate
export LIBTORCH_USE_PYTORCH=1

TORCH_LIB=$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__),"lib"))')
PY_PREFIX=$(python -c 'import sys; print(sys.base_prefix)')
PY_LIB="$PY_PREFIX/lib"
PY_VER=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

# The cargo-test binary is linked with `-undefined dynamic_lookup` (see
# engine-rs/.cargo/config.toml), so we must inject libpython + libtorch
# at launch time.
export DYLD_INSERT_LIBRARIES="$PY_LIB/libpython${PY_VER}.dylib:$TORCH_LIB/libtorch_cpu.dylib"
export DYLD_LIBRARY_PATH="$TORCH_LIB:$PY_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

cd engine-rs
exec cargo test --release --test endgame_pimc_vs_alpha_mu -- --nocapture "$@"
