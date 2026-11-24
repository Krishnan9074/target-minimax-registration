#!/usr/bin/env bash
# Reproduce every result, table and figure from scratch.
# Single-threaded BLAS keeps the multiprocessing pools from oversubscribing and
# makes runs bit-reproducible across machines.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir -p results/logs

run() { echo "== $1"; python3 -u "$1" > "results/logs/$(basename "$1" .py).log" 2>&1; }

run tests/test_theory.py
for i in $(seq 1 16); do
  run experiments/exp${i}_*.py
done

python3 experiments/make_tables.py
python3 experiments/make_figures.py
echo "done: results/, figures/, outputs/latex/"
