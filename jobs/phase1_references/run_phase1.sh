#!/bin/bash
# Launcher for Phase 1: Naive Self-Play References
# Only a few seeds needed to estimate stable reference returns.
#   IPD        -> 100 seeds (0..99)
#   CoinGame   ->  20 seeds (0..19)
#   InTheMatrix->  20 seeds (0..19)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- IPD self-play (100 seeds) ---
for seed in $(seq 0 99); do
    sbatch --export=ALL "$SCRIPT_DIR/R1_ipd_selfplay_tabular.sh" $seed
done
echo "Submitted 100 jobs for R1_ipd_selfplay_tabular"

# --- CoinGame self-play (20 seeds) ---
for seed in $(seq 0 19); do
    sbatch --export=ALL "$SCRIPT_DIR/R2_cg_selfplay.sh" $seed
done
echo "Submitted 20 jobs for R2_cg_selfplay"

# --- InTheMatrix self-play (20 seeds) ---
for seed in $(seq 0 19); do
    sbatch --export=ALL "$SCRIPT_DIR/R3_ipditm_selfplay.sh" $seed
done
echo "Submitted 20 jobs for R3_ipditm_selfplay"

echo "Phase 1 submission complete."
