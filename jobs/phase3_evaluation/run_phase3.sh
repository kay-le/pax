#!/bin/bash
# Launcher for Phase 3: Evaluation
# IMPORTANT: Update model_path and run_path in each eval yaml config before running.
#   IPD        -> 100 seeds (0..99)
#   CoinGame   ->  20 seeds (0..19)
#   InTheMatrix->  20 seeds (0..19)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- IPD evaluation (100 seeds) ---
IPD_EVAL_SCRIPTS=(
    E1_ipd_eval_welfare_shaper_att.sh
    E2_ipd_eval_welfare_shaper_mfos.sh
)
for script in "${IPD_EVAL_SCRIPTS[@]}"; do
    for seed in $(seq 0 99); do
        sbatch --export=ALL "$SCRIPT_DIR/$script" $seed
    done
    echo "Submitted 100 jobs for $script"
done

# --- InTheMatrix evaluation (20 seeds) ---
for seed in $(seq 0 19); do
    sbatch --export=ALL "$SCRIPT_DIR/E3_ipditm_eval_welfare_shaper_att.sh" $seed
done
echo "Submitted 20 jobs for E3_ipditm_eval_welfare_shaper_att"

# --- CoinGame evaluation (20 seeds) ---
for seed in $(seq 0 19); do
    sbatch --export=ALL "$SCRIPT_DIR/E4_cg_eval_welfare_shaper_att.sh" $seed
done
echo "Submitted 20 jobs for E4_cg_eval_welfare_shaper_att"

echo "Phase 3 submission complete."
