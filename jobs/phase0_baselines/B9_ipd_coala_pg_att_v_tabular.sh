#!/bin/bash
# Unified launcher for B9 IPD COALA-PG vs tabular
#
# COALA-PG baseline: Meulemans et al. (2025), "Multi-agent cooperation through
# learning-aware policy gradients", ICLR 2025 (arXiv:2410.18636).
#
# Usage:
#   bash B9_ipd_coala_pg_att_v_tabular.sh <platform> <seed> [wandb_mode]
#
# Platforms:
#   fir        — Fir cluster, 1×H100 MIG slice
#   tri        — Trillium cluster, 1×H100
#   tri-debug  — Trillium, 1×H100, 1h, tiny params for a smoke test
#
# wandb_mode: offline (default) | online
#   KEEP OFFLINE on Alliance compute nodes — they generally have no outbound
#   internet, so `online` will stall until WANDB_INIT_TIMEOUT and then degrade.
#   Run offline, then `wandb sync` from a login node.
#
# Examples:
#   bash B9_ipd_coala_pg_att_v_tabular.sh fir 0
#   bash B9_ipd_coala_pg_att_v_tabular.sh tri 0
#   bash B9_ipd_coala_pg_att_v_tabular.sh tri-debug 0

PLATFORM=${1:-tri}
SEED=${2:-0}
WANDB_MODE_ARG=${3:-offline}

# ──────────────────────────────────────────────────────────────────
# Auto-submit: if not already running under SLURM, sbatch ourselves
# ──────────────────────────────────────────────────────────────────
if [ -z "$SLURM_JOB_ID" ]; then
    case "$PLATFORM" in
        fir)
            sbatch \
                --account=def-jtyao_gpu \
                --job-name=B9_coala_ipd_s${SEED} \
                --nodes=1 \
                --ntasks=1 \
                --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1 \
                --cpus-per-task=6 \
                --mem=16G \
                --time=00:15:00 \
                --output=/scratch/lichenqi/output/%x-%N-%j.out \
                "$0" "$@"
            ;;
        tri)
            sbatch \
                --account=def-jtyao \
                --job-name=B9_coala_ipd_s${SEED} \
                --gpus-per-node=h100:1 \
                --cpus-per-task=6 \
                --time=08:00:00 \
                --output=/scratch/lichenqi/output/%x-%N-%j.out \
                "$0" "$@"
            ;;
        tri-debug)
            sbatch \
                --account=def-jtyao \
                --job-name=B9_coala_ipd_dbg_s${SEED} \
                --gpus-per-node=h100:1 \
                --cpus-per-task=6 \
                --time=1:00:00 \
                --output=/scratch/lichenqi/debug_output/%x-%N-%j.out \
                "$0" "$@"
            ;;
        *)
            echo "Unknown platform '$PLATFORM'. Use: fir, tri, or tri-debug"
            ;;
    esac
    exit $?
fi

# ──────────────────────────────────────────────────────────────────
# Actual job (running under SLURM from here)
# ──────────────────────────────────────────────────────────────────
module load StdEnv/2023 gcc/12.3
module load cuda/12.6
module load python/3.11.5
source /project/def-jtyao/lichenqi/pax_env_py3.11.5/bin/activate

export TMPDIR="${SLURM_TMPDIR:-/tmp}"

export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p "$MPLCONFIGDIR"

export WANDB_API_KEY="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"
mkdir -p "$TMPDIR/wandb" "$TMPDIR/wandb-cache" "$TMPDIR/wandb_config"

export WANDB_DIR="$TMPDIR/wandb"
export WANDB_CACHE_DIR="$TMPDIR/wandb-cache"
export WANDB_CONFIG_DIR="$TMPDIR/wandb_config"
export WANDB_SERVICE_TRANSPORT=tcp
export WANDB__SERVICE_WAIT=180
export WANDB_INIT_TIMEOUT=180
export WANDB_START_METHOD=thread

EXPERIMENT="ipd=coala_pg_v_tabular"
RESULTS_DIR="/scratch/lichenqi/results/B9_coala_ipd_seed${SEED}"
HYDRA_DIR="$TMPDIR/hydra_output"
EXP_OUTPUT="$HYDRA_DIR/exp"
mkdir -p "$RESULTS_DIR"

start_time=$(date +%s)
echo "=== B9 COALA-PG | Platform: $PLATFORM | Seed: $SEED | wandb: $WANDB_MODE_ARG | $(date '+%Y-%m-%d %H:%M:%S') ==="

cd /project/def-jtyao/lichenqi/pax

case "$PLATFORM" in
    fir|tri)
        # NOTE: COALA-PG is a policy-gradient runner. It has no ES population
        # and no resume support, so no popsize / resume_dir overrides here.
        python -m pax.experiment +experiment/$EXPERIMENT \
            seed=$SEED \
            ++wandb.mode=$WANDB_MODE_ARG \
            hydra.run.dir=$HYDRA_DIR
        ;;
    tri-debug|fir-debug)
        # Smoke test only: tiny M / T / B. These settings are far too small for
        # the estimator to shape anything (the co-player barely learns within a
        # meta-trajectory) — they only verify the pipeline runs end to end.
        echo "=== Debug smoke test (tiny params; not a learning run) ==="
        python -m pax.experiment +experiment/$EXPERIMENT \
            seed=$SEED \
            ++num_iters=10 \
            ++num_outer_steps=4 \
            ++num_inner_steps=10 \
            ++num_envs=4 \
            ++num_opps=2 \
            ++ppo1.num_minibatches=4 \
            ++save_interval=5 \
            ++wandb.mode=$WANDB_MODE_ARG \
            hydra.run.dir=$HYDRA_DIR
        ;;
esac

# ──────────────────────────────────────────────────────────────────
# Copy results to persistent storage
# ──────────────────────────────────────────────────────────────────
echo "Copying final results to $RESULTS_DIR ..."
cp -rL "$EXP_OUTPUT"/baseline-COALA-PG-*/ "$RESULTS_DIR/" 2>/dev/null
if [ "$WANDB_MODE_ARG" = "offline" ]; then
    mkdir -p /scratch/lichenqi/wandb_saved
    cp -rL "$WANDB_DIR"/wandb/offline-run-* /scratch/lichenqi/wandb_saved/ 2>/dev/null || true
    echo ">>> Sync later from a LOGIN node: wandb sync /scratch/lichenqi/wandb_saved/offline-run-* <<<"
fi

end_time=$(date +%s)
echo "=== Done: B9 COALA-PG $PLATFORM seed=$SEED | Elapsed: $((end_time - start_time))s ==="
