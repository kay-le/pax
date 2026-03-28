#!/bin/bash
# Unified launcher for W2 IPDITM welfare shaper att (self-play)
#
# Usage:
#   bash W2_sp_ipditm_welfare_shaper_att_one.sh <platform> <seed>
#
# Platforms:
#   fir        — Fir cluster, 4×H100, ~3h wall time
#   tri        — Trillium cluster, 4×H100, 1d20h wall time
#   tri-debug  — Trillium, 1×H100, 1h, small params, runs twice to test resume
#
# Examples:
#   bash W2_sp_ipditm_welfare_shaper_att_one.sh fir 21
#   bash W2_sp_ipditm_welfare_shaper_att_one.sh tri 0
#   bash W2_sp_ipditm_welfare_shaper_att_one.sh tri-debug 21

PLATFORM=${1:-tri}
SEED=${2:-0}

# ──────────────────────────────────────────────────────────────────
# Auto-submit: if not already running under SLURM, sbatch ourselves
# ──────────────────────────────────────────────────────────────────
if [ -z "$SLURM_JOB_ID" ]; then
    case "$PLATFORM" in
        fir)
            sbatch \
                --account=def-jtyao_gpu \
                --job-name=W2sp_ipditm_s${SEED} \
                --gpus-per-node=h100:4 \
                --cpus-per-task=12 \
                --mem=20G \
                --time=3:00:00 \
                --output=/scratch/lichenqi/output/%x-%N-%j.out \
                "$0" "$@"
            ;;
        tri)
            sbatch \
                --account=def-jtyao \
                --job-name=W2sp_ipditm_s${SEED} \
                --gpus-per-node=h100:4 \
                --time=1-20:00:00 \
                --output=/scratch/lichenqi/output/%x-%N-%j.out \
                "$0" "$@"
            ;;
        tri-debug)
            sbatch \
                --account=def-jtyao \
                --job-name=W2sp_ipditm_dbg_s${SEED} \
                --gpus-per-node=h100:1 \
                --cpus-per-task=6 \
                --time=1:00:00 \
                --output=/scratch/lichenqi/%x-%N-%j.out \
                "$0" "$@"
            ;;
        *)
            echo "ERROR: Unknown platform '$PLATFORM'. Use: fir, tri, tri-debug"
            exit 1
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
source /home/lichenqi/pax_env_py3.11.5/bin/activate

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

EXPERIMENT="ipditm=welfare_shaper_att"
RESUME_DIR="/scratch/lichenqi/resume/W2_sp_ipditm_seed${SEED}"
mkdir -p "$RESUME_DIR"

start_time=$(date +%s)
echo "=== Platform: $PLATFORM | Seed: $SEED | $(date '+%Y-%m-%d %H:%M:%S') ==="

cd /home/lichenqi/pax

case "$PLATFORM" in
    fir|tri)
        # Full training run — uses config defaults (5000 iters, popsize 128, etc.)
        python -m pax.experiment +experiment/$EXPERIMENT \
            seed=$SEED \
            ++num_devices=4 \
            ++welfare.resume_dir=$RESUME_DIR \
            hydra.run.dir=$TMPDIR/hydra_output
        ;;
    tri-debug)
        # Debug run — small params, run TWICE to test save/resume
        echo "=== Debug run 1/2 ==="
        python -m pax.experiment +experiment/$EXPERIMENT \
            seed=$SEED \
            ++num_iters=10 \
            ++popsize=40 \
            ++num_outer_steps=20 \
            ++num_inner_steps=80 \
            ++num_devices=1 \
            ++save_interval=5 \
            ++welfare.resume_dir=$RESUME_DIR \
            hydra.run.dir=$TMPDIR/hydra_output

        # Copy checkpoint to resume_dir so second run can find it
        cp -rL ./exp/welfare-*/ "$RESUME_DIR/"

        echo "=== Debug run 2/2 (testing resume) ==="
        python -m pax.experiment +experiment/$EXPERIMENT \
            seed=$SEED \
            ++num_iters=10 \
            ++popsize=40 \
            ++num_outer_steps=20 \
            ++num_inner_steps=80 \
            ++num_devices=1 \
            ++save_interval=5 \
            ++welfare.resume_dir=$RESUME_DIR \
            hydra.run.dir=$TMPDIR/hydra_output
        ;;
esac

# ──────────────────────────────────────────────────────────────────
# Copy results to persistent storage
# ──────────────────────────────────────────────────────────────────
cp -rL ./exp/welfare-*/ "$RESUME_DIR/"
mkdir -p /scratch/lichenqi/wandb_saved
cp -rL "$WANDB_DIR"/wandb/offline-run-* /scratch/lichenqi/wandb_saved/ 2>/dev/null || true

end_time=$(date +%s)
echo "=== Done: $PLATFORM seed=$SEED | Elapsed: $((end_time - start_time))s ==="
