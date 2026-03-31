SEED=${1:-0}
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

EXPERIMENT="ipd=welfare_shaper_att_v_tabular"
RESUME_DIR="/scratch/lichenqi/resume/W1_sp_ipd_seed${SEED}"
HYDRA_DIR="$TMPDIR/hydra_output"
# Hydra changes CWD to HYDRA_DIR inside Python, so save_dir ends up here:
EXP_OUTPUT="$HYDRA_DIR/exp"
mkdir -p "$RESUME_DIR"

start_time=$(date +%s)
echo "=== Platform: fir-debug | Seed: $SEED | $(date '+%Y-%m-%d %H:%M:%S') ==="

cd /project/def-jtyao/lichenqi/pax


echo "=== Debug run 1/2 (gen 0-5) ==="
python -m pax.experiment +experiment/$EXPERIMENT \
    seed=$SEED \
    ++num_iters=11 \
    ++popsize=40 \
    ++num_outer_steps=20 \
    ++num_inner_steps=80 \
    ++num_devices=1 \
    ++save_interval=5 \
    ++welfare.resume_dir=$RESUME_DIR \
    hydra.run.dir=$HYDRA_DIR

# Copy checkpoint to resume_dir so second run can find it
echo "Copying checkpoints from $EXP_OUTPUT to $RESUME_DIR ..."
cp -rL "$EXP_OUTPUT"/welfare-*/ "$RESUME_DIR/" 2>/dev/null
echo "Resume dir contents:"
find "$RESUME_DIR" -name "generation_*" | sort

echo "=== Debug run 2/2 (gen 6-10, testing resume) ==="
python -m pax.experiment +experiment/$EXPERIMENT \
    seed=$SEED \
    ++num_iters=21 \
    ++popsize=40 \
    ++num_outer_steps=20 \
    ++num_inner_steps=80 \
    ++num_devices=1 \
    ++save_interval=5 \
    ++welfare.resume_dir=$RESUME_DIR \
    hydra.run.dir=$HYDRA_DIR

# ──────────────────────────────────────────────────────────────────
# Copy results to persistent storage
# ──────────────────────────────────────────────────────────────────
echo "Copying final results to $RESUME_DIR ..."
cp -rL "$EXP_OUTPUT"/welfare-*/ "$RESUME_DIR/" 2>/dev/null
mkdir -p /scratch/lichenqi/wandb_saved
cp -rL "$WANDB_DIR"/wandb/offline-run-* /scratch/lichenqi/wandb_saved/ 2>/dev/null || true

end_time=$(date +%s)
echo "=== Done: $PLATFORM seed=$SEED | Elapsed: $((end_time - start_time))s ==="
