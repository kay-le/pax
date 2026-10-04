#!/bin/bash
# Unified launcher for E5 IPD COALA-PG evaluation.
#
# Evaluates the trained shaper checkpoint specified in the selected eval YAML
# against NUM_SEEDS independent fresh naive A2C learners. The shaper is frozen;
# the co-player learns for num_outer_steps episodes per trial. Per-episode
# returns, welfare and IR slack are logged so the CONVERGED behaviour can be
# read off.
#
# Usage:
#   bash E5_ipd_eval_coala_pg.sh <platform> <condition> [num_seeds] [seed_start] [wandb_mode]
#
# condition: selfish | unconstrained | lagrangian
# aliases: welfare -> unconstrained, constrained -> lagrangian
#   selfish    -> eval_selfish_coala_pg_v_tabular     CoalaPG,           1 value head
#                 trained from: coala_pg_v_tabular
#   unconstrained
#              -> eval_unconstrained_lagrangian_coala_pg_v_tabular
#                 CoalaPGLagrangian, 3 value heads
#                 trained from: unconstrained_lagrangian_coala_pg_v_tabular
#                 (the controlled ablation: same architecture as the
#                  lagrangian condition, multipliers frozen at zero. Add
#                  ++agent1=CoalaPG if your welfare checkpoint instead came
#                  from the standalone welfare_coala_pg_v_tabular config.)
#   lagrangian -> eval_lagrangian_coala_pg_v_tabular  CoalaPGLagrangian, 3 value heads
#                 trained from: lagrangian_coala_pg_v_tabular
#
# `unconstrained` and `lagrangian` checkpoints are architecturally IDENTICAL
# (both 3-head), but they now have separate training config names, W&B groups
# and result directories.
#
# Before running, set `model_path` and, if needed, `run_path` in the selected
# eval YAML.
#
# The condition MUST match the checkpoint architecture. `selfish` uses CoalaPG
# with one value head. `unconstrained` and `lagrangian` both use
# CoalaPGLagrangian with three value heads; they differ by trained objective,
# W&B group and result directory, not by parameter-tree shape.
#
# Examples:
#   # one trained seed, 20 fresh co-players
#   bash E5_ipd_eval_coala_pg.sh fir lagrangian
#   # 5 co-players, seeds 100-104
#   bash E5_ipd_eval_coala_pg.sh fir unconstrained 5 100
#   # smoke test
#   bash E5_ipd_eval_coala_pg.sh tri-debug lagrangian
#
# To sweep training seeds, call this once per checkpoint.

PLATFORM=${1:-tri}
CONDITION=${2:-lagrangian}
NUM_SEEDS=${3:-20}
SEED_START=${4:-0}
WANDB_MODE_ARG=${5:-online}

case "$CONDITION" in
    selfish)    EXPERIMENT_NAME="eval_selfish_coala_pg_v_tabular" ;;
    unconstrained|welfare)
                EXPERIMENT_NAME="eval_unconstrained_lagrangian_coala_pg_v_tabular" ;;
    lagrangian|constrained)
                EXPERIMENT_NAME="eval_lagrangian_coala_pg_v_tabular" ;;
    *)
        echo "Unknown condition '$CONDITION'. Use: selfish | unconstrained | lagrangian" >&2
        exit 1
        ;;
esac

if [ -z "$SLURM_JOB_ID" ]; then
    case "$PLATFORM" in
        fir)
            sbatch \
                --account=def-jtyao_gpu \
                --job-name=E5_eval_${CONDITION} \
                --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1 \
                --cpus-per-task=2 \
                --mem=8G \
                --time=1:00:00 \
                --output=/scratch/lichenqi/eval/output/%x-%N-%j.out \
                "$0" "$@"
            ;;
        tri)
            sbatch \
                --account=def-jtyao \
                --job-name=E5_eval_${CONDITION} \
                --gpus-per-node=h100:1 \
                --cpus-per-task=2 \
                --mem=8G \
                --time=1:00:00 \
                --output=/scratch/lichenqi/output/%x-%N-%j.out \
                "$0" "$@"
            ;;
        tri-debug)
            sbatch \
                --account=def-jtyao \
                --job-name=E5_eval_dbg_${CONDITION} \
                --gpus-per-node=h100:1 \
                --cpus-per-task=2 \
                --mem=8G \
                --time=0:30:00 \
                --output=/scratch/lichenqi/debug_output/%x-%N-%j.out \
                "$0" "$@"
            ;;
        *)
            echo "Unknown platform '$PLATFORM'. Use: fir, tri, or tri-debug" >&2
            exit 1
            ;;
    esac
    exit $?
fi

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

EXPERIMENT="ipd=${EXPERIMENT_NAME}"
HYDRA_DIR="$TMPDIR/hydra_output"
SEED_END=$((SEED_START + NUM_SEEDS - 1))

start_time=$(date +%s)
echo "=== E5 COALA-PG EVAL ==="
echo "  platform  : $PLATFORM"
echo "  condition : $CONDITION  ($EXPERIMENT_NAME)"
echo "  checkpoint: from pax/conf/experiment/ipd/${EXPERIMENT_NAME}.yaml"
echo "  seeds     : ${SEED_START}..${SEED_END}  ($NUM_SEEDS fresh co-players)"
echo "  wandb     : $WANDB_MODE_ARG"
echo "  started   : $(date '+%Y-%m-%d %H:%M:%S')"
echo

cd /project/def-jtyao/lichenqi/pax

case "$PLATFORM" in
    fir|tri)
        for ((offset=0; offset<NUM_SEEDS; offset++)); do
            run_seed=$((SEED_START + offset))
            run_start_time=$(date +%s)
            echo "=== Trial $((offset + 1))/$NUM_SEEDS | eval seed=$run_seed | $(date '+%H:%M:%S') ==="

            python -m pax.experiment +experiment/$EXPERIMENT \
                seed=$run_seed \
                ++wandb.mode=$WANDB_MODE_ARG \
                hydra.run.dir="$HYDRA_DIR/seed_${run_seed}"

            run_status=$?
            run_end_time=$(date +%s)
            if [ $run_status -ne 0 ]; then
                echo "=== FAILED: seed=$run_seed | Elapsed: $((run_end_time - run_start_time))s ===" >&2
                exit $run_status
            fi
            echo "=== Completed: seed=$run_seed | Elapsed: $((run_end_time - run_start_time))s ==="
        done
        ;;
    tri-debug|fir-debug)
        # Smoke test: one trial, fewer inner episodes. Verifies the checkpoint
        # loads and the rollout runs; the numbers are not a result.
        echo "=== Debug smoke test (1 trial, M=5) ==="
        python -m pax.experiment +experiment/$EXPERIMENT \
            seed=$SEED_START \
            ++num_outer_steps=5 \
            ++wandb.mode=$WANDB_MODE_ARG \
            hydra.run.dir=$HYDRA_DIR
        ;;
esac

end_time=$(date +%s)
echo "=== Done: E5 eval $CONDITION | $NUM_SEEDS trials | Elapsed: $((end_time - start_time))s ==="
