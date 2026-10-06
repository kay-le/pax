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
#   bash E5_ipd_eval_coala_pg.sh <platform> <condition> [num_seeds|seed_csv] [seed_start] [wandb_mode]
#
# condition: selfish | welfare | constrained
# aliases: unconstrained -> welfare, lagrangian -> constrained
#   selfish       -> eval_selfish_coala_pg_v_tabular     CoalaPG, 1 value head
#                    trained from: coala_pg_v_tabular
#   welfare       -> eval_welfare_coala_pg_v_tabular     CoalaPG, 1 value head
#                    trained from: welfare_coala_pg_v_tabular
#   constrained   -> eval_lagrangian_coala_pg_v_tabular  CoalaPGLagrangian, 3 heads
#                    trained from: lagrangian_coala_pg_v_tabular
#
# Before running, either set `model_path` / `run_path` in the selected eval
# YAML, or provide a seed that appears in that YAML's `checkpoints` map.
#
# The condition MUST match the checkpoint architecture. `selfish` and
# `welfare` use CoalaPG with one value head. `constrained` uses
# CoalaPGLagrangian with three value heads.
#
# Examples:
#   # one trained seed, 20 fresh co-players
#   bash E5_ipd_eval_coala_pg.sh fir constrained
#   # 5 co-players, seeds 100-104
#   bash E5_ipd_eval_coala_pg.sh fir welfare 5 100
#   # non-contiguous training/checkpoint seeds from the eval YAML checkpoint map
#   bash E5_ipd_eval_coala_pg.sh fir constrained 13992,14002,14012,24343,44545,67655
#   # smoke test
#   bash E5_ipd_eval_coala_pg.sh tri-debug constrained
#
# To sweep non-contiguous training/checkpoint seeds, pass them as a comma-
# separated third argument.

PLATFORM=${1:-tri}
CONDITION=${2:-lagrangian}
EXTRA_OVERRIDES=()
SEEDS_ARG=${3:-20}
if [[ "$SEEDS_ARG" == *,* ]]; then
    IFS=',' read -r -a SEED_LIST <<< "$SEEDS_ARG"
    NUM_SEEDS=${#SEED_LIST[@]}
    SEED_START=${SEED_LIST[0]}
    WANDB_MODE_ARG=${4:-online}
else
    NUM_SEEDS=$SEEDS_ARG
    SEED_START=${4:-0}
    WANDB_MODE_ARG=${5:-online}
    SEED_LIST=()
    for ((offset=0; offset<NUM_SEEDS; offset++)); do
        SEED_LIST+=($((SEED_START + offset)))
    done
fi

case "$CONDITION" in
    selfish)    EXPERIMENT_NAME="eval_selfish_coala_pg_v_tabular" ;;
    welfare|unconstrained)
                EXPERIMENT_NAME="eval_welfare_coala_pg_v_tabular" ;;
    constrained|lagrangian)
                EXPERIMENT_NAME="eval_lagrangian_coala_pg_v_tabular" ;;
    *)
        echo "Unknown condition '$CONDITION'. Use: selfish | welfare | constrained" >&2
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
if [[ "$SEEDS_ARG" == *,* ]]; then
    echo "  seeds     : ${SEED_LIST[*]}  ($NUM_SEEDS checkpoint/eval seeds)"
else
    echo "  seeds     : ${SEED_START}..${SEED_END}  ($NUM_SEEDS fresh co-players)"
fi
echo "  wandb     : $WANDB_MODE_ARG"
echo "  started   : $(date '+%Y-%m-%d %H:%M:%S')"
echo

cd /project/def-jtyao/lichenqi/pax

case "$PLATFORM" in
    fir|tri)
        for ((offset=0; offset<NUM_SEEDS; offset++)); do
            run_seed=${SEED_LIST[$offset]}
            run_start_time=$(date +%s)
            echo "=== Trial $((offset + 1))/$NUM_SEEDS | eval seed=$run_seed | $(date '+%H:%M:%S') ==="

            python -m pax.experiment +experiment/$EXPERIMENT \
                seed=$run_seed \
                ++wandb.mode=$WANDB_MODE_ARG "${EXTRA_OVERRIDES[@]}" \
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
