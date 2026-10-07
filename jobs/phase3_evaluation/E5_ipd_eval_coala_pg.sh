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
#   bash E5_ipd_eval_coala_pg.sh <platform> <condition> [num_eval_seeds] [eval_seed_start] [wandb_mode] [checkpoint_seed_csv] [hydra_overrides...]
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
#   # 20 eval seeds, 100-119, against selected checkpoint seeds
#   bash E5_ipd_eval_coala_pg.sh fir constrained 20 100 online 14002,24343,67655
#   # smoke test
#   bash E5_ipd_eval_coala_pg.sh tri-debug constrained
#
# To sweep non-contiguous training/checkpoint seeds, pass them as a comma-
# separated sixth argument.

PLATFORM=${1:-tri}
CONDITION=${2:-lagrangian}
EXTRA_OVERRIDES=()
NUM_EVAL_SEEDS=${3:-20}
EVAL_SEED_START=${4:-0}
WANDB_MODE_ARG=${5:-online}
CHECKPOINTS_ARG=${6:-}
if [ "$#" -ge 7 ]; then
    EXTRA_OVERRIDES=("${@:7}")
fi

case "$CONDITION" in
    selfish)
        EXPERIMENT_NAME="eval_selfish_coala_pg_v_tabular"
        DEFAULT_CHECKPOINTS="13992,14002,14012,24343,44545,67655"
        ;;
    welfare|unconstrained)
        EXPERIMENT_NAME="eval_welfare_coala_pg_v_tabular"
        DEFAULT_CHECKPOINTS="13992,14002,14012,24343,44545,67655"
        ;;
    constrained|lagrangian)
        EXPERIMENT_NAME="eval_lagrangian_coala_pg_v_tabular"
        DEFAULT_CHECKPOINTS="13992,14002,14012,24343,44545,67655"
        ;;
    *)
        echo "Unknown condition '$CONDITION'. Use: selfish | welfare | constrained" >&2
        exit 1
        ;;
esac

CHECKPOINTS_ARG=${CHECKPOINTS_ARG:-$DEFAULT_CHECKPOINTS}
IFS=',' read -r -a CHECKPOINT_LIST <<< "$CHECKPOINTS_ARG"
NUM_CHECKPOINTS=${#CHECKPOINT_LIST[@]}

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
EVAL_SEED_END=$((EVAL_SEED_START + NUM_EVAL_SEEDS - 1))
TOTAL_TRIALS=$((NUM_CHECKPOINTS * NUM_EVAL_SEEDS))

start_time=$(date +%s)
echo "=== E5 COALA-PG EVAL ==="
echo "  platform  : $PLATFORM"
echo "  condition : $CONDITION  ($EXPERIMENT_NAME)"
echo "  checkpoint: from pax/conf/experiment/ipd/${EXPERIMENT_NAME}.yaml"
echo "  checkpoints: ${CHECKPOINT_LIST[*]}  ($NUM_CHECKPOINTS checkpoints)"
echo "  eval seeds : ${EVAL_SEED_START}..${EVAL_SEED_END}  ($NUM_EVAL_SEEDS per checkpoint)"
echo "  trials    : $TOTAL_TRIALS"
echo "  wandb     : $WANDB_MODE_ARG"
if [ "${#EXTRA_OVERRIDES[@]}" -gt 0 ]; then
    echo "  overrides : ${EXTRA_OVERRIDES[*]}"
fi
echo "  started   : $(date '+%Y-%m-%d %H:%M:%S')"
echo

cd /project/def-jtyao/lichenqi/pax

case "$PLATFORM" in
    fir|tri)
        trial_idx=0
        for ((ckpt_offset=0; ckpt_offset<NUM_CHECKPOINTS; ckpt_offset++)); do
            checkpoint_seed=${CHECKPOINT_LIST[$ckpt_offset]}
            for ((eval_offset=0; eval_offset<NUM_EVAL_SEEDS; eval_offset++)); do
                run_seed=$((EVAL_SEED_START + eval_offset))
                trial_idx=$((trial_idx + 1))
                run_start_time=$(date +%s)
                echo "=== Trial ${trial_idx}/${TOTAL_TRIALS} | checkpoint seed=$checkpoint_seed | eval seed=$run_seed | $(date '+%H:%M:%S') ==="

                python -m pax.experiment +experiment/$EXPERIMENT \
                    seed=$run_seed \
                    ++checkpoint_seed=$checkpoint_seed \
                    ++wandb.name=ckpt-${checkpoint_seed}-eval-seed-${run_seed} \
                    ++wandb.mode=$WANDB_MODE_ARG "${EXTRA_OVERRIDES[@]}" \
                    hydra.run.dir="$HYDRA_DIR/ckpt_${checkpoint_seed}/eval_seed_${run_seed}"

                run_status=$?
                run_end_time=$(date +%s)
                if [ $run_status -ne 0 ]; then
                    echo "=== FAILED: checkpoint seed=$checkpoint_seed | eval seed=$run_seed | Elapsed: $((run_end_time - run_start_time))s ===" >&2
                    exit $run_status
                fi
                echo "=== Completed: checkpoint seed=$checkpoint_seed | eval seed=$run_seed | Elapsed: $((run_end_time - run_start_time))s ==="
            done
        done
        ;;
    tri-debug|fir-debug)
        # Smoke test: one trial, fewer inner episodes. Verifies the checkpoint
        # loads and the rollout runs; the numbers are not a result.
        checkpoint_seed=${CHECKPOINT_LIST[0]}
        run_seed=$EVAL_SEED_START
        echo "=== Debug smoke test (1 trial, M=5) ==="
        python -m pax.experiment +experiment/$EXPERIMENT \
            seed=$run_seed \
            ++checkpoint_seed=$checkpoint_seed \
            ++wandb.name=ckpt-${checkpoint_seed}-eval-seed-${run_seed} \
            ++num_outer_steps=5 \
            ++wandb.mode=$WANDB_MODE_ARG \
            "${EXTRA_OVERRIDES[@]}" \
            hydra.run.dir=$HYDRA_DIR
        ;;
esac

end_time=$(date +%s)
echo "=== Done: E5 eval $CONDITION | $TOTAL_TRIALS trials | Elapsed: $((end_time - start_time))s ==="
