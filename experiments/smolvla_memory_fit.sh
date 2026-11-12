#!/bin/bash
# Memory test of the SmolVLA recipe (budget table): 12 optimizer steps of the recipe batch 64, split into MICRO x ACCUM, fp32 or bf16; peak memory and seconds per step.
cd "$(dirname "$0")/../.."
D=data/study01/mujoco_nominal_200_seeds0-199; O=outputs/study01/o0
run() { # NAME ACCUM EXTRA...
  n=$1; a=$2; shift 2
  uv run python research/imitation_learning/train_smolvla.py --dataset $D --episodes 20 --steps 12 --batch-size 64 --accum $a --log-every 4 --save-every 100000 --warmup 4 --out $O/fit_$n --num-workers 2 "$@" > $O/fit_$n.log 2>&1
}
run fp32_a32 32
run fp32_a16 16
run bf16_a16 16 --bf16
run bf16_a8 8 --bf16
run bf16_a4 4 --bf16
echo "o0 fit done $(date)" >> $O/fit.log
