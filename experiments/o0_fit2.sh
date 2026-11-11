#!/bin/bash
# V3 O0 fit scan 2: smaller picture size inside the model (the frames are 128 px; the base model resizes to 512 with padding)
cd "$(dirname "$0")/../.."
D=data/study01/mujoco_nominal_200_seeds0-199; O=outputs/study01/o0
run() { n=$1; a=$2; shift 2
  uv run python research/imitation_learning/train_smolvla.py --dataset $D --episodes 20 --steps 12 --batch-size 64 --accum $a --log-every 4 --save-every 100000 --warmup 4 --out $O/fit_$n --num-workers 2 "$@" > $O/fit_$n.log 2>&1
}
run bf16_m256_a2 2 --bf16 --model-image-size 256
run bf16_m256_a1 1 --bf16 --model-image-size 256
run bf16_m384_a4 4 --bf16 --model-image-size 384
run fp32_m256_a2 2 --model-image-size 256
echo "o0 fit2 done $(date)" >> $O/fit.log
