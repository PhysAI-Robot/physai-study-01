#!/bin/bash
# The pad test (pads table): seeds that failed on Newton, run again with the renderer as it is (control) and with the collision geoms hidden
cd "$(dirname "$0")/../.."
STUDY=outputs/study01
M=research/configs/so101_randomized_pick_place.yaml
HF=checkpoints/rizgiak/physai-study-01-checkpoints
declare -A CK=( [nominal_seed3]=$HF/act_nominal_seed3 [nominal400_seed1]=$HF/act_nominal400_seed1 )
for model in nominal_seed3 nominal400_seed1; do
  for seed in $(python -c "import json;print(' '.join(map(str,json.load(open('research/experiments/newton_pad_test_seeds.json'))['$model'])))"); do
    for variant in control hidden; do
      out=$STUDY/pad_test/${model}_${variant}_$seed.json
      [ -f $out ] && continue
      uv run python research/experiments/newton_renderer_variant.py --variant $variant -- --manifest $M --sim mujoco --render newton --policy lerobot --checkpoint ${CK[$model]} --seed $seed --episodes 1 --json $out > $STUDY/logs/pad_${model}_${variant}_$seed.log 2>&1
    done
  done
done
echo "pad test done $(date)" >> $STUDY/logs/pad_test.log
