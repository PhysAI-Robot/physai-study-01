#!/bin/bash
# The four models with the biggest drop between the two Newton renderers, scored on Newton with the renderer as it was before the fix (pads table)
cd "$(dirname "$0")/../.."
STUDY=outputs/study01
M=research/configs/so101_randomized_pick_place.yaml
HF=hf:rizgiak/physai-study-01-checkpoints
mkdir -p $STUDY/newton_old_renderer
for n in act_nominal_unseeded act_isaac_only_100_unseeded act_isaac_only_400_unseeded act_nominal_seed1; do
  [ -f $STUDY/newton_old_renderer/$n.json ] && continue
  uv run python research/experiments/newton_renderer_variant.py --variant old -- --manifest $M --sim mujoco --render newton --policy lerobot --checkpoint $HF/$n --seed 1000 --episodes 300 --json $STUDY/newton_old_renderer/$n.json > $STUDY/logs/oldrend_$n.log 2>&1
  echo "$n: $(grep 'policy=lerobot' $STUDY/logs/oldrend_$n.log | cut -c1-70) $(date)" >> $STUDY/logs/old_renderer.log
done
echo "old renderer done $(date)" >> $STUDY/logs/old_renderer.log
