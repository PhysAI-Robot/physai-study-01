#!/bin/bash
# The nine nominal-look ACT runs on Newton, seeds 1000-1299, with the key light mirrored (Section "failures"). Do not run it beside another GPU job.
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; M=research/configs/so101_randomized_pick_place.yaml; HF=hf:rizgiak/physai-study-01-checkpoints
mkdir -p $STUDY/light_mirrored
run() { # name checkpoint
  [ -f $STUDY/light_mirrored/$1.json ] || uv run python research/experiments/newton_light_variant.py -- --manifest $M --sim mujoco --render newton --policy lerobot --checkpoint $2 --seed 1000 --episodes 300 --json $STUDY/light_mirrored/$1.json > $STUDY/logs/light_mirrored_$1.log 2>&1
  echo "$1: $(grep 'policy=lerobot' $STUDY/logs/light_mirrored_$1.log | cut -c1-60) $(date)" >> $STUDY/logs/light_mirrored_queue.log
}
for n in act_nominal_unseeded act_nominal_seed1 act_nominal_seed2 act_nominal_seed3; do run $n $HF/$n; done
for s in 4 5; do run nominal200_seed$s $HF/act_nominal_seed$s; done
for s in 1 2 3; do run nominal400_seed$s $HF/act_nominal400_seed$s; done
echo "light_mirrored queue done $(date)" >> $STUDY/logs/light_mirrored_queue.log
