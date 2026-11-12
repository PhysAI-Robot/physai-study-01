#!/bin/bash
# The looks outside the training ranges on Newton with the fixed renderer (looks table): all four changes together, and each alone
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; M=research/configs/so101_randomized_pick_place.yaml; HF=hf:rizgiak/physai-study-01-checkpoints
mkdir -p $STUDY/newton_looks
run() { # model checkpoint look episodes
  o=$STUDY/newton_looks/$1_$3
  [ -f $o.json ] || uv run python research/experiments/newton_look.py $3 -- --manifest $M --sim mujoco --render newton --policy lerobot --checkpoint $2 --seed 1000 --episodes $4 --json $o.json > $o.log 2>&1
  echo "$1 $3: $(grep 'policy=lerobot' $o.log | cut -c1-60) $(date)" >> $STUDY/logs/newton_looks_queue.log
}
for m in act_randomized_look_seed0 act_randomized_look_seed1 act_randomized_look_seed2 act_nominal_unseeded act_both_renderers_unseeded; do run $m $HF/$m all_unseen 300; done
for m in act_randomized_look_seed0 act_randomized_look_seed1; do for c in floor_fine floor_bw sky_dark light_low; do run $m $HF/$m $c 100; done; done
echo "newton_looks queue done $(date)" >> $STUDY/logs/newton_looks_queue.log
