#!/bin/bash
# The four ViperX models on Newton with the held-out look, 300 seeds each, no training (Section "attribution", second robot).
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; M=research/configs/viperx300s_single_cube_randomized_place.yaml; HF=hf:rizgiak/physai-study-01-checkpoints
mkdir -p $STUDY/viperx_heldout_look
for n in randlook_seed1 randlook_seed2 nominal_seed1 nominal_seed2; do
  [ -f $STUDY/viperx_heldout_look/${n}_heldout.json ] || uv run python research/experiments/newton_look.py heldout_v2 -- --manifest $M --sim mujoco --render newton --policy lerobot --checkpoint $HF/viperx_act_${n/randlook/randomized_look} --seed 1000 --episodes 300 --json $STUDY/viperx_heldout_look/${n}_heldout.json > $STUDY/logs/viperx_heldout_look_${n}_heldout.log 2>&1
  echo "viperx $n heldout: $(grep -h 'policy=lerobot' $STUDY/logs/viperx_heldout_look_${n}_heldout.log | cut -c1-60) $(date)" >> $STUDY/logs/viperx_heldout_look_queue.log
done
echo "viperx_heldout_look queue done $(date)" >> $STUDY/logs/viperx_heldout_look_queue.log
