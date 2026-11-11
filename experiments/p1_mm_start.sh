#!/bin/bash
# V3 P1 cell MM-start: MuJoCo physics and pictures with Isaac Sim's cube start (the table of the same model's Isaac-physics run), no Isaac Sim process.
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; HF=hf:rizgiak/physai-study-01-checkpoints; mkdir -p $STUDY/mm_start
SO=research/configs/so101_randomized_pick_place.yaml; VX=research/configs/viperx300s_single_cube_randomized_place.yaml
run() { # NAME MANIFEST CHECKPOINT EPISODES
  o=$STUDY/mm_start/$1.json; [ -f $o ] && return
  uv run python scripts/eval_policy.py --manifest $2 --policy lerobot --checkpoint $3 --seed 1000 --episodes $4 --cube-start research/experiments/cube_start_$1.json --json $o > $STUDY/logs/p1_mm_$1.log 2>&1
}
run so101_nominal_seed1 $SO $HF/act_nominal_seed1 100
run so101_nominal_seed2 $SO $HF/act_nominal_seed2 100
run so101_randlook_seed0 $SO $HF/act_randomized_look_seed0 100
run viperx_nominal_seed1 $VX $STUDY/n4/act_nominal_seed1 300
run viperx_nominal_seed2 $VX $STUDY/n4/act_nominal_seed2 300
echo "p1 mm_start done $(date)" >> $STUDY/logs/p1_queue.log
