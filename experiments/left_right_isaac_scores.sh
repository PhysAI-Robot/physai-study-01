#!/bin/bash
# The left/right check (Section "failures"): Isaac Sim seeds 1100-1299 for the nine nominal-look ACT runs (four 200-demonstration runs, two more, and three 400-demonstration runs).
# Every shard is skipped when done.
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; M=research/configs/so101_randomized_pick_place.yaml; HF=hf:rizgiak/physai-study-01-checkpoints
isaac() { # name checkpoint
  for s in $(seq 1100 10 1290); do
    d=$STUDY/isaac_left_right/$1/s$s; [ -f $d/merged.json ] && continue
    OMNI_KIT_ACCEPT_EULA=YES uv run python scripts/eval_policy_sharded.py --out $d --seed $s --episodes 10 --shard-size 10 -- --manifest $M --sim isaac --policy lerobot --checkpoint $2 --video failures --camera front --out $d/videos > $STUDY/logs/isaac_left_right_$1_s$s.log 2>&1
  done
  echo "isaac 1100-1299 $1 $(date)" >> $STUDY/logs/left_right_queue.log
}
for n in act_nominal_unseeded act_nominal_seed1 act_nominal_seed2 act_nominal_seed3; do isaac $n $HF/$n; done
for s in 4 5; do isaac nominal200_seed$s $HF/act_nominal_seed$s; done
for s in 1 2 3; do isaac nominal400_seed$s $HF/act_nominal400_seed$s; done
echo "left_right queue done $(date)" >> $STUDY/logs/left_right_queue.log
