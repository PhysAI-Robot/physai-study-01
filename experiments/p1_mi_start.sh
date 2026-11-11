#!/bin/bash
# V3 P1 cell MI-start (SO-101): MuJoCo physics with Isaac Sim's pictures and Isaac Sim's cube start, seeds 1000-1099. Waits for the MM-start queue (one process at a time);
# one Isaac process at a time; finished shards are skipped; failure videos only.
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; HF=hf:rizgiak/physai-study-01-checkpoints; M=research/configs/so101_randomized_pick_place.yaml; export OMNI_KIT_ACCEPT_EULA=YES
while ! grep -q "p1 mm_start done" $STUDY/logs/p1_queue.log 2>/dev/null; do sleep 60; done
for m in nominal_seed1 nominal_seed2 randlook_seed0; do
  ck=act_$m; [ $m = randlook_seed0 ] && ck=act_randomized_look_seed0
  for k in 1000 1010 1020 1030 1040 1050 1060 1070 1080 1090; do
    d=$STUDY/mi_start/so101_$m/s$k; [ -f $d/merged.json ] && continue
    uv run python scripts/eval_policy_sharded.py --out $d --seed $k --episodes 10 --shard-size 10 -- --manifest $M --sim mujoco --render isaac --policy lerobot --checkpoint $HF/$ck --cube-start research/experiments/cube_start_so101_$m.json --video failures --camera front --out $d/videos > $STUDY/logs/p1_mi_so101_${m}_s$k.log 2>&1
  done
  echo "mi_start so101_$m $(date)" >> $STUDY/logs/p1_queue.log
done
echo "p1 mi_start done $(date)" >> $STUDY/logs/p1_queue.log
