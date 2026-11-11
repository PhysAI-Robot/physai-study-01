#!/bin/bash
# V2 N13: Isaac Sim physics with MuJoCo pictures (--sim isaac --render mujoco), seeds 1000-1099, no training.
# Models: act_nominal_seed1, act_nominal_seed2 (N3), act_randomized_look_seed0 (consistency check). One Isaac process at a time; finished shards are skipped.
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; M=research/configs/so101_randomized_pick_place.yaml; HF=hf:rizgiak/physai-study-01-checkpoints
mkdir -p $STUDY/n13
for m in act_nominal_seed1 act_nominal_seed2 act_randomized_look_seed0; do
  for k in 1000 1010 1020 1030 1040 1050 1060 1070 1080 1090; do
    d=$STUDY/n13/$m/s$k; [ -f $d/merged.json ] && continue
    OMNI_KIT_ACCEPT_EULA=YES uv run python scripts/eval_policy_sharded.py --out $d --seed $k --episodes 10 --shard-size 10 -- --manifest $M --sim isaac --render mujoco --policy lerobot --checkpoint $HF/$m --video failures --camera front --out $d/videos > $STUDY/logs/n13_${m}_s$k.log 2>&1
  done
  echo "$m $(date)" >> $STUDY/logs/n13_queue.log
done
echo "n13 queue done $(date)" >> $STUDY/logs/n13_queue.log
