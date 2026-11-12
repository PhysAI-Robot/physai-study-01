#!/bin/bash
# The randomized look plus black-and-white floor tiles (30% of the episodes; Section "attribution"). 400 demonstrations (seeds 0-399), 120k steps, chunk 100,
# training seeds 1, 2, 3. Each model: MuJoCo, Newton (standard look) 1000-1299, Isaac 1000-1099, and Newton on the held-out look.
# It can run beside newton_looks.sh (Newton, little GPU); no Isaac Sim process runs at the same time.
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; M=research/configs/so101_randomized_pick_place.yaml; CFG=research/configs/so101_randomized_pick_place_visual_dr_bw.yaml
mkdir -p $STUDY/bw_look_runs
DATA=data/study01/mujoco_randomized-look-bw_400_seeds0-399
[ -f $DATA/meta.json ] || uv run python research/scripts/collect_demos.py --manifest $CFG --episodes 400 --seed 0 --dataset $DATA > $STUDY/logs/bw_look_runs_collect.log 2>&1
echo "collected $(tail -n 2 $STUDY/logs/bw_look_runs_collect.log | tr '\n' ' ' | cut -c1-120) $(date)" >> $STUDY/logs/bw_look_runs_queue.log
for s in 1 2 3; do
  out=$STUDY/act_randlook_bw_120k_seed$s; n=randlook_bw_seed$s
  [ -f $STUDY/logs/bw_look_runs_train_$s.done ] || { uv run python research/imitation_learning/train_act.py --dataset $DATA --steps 120000 --seed $s --chunk-size 100 --num-workers 0 --out $out > $STUDY/logs/bw_look_runs_train_$s.out.log 2> $STUDY/logs/bw_look_runs_train_$s.err.log && touch $STUDY/logs/bw_look_runs_train_$s.done; echo "trained $n $(date)" >> $STUDY/logs/bw_look_runs_queue.log; }
  [ -f $STUDY/bw_look_runs/${n}_mujoco.json ] || uv run python scripts/eval_policy.py --manifest $M --policy lerobot --checkpoint $out --seed 1000 --episodes 300 --json $STUDY/bw_look_runs/${n}_mujoco.json > $STUDY/logs/bw_look_runs_${n}_mujoco.log 2>&1
  [ -f $STUDY/bw_look_runs/${n}_newton.json ] || uv run python scripts/eval_policy.py --manifest $M --sim mujoco --render newton --policy lerobot --checkpoint $out --seed 1000 --episodes 300 --json $STUDY/bw_look_runs/${n}_newton.json > $STUDY/logs/bw_look_runs_${n}_newton.log 2>&1
  [ -f $STUDY/bw_look_runs/${n}_heldout.json ] || uv run python research/experiments/newton_look.py heldout_look -- --manifest $M --sim mujoco --render newton --policy lerobot --checkpoint $out --seed 1000 --episodes 300 --json $STUDY/bw_look_runs/${n}_heldout.json > $STUDY/logs/bw_look_runs_${n}_heldout.log 2>&1
  echo "scored $n: $(grep -h 'policy=lerobot' $STUDY/logs/bw_look_runs_${n}_mujoco.log $STUDY/logs/bw_look_runs_${n}_newton.log $STUDY/logs/bw_look_runs_${n}_heldout.log | cut -c1-60 | tr '\n' ' ') $(date)" >> $STUDY/logs/bw_look_runs_queue.log
done
for s in 1 2 3; do for k in 1000 1010 1020 1030 1040 1050 1060 1070 1080 1090; do
  d=$STUDY/isaac_bw_look_runs/randlook_bw_seed$s/s$k; [ -f $d/merged.json ] && continue
  OMNI_KIT_ACCEPT_EULA=YES uv run python scripts/eval_policy_sharded.py --out $d --seed $k --episodes 10 --shard-size 10 -- --manifest $M --sim isaac --policy lerobot --checkpoint $STUDY/act_randlook_bw_120k_seed$s --video failures --camera front --out $d/videos > $STUDY/logs/isaac_bw_look_runs_s${s}_$k.log 2>&1
done; echo "isaac randlook_bw_seed$s $(date)" >> $STUDY/logs/bw_look_runs_queue.log; done
echo "bw_look_runs queue done $(date)" >> $STUDY/logs/bw_look_runs_queue.log
