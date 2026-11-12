#!/bin/bash
# The SmolVLA ladder (SmolVLA table; recipe in Section "methods", memory test in smolvla_memory_fit.sh): for each of six SmolVLA runs (nominal and randomized look, training seeds 1, 2, 3):
# train, validation on MuJoCo 900-999 (not a result), then the scores of the ladder: MuJoCo and Newton on seeds 1000-1299, Isaac Sim on 1000-1099, and the MI cell
# (MuJoCo physics, Isaac Sim pictures) and the IM cell (Isaac Sim physics, MuJoCo pictures) on 1000-1099 for all six. After the FIRST run's validation the
# stop rule written beforehand applies: fewer than 50 of 100 validation episodes started => the queue stops (reported as a negative result, not repaired). Finished steps are skipped,
# so the queue can be started again after an interruption (a training run restarts from the pretrained model). One Isaac process at a time (training never overlaps it: the steps run in order).
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; SO=research/configs/so101_randomized_pick_place.yaml; LOG=$STUDY/logs/smolvla_queue.log
NOM="data/study01/mujoco_nominal_200_seeds0-199 data/study01/mujoco_nominal_200_seeds200-399"
RL="data/study01/mujoco_randomized-look_400_seeds0-399"
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p $STUDY/smolvla
say() { echo "$* $(date)" >> $LOG; }
json_run() { # NAME OUTFILE EPISODES SEED EXTRA...   (one process, all episodes)
  n=$1; o=$2; ep=$3; sd=$4; shift 4
  [ -f $o ] || uv run python scripts/eval_policy.py --manifest $SO --policy lerobot --checkpoint $STUDY/smolvla_$n --seed $sd --episodes $ep --json $o "$@" > ${o%.json}.log 2>&1
}
isaac_cell() { # NAME CELL FLAGS...   (shards of 10 seeds, 1000-1099)
  n=$1; c=$2; shift 2
  for k in 1000 1010 1020 1030 1040 1050 1060 1070 1080 1090; do
    d=$STUDY/smolvla/$n/$c/s$k; [ -f $d/merged.json ] && continue
    uv run python scripts/eval_policy_sharded.py --out $d --seed $k --episodes 10 --shard-size 10 -- --manifest $SO --policy lerobot --checkpoint $STUDY/smolvla_$n "$@" --video failures --camera front --out $d/videos > $STUDY/logs/smolvla_${n}_${c}_s$k.log 2>&1
  done
}
first=1
for job in "nominal 1" "randlook 1" "nominal 2" "randlook 2" "nominal 3" "randlook 3"; do
  set -- $job; look=$1; seed=$2; n=${look}_seed$seed; out=$STUDY/smolvla_$n; data=$NOM; [ $look = randlook ] && data=$RL
  mkdir -p $STUDY/smolvla/$n
  if ! grep -q '"step": 20000' $out/training_log.json 2>/dev/null; then
    say "train $n start"
    uv run python research/imitation_learning/train_smolvla.py --dataset $data --steps 20000 --batch-size 64 --accum 2 --model-image-size 256 --seed $seed --out $out --log-every 100 --save-every 2000 > $STUDY/logs/smolvla_train_$n.log 2>&1
  fi
  say "train $n end"
  json_run $n $STUDY/smolvla_val_$n.json 100 900
  if [ $first = 1 ]; then
    first=0
    started=$(python -c "import json;print(sum(e.get('first_move_step',-1)>=0 for e in json.load(open('$STUDY/smolvla_val_$n.json'))['results']))" 2>/dev/null || echo 0)
    say "first run $n: $started of 100 validation episodes started"
    if [ "$started" -lt 50 ]; then say "STOPS (stop rule written beforehand): fewer than half of the validation episodes started"; exit 0; fi
  fi
  json_run $n $STUDY/smolvla/$n/mujoco.json 300 1000
  json_run $n $STUDY/smolvla/$n/newton.json 300 1000 --sim mujoco --render newton
  say "$n mujoco and newton done"
  isaac_cell $n isaac --sim isaac
  say "$n isaac done"
  isaac_cell $n mi --sim mujoco --render isaac
  say "$n mi done"
  isaac_cell $n im --sim isaac --render mujoco; say "$n im done"
done
say "smolvla queue done"
