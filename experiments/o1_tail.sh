#!/bin/bash
# V3 O1 tail (2026-11-03): scores randlook_seed1 (validation, MuJoCo, Newton, Isaac, MI). o1_queue2.sh skipped it: isaac_cell sets the global $n, so after the isaac_r2 / im_r2 repeat
# of nominal_seed1 the rest of the randlook_seed1 job ran on nominal_seed1 (every step already done) and the queue went on to nominal_seed2. This waits for o1_queue2.sh (PID 20416) to end.
# Start it with the full path of Git's bash.exe (plain "bash" in PowerShell is WSL). That bash has no /usr/bin on its PATH, and without grep a finished training run looks unfinished.
export PATH="/usr/bin:/bin:$HOME/.local/bin:$PATH"
for tool in uv grep date tasklist python; do command -v $tool > /dev/null || { echo "o1_queue2.sh: no $tool, stopping before anything runs" >> $(dirname "$0")/../../outputs/study01/logs/o1_queue.log; exit 1; }; done
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; SO=research/configs/so101_randomized_pick_place.yaml; LOG=$STUDY/logs/o1_queue.log
NOM="data/study01/mujoco_nominal_200_seeds0-199 data/study01/mujoco_nominal_200_seeds200-399"
RL="data/study01/mujoco_randomized-look_400_seeds0-399"
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p $STUDY/o1
say() { echo "$* $(date)" >> $LOG; }
json_run() { # NAME OUTFILE EPISODES SEED EXTRA...   (one process, all episodes)
  n=$1; o=$2; ep=$3; sd=$4; shift 4
  [ -f $o ] || uv run python scripts/eval_policy.py --manifest $SO --policy lerobot --checkpoint $STUDY/smolvla_$n --seed $sd --episodes $ep --json $o "$@" > ${o%.json}.log 2>&1
}
isaac_cell() { # NAME CELL FLAGS...   (shards of 10 seeds, 1000-1099)
  n=$1; c=$2; shift 2
  for k in 1000 1010 1020 1030 1040 1050 1060 1070 1080 1090; do
    d=$STUDY/o1/$n/$c/s$k; [ -f $d/merged.json ] && continue
    uv run python scripts/eval_policy_sharded.py --out $d --seed $k --episodes 10 --shard-size 10 -- --manifest $SO --policy lerobot --checkpoint $STUDY/smolvla_$n "$@" --video failures --camera front --out $d/videos > $STUDY/logs/o1_${n}_${c}_s$k.log 2>&1
  done
}
[ -x $HOME/.local/bin/uv.exe ] || { echo "no uv" >> $LOG; exit 1; }
while tasklist //FI "PID eq 20416" 2>/dev/null | grep -q bash.exe; do sleep 60; done  # o1_queue2.sh
first=0
for job in "randlook 1"; do
  set -- $job; look=$1; seed=$2; n=${look}_seed$seed; out=$STUDY/smolvla_$n; data=$NOM; [ $look = randlook ] && data=$RL
  mkdir -p $STUDY/o1/$n
  if ! grep -q '"step": 20000' $out/training_log.json 2>/dev/null; then
    say "train $n start"
    uv run python research/imitation_learning/train_smolvla.py --dataset $data --steps 20000 --batch-size 64 --accum 2 --model-image-size 256 --seed $seed --out $out --log-every 100 --save-every 2000 > $STUDY/logs/o1_train_$n.log 2>&1
  fi
  say "train $n end"
  json_run $n $STUDY/o1_val_$n.json 100 900
  if [ $first = 1 ]; then
    first=0
    started=$(python -c "import json;print(sum(e.get('first_move_step',-1)>=0 for e in json.load(open('$STUDY/o1_val_$n.json'))['results']))" 2>/dev/null || echo 0)
    say "first run $n: $started of 100 validation episodes started"
    if [ "$started" -lt 50 ]; then say "O1 STOPS (stop rule of the study file): fewer than half of the validation episodes started"; exit 0; fi
  fi
  json_run $n $STUDY/o1/$n/mujoco.json 300 1000
  json_run $n $STUDY/o1/$n/newton.json 300 1000 --sim mujoco --render newton
  say "$n mujoco and newton done"
  isaac_cell $n isaac --sim isaac
  say "$n isaac done"
  isaac_cell $n mi --sim mujoco --render isaac
  say "$n mi done"
  [ $look = nominal ] && { isaac_cell $n im --sim isaac --render mujoco; say "$n im done"; }
done
say "o1 tail done (randlook_seed1)"
