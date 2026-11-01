#!/bin/bash
# V2 N12 pilot: one worker. usage: n12_run.sh WORKER NWORKERS [EPISODES]. Each model of n12_models.json (index mod NWORKERS == WORKER) is scored on MuJoCo,
# seeds 1000-1099, unedited and under the edits fixed in the study file; finished runs are skipped.
cd "$(dirname "$0")/../../.."
V2=outputs/study01/v2; W=$1; N=$2; EP=${3:-100}; OUT=${N12_OUT:-$V2/n12}; LIST=${N12_MODELS:-$V2/n12_models.json}
SO=research/configs/so101_randomized_pick_place.yaml; PLAIN=research/configs/so101_randomized_pick_place_plainfloor.yaml; VX=research/configs/viperx300s_single_cube_randomized_place.yaml
mkdir -p $OUT/logs
python -c "
import json, sys
sys.stdout.reconfigure(newline='\n')
for i, m in enumerate(json.load(open('$LIST'))):
    if i % $N == $W: print(m['name'], m['robot'], m['checkpoint'])
" > $OUT/logs/models_$W.txt
while read name robot ck; do
  M=$SO; [ $robot = viperx ] && M=$VX
  for edit in none blur light05 light14 cam10 arm_tint bg_dim plain; do
    [ $robot = viperx ] && case $edit in arm_tint|bg_dim|plain) continue;; esac
    o=$OUT/$name.$edit.json; [ -f $o ] && continue
    M2=$M; extra=""
    case $edit in
      blur|arm_tint|bg_dim) extra="--policy-arg image_edit=$edit";;
      light05) extra="--lighting-scale 0.5 --nominal-physics";; light14) extra="--lighting-scale 1.4 --nominal-physics";;
      cam10) extra="--camera-jitter 0.01 --camera-shift-unknown --nominal-physics";; plain) M2=$PLAIN;;
    esac
    uv run python scripts/eval_policy.py --manifest $M2 --policy lerobot --checkpoint $ck --seed 1000 --episodes $EP --json $o $extra > $OUT/logs/$name.$edit.log 2>&1
    echo "$name $edit: $(grep -h 'policy=lerobot' $OUT/logs/$name.$edit.log | cut -c1-60) $(date)" >> $OUT/progress_$W.log
  done
done < $OUT/logs/models_$W.txt
echo "worker $W done $(date)" >> $OUT/progress_$W.log
