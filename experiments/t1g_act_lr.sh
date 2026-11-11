#!/bin/bash
# V3 T1g: the last ACT learning-rate run ("What remains" of the study file, decided with the user 2026-11-11).
# ACT, 200 nominal demonstrations, chunk 100, learning rate 5e-6, 120k steps (twice the 60k of T1f, whose loss at 5e-6 was still falling); seed 1.
# Validation only: MuJoCo seeds 900-999 (100 episodes). Bar written before the run: at least 90 = the lower learning rate was undertrained;
# below 90 = ACT is sensitive to the learning rate. Nothing else is run for ACT and no ACT bar or claim changes.
# Finished steps are skipped (the trainer cannot resume: a run that did not finish restarts from the start). Isaac Sim must not run on the GPU at the same time.
export PATH="/usr/bin:/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; T=$STUDY/t1; LOG=$STUDY/logs/t1_queue.log; SO=research/configs/so101_randomized_pick_place.yaml
NOM200=data/study01/mujoco_nominal_200_seeds0-199
TR=research/imitation_learning
say() { echo "$* $(date)" >> $LOG; }
set -e
trap 'say "T1g STOPPED by an error at line $LINENO (see the logs in $T)"' ERR
n=t1g_act_lr_5e-6_120k; out=$T/$n
say "t1g queue (re)start"
if [ ! -f $out/DONE ]; then
  say "train $n start"
  uv run python $TR/train_act.py --dataset $NOM200 --steps 120000 --chunk-size 100 --num-workers 0 --seed 1 --save-every 10000 --keep-every 10000 --lr 5e-6 --out $out > $T/train_$n.log 2>&1
  touch $out/DONE
  say "train $n end: $(uv run python -c "import json;d=json.load(open('$out/training_log.json'))[-1];print('step',d['step'],'loss',round(d['loss'],4),'elapsed_h',round(d['elapsed_s']/3600,2))")"
fi
j=$T/val_$n.json
[ -f $j ] || uv run python scripts/eval_policy.py --manifest $SO --policy lerobot --checkpoint $out --seed 900 --episodes 100 --json $j > $T/val_$n.log 2>&1
s=$(uv run python -c "import json;print(sum(r['success'] for r in json.load(open('$j'))['results']))")
say "val $n: $s/100 (bar: at least 90 = undertrained, below 90 = sensitive; T1f at 60k: 88)"
say "T1g done"
