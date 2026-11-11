#!/bin/bash
# V3 T1, Part 1: the tuning of the study file ("After Q1: T1", "Part 1 run settings"), T1a to T1f in order, one GPU job at a time.
# Choices only on MuJoCo seeds 900-999 (100 episodes), 8-point rule; the learning rate is tuned for nobody (T1f is a sensitivity check).
# Finished steps are skipped (a training run that did not finish restarts from the start: the trainers cannot resume), so the queue can be started again.
# Stops (and says so in the log) on an error, on more than 100 h since the first start, and in T1f when a point differs from 98 by 8 points or more.
export PATH="/usr/bin:/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; T=$STUDY/t1; LOG=$STUDY/logs/t1_queue.log; SO=research/configs/so101_randomized_pick_place.yaml
NOM200=data/study01/mujoco_nominal_200_seeds0-199
NOM400="$NOM200 data/study01/mujoco_nominal_200_seeds200-399"
RL=data/study01/mujoco_randomized-look_400_seeds0-399
TR=research/imitation_learning
mkdir -p $T
say() { echo "$* $(date)" >> $LOG; }
set -e
trap 'say "T1 STOPPED by an error at line $LINENO (see the logs in $T)"' ERR
[ -f $T/start_epoch ] || date +%s > $T/start_epoch
say "t1 queue (re)start"

budget() { # more than 100 h since the first start stops the work
  h=$(( ($(date +%s) - $(cat $T/start_epoch)) / 3600 ))
  if [ $h -gt 100 ]; then say "T1 STOPPED: $h h since the first start, more than the 100 h of the study file"; exit 0; fi
}
train() { # NAME COMMAND...   (the command gets --out $T/NAME appended)
  n=$1; shift; out=$T/$n
  [ -f $out/DONE ] && return 0
  budget
  say "train $n start"
  "$@" --out $out > $T/train_$n.log 2>&1
  touch $out/DONE
  say "train $n end: $(uv run python -c "import json;d=json.load(open('$out/training_log.json'))[-1];print('step',d['step'],'loss',round(d['loss'],4),'elapsed_h',round(d['elapsed_s']/3600,2))")"
}
score() { # NAME CHECKPOINT [eval_policy args...]: 100 validation episodes, MuJoCo, seeds 900-999
  n=$1; ck=$2; shift 2; j=$T/val_$n.json
  [ -f $j ] || uv run python scripts/eval_policy.py --manifest $SO --policy lerobot --checkpoint $ck --seed 900 --episodes 100 "$@" --json $j > $T/val_$n.log 2>&1
  say "val $n: $(uv run python -c "import json;print(sum(r['success'] for r in json.load(open('$j'))['results']))")/100"
}
val() { uv run python -c "import json;print(sum(r['success'] for r in json.load(open('$T/val_$1.json'))['results']))"; }

SMOL="uv run python $TR/train_smolvla.py --steps 40000 --batch-size 64 --accum 2 --model-image-size 256 --seed 1 --log-every 100 --save-every 10000 --keep-every 10000"

# T1a, T1b: SmolVLA 40k steps, nominal and randomized look (400 demonstrations), four checkpoints each
train t1a_smolvla_nominal_40k $SMOL --dataset $NOM400
for k in 010000 020000 030000 040000; do score t1a_$k $T/t1a_smolvla_nominal_40k/step_$k; done
train t1b_smolvla_randlook_40k $SMOL --dataset $RL
for k in 010000 020000 030000 040000; do score t1b_$k $T/t1b_smolvla_randlook_40k/step_$k; done

# T1c: actions per chunk (50 current, 25, 10) on the nominal checkpoint that stands after T1a: the best of the four if it has 99 or more (91 + 8), else O1's nominal_seed1
read best bscore < <(uv run python -c "
import json
s=[(sum(r['success'] for r in json.load(open('$T/val_t1a_%s.json'%k))['results']),k) for k in ('010000','020000','030000','040000')]
top=max(v for v,_ in s); print(next(k for v,k in s if v==top), top)")
if [ "$bscore" -ge 99 ]; then CK=$T/t1a_smolvla_nominal_40k/step_$best; REF=$bscore; else CK=$STUDY/smolvla_nominal_seed1; REF=91; fi
say "t1c checkpoint: $CK (T1a best: step_$best $bscore; reference score at 50 actions per chunk: $REF)"
score t1c_n25 $CK --policy-arg n_action_steps=25
score t1c_n10 $CK --policy-arg n_action_steps=10

# T1d, T1e: Diffusion Policy (89.2M, batch 16), 16 of 64 actions played as in V2; only the final checkpoint is scored
DP="uv run python $TR/train_diffusion.py --lr 1e-4 --num-workers 0 --seed 1 --save-every 10000 --keep-every 10000"
train t1d_dp_nominal200_200k $DP --dataset $NOM200 --steps 200000
score t1d_final $T/t1d_dp_nominal200_200k --policy-arg n_action_steps=16
train t1e_dp_nominal400_100k $DP --dataset $NOM400 --steps 100000
score t1e_final $T/t1e_dp_nominal400_100k --policy-arg n_action_steps=16

# T1f: ACT learning-rate sensitivity (200 demonstrations, chunk 100, 60k steps; reference 98); a point 8 or more away from 98 stops the queue
ACT="uv run python $TR/train_act.py --dataset $NOM200 --steps 60000 --chunk-size 100 --num-workers 0 --seed 1 --save-every 10000 --keep-every 10000"
for pt in "x2 2e-5" "x0.5 5e-6"; do
  set -- $pt; n=t1f_act_lr_$1
  train $n $ACT --lr $2
  score $n $T/$n
  s=$(val $n); d=$(( s > 98 ? s - 98 : 98 - s ))
  if [ $d -ge 8 ]; then say "T1f STOP: learning rate $2 scores $s against 98 (difference $d, 8 or more): the numbers go to the user, nothing of ACT is re-run"; exit 0; fi
done
say "T1 part 1 done"
