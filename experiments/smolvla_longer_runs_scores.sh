#!/bin/bash
# Scores of the longer SmolVLA runs (tuning table, "SmolVLA with more steps"): no training. Four SmolVLA checkpoints saved by tuning_search.sh (30k and 40k steps) are scored in the cells of smolvla_ladder.sh:
# MuJoCo on seeds 1000-1299 (300 episodes), Isaac Sim (isaac), MI and IM on seeds 1000-1099 in shards of 10 seeds with --video failures. No Newton.
# Order: randomized look 30k, randomized look 40k, nominal 30k, nominal 40k; one Isaac Sim process at a time. Same flags as smolvla_ladder.sh.
# Stop rules: fewer than 50 of 100 episodes started in an Isaac Sim cell (isaac, mi, im); more than 40 h since the first start; a shard that still fails after its retries.
# Finished cells are skipped, so the queue can be started again. Start it with the full path of Git's bash.exe (plain "bash" in PowerShell is WSL).
export PATH="/usr/bin:/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; T=$STUDY/tuning; P=$T/longer_runs; SO=research/configs/so101_randomized_pick_place.yaml; LOG=$STUDY/logs/longer_runs_queue.log
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p $P
say() { echo "$* $(date)" >> $LOG; }
for tool in uv grep date python; do command -v $tool > /dev/null || { say "T1P2 no $tool, stopping before anything runs"; exit 1; }; done
[ -f $P/start_epoch ] || date +%s > $P/start_epoch
say "longer runs queue (re)start"

budget() { # more than 40 h since the first start stops the work
  h=$(( ($(date +%s) - $(cat $P/start_epoch)) / 3600 ))
  if [ $h -gt 40 ]; then say "longer runs STOPPED: $h h since the first start, more than the 40 h allowed"; exit 0; fi
}
started() { # DIR: number of episodes of the cell that started (first_move_step >= 0) / number of episodes
  python -c "
import glob, json
r = [e for f in glob.glob('$1/s*/shard-*.json') for e in json.load(open(f))['results']]
print(sum(e.get('first_move_step', -1) >= 0 for e in r), len(r))"
}
isaac_cell() { # NAME CKPT CELL FLAGS...   (shards of 10 seeds, 1000-1099; the shard script retries a render glitch in a fresh process)
  n=$1; ck=$2; c=$3; shift 3
  budget
  for k in 1000 1010 1020 1030 1040 1050 1060 1070 1080 1090; do
    d=$P/$n/$c/s$k; [ -f $d/merged.json ] && continue
    uv run python scripts/eval_policy_sharded.py --out $d --seed $k --episodes 10 --shard-size 10 -- --manifest $SO --policy lerobot --checkpoint $ck "$@" --video failures --camera front --out $d/videos > $P/$n/${c}_s$k.log 2>&1
    [ -f $d/merged.json ] || { say "T1P2 STOPPED: $n $c shard $k has no result after its retries (see $d/attempts.jsonl)"; exit 0; }
  done
  set -- $(started $P/$n/$c)
  say "$n $c: $1 of $2 episodes started"
  if [ "$1" -lt 50 ]; then say "T1P2 STOPPED (stop rule): fewer than 50 of 100 episodes started in $n $c"; exit 0; fi
}

for job in "randlook_30k t1b_smolvla_randlook_40k step_030000" "randlook_40k t1b_smolvla_randlook_40k step_040000" "nominal_30k t1a_smolvla_nominal_40k step_030000" "nominal_40k t1a_smolvla_nominal_40k step_040000"; do
  set -- $job; n=$1; ck=$T/$2/$3
  mkdir -p $P/$n
  say "$n start ($ck)"
  budget
  [ -f $P/$n/mujoco.json ] || uv run python scripts/eval_policy.py --manifest $SO --policy lerobot --checkpoint $ck --seed 1000 --episodes 300 --json $P/$n/mujoco.json > $P/$n/mujoco.log 2>&1
  [ -f $P/$n/mujoco.json ] || { say "T1P2 STOPPED: $n mujoco has no result (see $P/$n/mujoco.log)"; exit 0; }
  say "$n mujoco: $(python -c "import json;print(sum(r['success'] for r in json.load(open('$P/$n/mujoco.json'))['results']))")/300"
  isaac_cell $n $ck isaac --sim isaac
  isaac_cell $n $ck mi --sim mujoco --render isaac
  isaac_cell $n $ck im --sim isaac --render mujoco
  say "$n done"
done
say "longer runs done"
