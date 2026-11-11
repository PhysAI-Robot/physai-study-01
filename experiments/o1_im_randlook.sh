#!/bin/bash
# V3 O1, IM cell of the three randomized-look runs (2026-11-04): no bar of O1 uses it, the full factorial table of the paper does (V2 scored it for the ACT randomized-look run too).
# Waits for o1_tail.sh (PID 26268), which waits for o1_queue2.sh, so Isaac Sim never runs twice.
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
while tasklist //FI "PID eq 26268" 2>/dev/null | grep -q bash.exe; do sleep 60; done
for m in randlook_seed1 randlook_seed2 randlook_seed3; do
  isaac_cell $m im --sim isaac --render mujoco; say "$m im done"
done
say "o1 randlook im done"
