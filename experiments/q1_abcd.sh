#!/bin/bash
# V3 Q1, groups A to D (Isaac Sim 6.2.0.0 re-score of the models whose Isaac Sim score in the paper is from 6.1.0.0): visual servo 1000-1299, the two chunk ACT runs, the four ACT trained on replays on Isaac Sim,
# the two ACT trained on both renderers, 1000-1099 each (1,100 episodes). Same harness as O1 (scripts/eval_policy_sharded.py, shards of 10 seeds, a shard with the start glitch is repeated whole by the sharded runner),
# videos of failures only, local checkpoints (sha256 already checked against Hugging Face). One Isaac Sim process at a time. Finished shards are skipped, so a restart resumes.
# Start it with the full path of Git's bash.exe (plain "bash" in PowerShell is WSL, which has no uv).
export PATH="/usr/bin:/bin:$HOME/.local/bin:$PATH"
LOG=$(dirname "$0")/../../outputs/study01/logs/q1_queue.log
for tool in uv grep date python; do command -v $tool > /dev/null || { echo "q1_abcd.sh: no $tool, stopping before anything runs" >> $LOG; exit 1; }; done
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; Q=$STUDY/q1; SO=research/configs/so101_randomized_pick_place.yaml
CK=checkpoints/rizgiak/physai-study-01-checkpoints
export OMNI_KIT_ACCEPT_EULA=YES
say() { echo "$* $(date)" >> $LOG; }
cell() { # GROUP NAME FIRST_SEED LAST_SEED POLICY_ARGS...   (shards of 10 seeds)
  g=$1; nm=$2; a=$3; b=$4; shift 4
  mkdir -p $Q/$g/$nm
  for k in $(seq $a 10 $b); do
    d=$Q/$g/$nm/s$k; [ -f $d/merged.json ] && continue
    uv run python scripts/eval_policy_sharded.py --out $d --seed $k --episodes 10 --shard-size 10 -- --manifest $SO --sim isaac "$@" --video failures --camera front --out $d/videos > $STUDY/logs/q1_${g}_${nm}_s$k.log 2>&1
  done
  say "$g $nm $a-$b done"
}
say "Q1 A-D start"
cell A_visual_servo visual_servo 1000 1290 --policy visual_servo
for n in act_chunk30_100demos_unseeded act_chunk100_100demos_unseeded; do cell B_chunk $n 1000 1090 --policy lerobot --checkpoint $CK/$n; done
for n in act_isaac_only_100_unseeded act_isaac_only_200_unseeded act_isaac_only_200_seed1 act_isaac_only_400_unseeded; do cell C_isaac_only $n 1000 1090 --policy lerobot --checkpoint $CK/$n; done
for n in act_both_renderers_unseeded act_both_renderers_seed1; do cell D_both_renderers $n 1000 1090 --policy lerobot --checkpoint $CK/$n; done
say "Q1 A-D done"
