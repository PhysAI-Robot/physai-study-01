#!/bin/bash
# Isaac Sim scores of the control runs, groups E to H (the folder names under outputs/study01/isaac_controls/). Run it after isaac_scores_controls_a.sh has finished,
# so that Isaac Sim never runs next to another job. Shards of 10 seeds, failure videos only, local copies of the checkpoints:
#   E  replay of the expert's actions (seeds 0-49), replay of the MuJoCo actions of act_nominal_unseeded (seeds 900-999; the dataset is the MuJoCo rollouts of act_nominal_unseeded in data/study01/probe_act_reads_mujoco/act200_mujoco, see read_probe.py)
#   F  validation seeds 900-999 of act_nominal_unseeded (base) and of the jitter model
#   G  act_nominal_unseeded with 50 and 25 actions per chunk, 1000-1099
#   H  diffusion_policy_randomized_look_seed0 (16 actions per chunk as before), 1000-1099
# If a shard does not produce its merged.json, the queue STOPS and logs it; nothing is fixed on the way.
# Start it with the full path of Git's bash.exe on Windows (plain "bash" in PowerShell is WSL, which has no uv). Feeds the replay controls, the validation of the base ACT and the chunk table.
export PATH="/usr/bin:/bin:$HOME/.local/bin:$PATH"
LOG=$(dirname "$0")/../../outputs/study01/logs/isaac_controls_queue.log
for tool in uv grep date python; do command -v $tool > /dev/null || { echo "isaac_scores_controls_b.sh: no $tool, stopping before anything runs" >> $LOG; exit 1; }; done
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; Q=$STUDY/isaac_controls; SO=research/configs/so101_randomized_pick_place.yaml
CK=checkpoints/rizgiak/physai-study-01-checkpoints
export OMNI_KIT_ACCEPT_EULA=YES
say() { echo "$* $(date)" >> $LOG; }
cell() { # GROUP NAME FIRST_SEED LAST_SEED POLICY_ARGS...   (shards of 10 seeds)
  g=$1; nm=$2; a=$3; b=$4; shift 4
  mkdir -p $Q/$g/$nm
  for k in $(seq $a 10 $b); do
    d=$Q/$g/$nm/s$k; [ -f $d/merged.json ] && continue
    uv run python scripts/eval_policy_sharded.py --out $d --seed $k --episodes 10 --shard-size 10 -- --manifest $SO --sim isaac "$@" --video failures --camera front --out $d/videos > $STUDY/logs/isaac_controls_${g}_${nm}_s$k.log 2>&1
    [ -f $d/merged.json ] || { say "controls E-H STOPPED: $g $nm shard $k produced no merged.json (see logs/isaac_controls_${g}_${nm}_s$k.log)"; exit 1; }
  done
  say "$g $nm $a-$b done"
}
say "controls E-H start"
cell E_replay expert_replay 0 40 --policy replay --dataset data/study01/mujoco_nominal_200_seeds0-199
cell E_replay act_nominal_unseeded_replay 900 990 --policy replay --dataset data/study01/probe_act_reads_mujoco/act200_mujoco
cell F_validation act_nominal_unseeded 900 990 --policy lerobot --checkpoint $CK/act_nominal_unseeded
cell F_validation act_nominal_color_jitter_unseeded 900 990 --policy lerobot --checkpoint $CK/act_nominal_color_jitter_unseeded
cell G_chunk act_nominal_unseeded_n50 1000 1090 --policy lerobot --checkpoint $CK/act_nominal_unseeded --policy-arg n_action_steps=50
cell G_chunk act_nominal_unseeded_n25 1000 1090 --policy lerobot --checkpoint $CK/act_nominal_unseeded --policy-arg n_action_steps=25
cell H_diffusion diffusion_policy_randomized_look_seed0 1000 1090 --policy lerobot --checkpoint $CK/diffusion_policy_randomized_look_seed0 --policy-arg n_action_steps=16
say "controls E-H done"
