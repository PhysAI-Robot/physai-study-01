#!/bin/bash
# The ACT and Diffusion Policy runs scored on Newton (MuJoCo physics, Newton renderer) with the fixed renderer, seeds 1000-1299
cd "$(dirname "$0")/../.."
STUDY=outputs/study01
M=research/configs/so101_randomized_pick_place.yaml
HF=hf:rizgiak/physai-study-01-checkpoints
run() { # name checkpoint [extra args]
  name=$1; ck=$2; shift 2
  [ -f $STUDY/newton_after_fix/$name.json ] && return
  uv run python scripts/eval_policy.py --manifest $M --sim mujoco --render newton --policy lerobot --checkpoint $ck "$@" --seed 1000 --episodes 300 --json $STUDY/newton_after_fix/$name.json > $STUDY/logs/rescore_$name.log 2>&1
  echo "$name: $(grep 'policy=lerobot' $STUDY/logs/rescore_$name.log | cut -c1-70) $(date)" >> $STUDY/logs/newton_after_fix.log
}
for s in 1 2 3; do run nominal400_seed$s $HF/act_nominal400_seed$s; done
for n in act_randomized_look_seed0 act_randomized_look_seed1 act_randomized_look_seed2 act_nominal_unseeded act_nominal_seed1 act_nominal_seed2 act_nominal_seed3 act_both_renderers_unseeded act_both_renderers_seed1 act_isaac_only_100_unseeded act_isaac_only_200_unseeded act_isaac_only_200_seed1 act_isaac_only_400_unseeded act_chunk30_100demos_unseeded act_chunk100_100demos_unseeded; do run $n $HF/$n; done
for n in diffusion_policy_v2_nominal_seed0 diffusion_policy_v2_randomized_look_seed0; do run $n $HF/$n --policy-arg n_action_steps=16; done
echo "newton rescore done $(date)" >> $STUDY/logs/newton_after_fix.log
