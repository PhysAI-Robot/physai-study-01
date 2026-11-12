#!/bin/bash
# The eight Isaac Sim rendering settings (rendering-settings table, Section "attribution"): settings moved one at a time toward MuJoCo (isaac_render_variant.py), the nominal ACT runs act_nominal_seed1 and act_nominal_seed2
# on Isaac seeds 1000-1099 (the unchanged Isaac scores of the same stack are isaac_nominal/act_nominal_seed*). Order: floor_raw first, then the cumulative
# set, then the other single settings. One Isaac process at a time; every shard is skipped when its result exists.
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; M=research/configs/so101_randomized_pick_place.yaml; HF=hf:rizgiak/physai-study-01-checkpoints
mkdir -p $STUDY/isaac_render_settings
run() { # variant model
  for k in 1000 1010 1020 1030 1040 1050 1060 1070 1080 1090; do
    d=$STUDY/isaac_render_settings/$1/$2/s$k; [ -f $d/merged.json ] && continue
    ISAAC_VARIANT=$1 OMNI_KIT_ACCEPT_EULA=YES uv run python scripts/eval_policy_sharded.py --eval-script research/experiments/isaac_render_variant.py --out $d --seed $k --episodes 10 --shard-size 10 -- --manifest $M --sim isaac --policy lerobot --checkpoint $HF/$2 --video failures --camera front --out $d/videos > $STUDY/logs/isaac_render_settings_$1_$2_s$k.log 2>&1
  done
  echo "$1 $2 $(date)" >> $STUDY/logs/isaac_render_settings_queue.log
}
for v in floor_raw cumulative hard_shadows gi_off ao_off reflections_off spot_light sky_dome; do for m in act_nominal_seed1 act_nominal_seed2; do run $v $m; done; done
echo "isaac_render_settings queue done $(date)" >> $STUDY/logs/isaac_render_settings_queue.log
