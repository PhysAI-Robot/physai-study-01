#!/bin/bash
# ACT trained on MuJoCo images that show the SO-101's contact pads (act_pad_visible_seed1), scored in four renderings (pads table, Section "attribution"); exploratory
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; M=research/configs/so101_randomized_pick_place.yaml; HF=hf:rizgiak/physai-study-01-checkpoints
PAD="uv run python research/scripts/with_pad_visible.py"
D=data/study01/mujoco_nominal_padvisible_200_seeds0-199
mkdir -p $STUDY/pad_visible
[ -f $D/meta.json ] || $PAD research/scripts/collect_demos.py --manifest $M --episodes 200 --seed 0 --dataset $D > $STUDY/logs/pad_visible_collect.log 2>&1
echo "pad_visible demos: $(tail -n 2 $STUDY/logs/pad_visible_collect.log | tr '\n' ' ' | cut -c1-110)" >> $STUDY/logs/pad_visible_queue.log
CK=$STUDY/pad_visible/act_padvisible_200_60k_seed1
[ -f $STUDY/logs/pad_visible_train.done ] || { uv run python research/imitation_learning/train_act.py --dataset $D --episodes 200 --steps 60000 --chunk-size 100 --num-workers 0 --seed 1 --out $CK > $STUDY/logs/pad_visible_train.out.log 2> $STUDY/logs/pad_visible_train.err.log && touch $STUDY/logs/pad_visible_train.done; echo "pad_visible trained $(date)" >> $STUDY/logs/pad_visible_queue.log; }
ev() { # name cmd-prefix... : 300 seeds
  name=$1; shift
  [ -f $STUDY/pad_visible/$name.json ] || "$@" scripts/eval_policy.py --manifest $M --policy lerobot --checkpoint $CK --seed 1000 --episodes 300 --json $STUDY/pad_visible/$name.json > $STUDY/logs/pad_visible_$name.log 2>&1
  echo "$name: $(grep 'policy=lerobot' $STUDY/logs/pad_visible_$name.log | cut -c1-60) $(date)" >> $STUDY/logs/pad_visible_queue.log
}
# MuJoCo without the pad, MuJoCo with it
[ -f $STUDY/pad_visible/mujoco_nopad.json ] || uv run python scripts/eval_policy.py --manifest $M --policy lerobot --checkpoint $CK --seed 1000 --episodes 300 --json $STUDY/pad_visible/mujoco_nopad.json > $STUDY/logs/pad_visible_mujoco_nopad.log 2>&1
echo "mujoco_nopad: $(grep 'policy=lerobot' $STUDY/logs/pad_visible_mujoco_nopad.log | cut -c1-60)" >> $STUDY/logs/pad_visible_queue.log
[ -f $STUDY/pad_visible/mujoco_pad.json ] || $PAD scripts/eval_policy.py --manifest $M --policy lerobot --checkpoint $CK --seed 1000 --episodes 300 --json $STUDY/pad_visible/mujoco_pad.json > $STUDY/logs/pad_visible_mujoco_pad.log 2>&1
echo "mujoco_pad: $(grep 'policy=lerobot' $STUDY/logs/pad_visible_mujoco_pad.log | cut -c1-60)" >> $STUDY/logs/pad_visible_queue.log
# Newton with the pad (the renderer before the fix), Newton without it (the fixed one)
[ -f $STUDY/pad_visible/newton_pad.json ] || uv run python research/experiments/newton_renderer_variant.py --variant old -- --manifest $M --sim mujoco --render newton --policy lerobot --checkpoint $CK --seed 1000 --episodes 300 --json $STUDY/pad_visible/newton_pad.json > $STUDY/logs/pad_visible_newton_pad.log 2>&1
echo "newton_pad: $(grep 'policy=lerobot' $STUDY/logs/pad_visible_newton_pad.log | cut -c1-60)" >> $STUDY/logs/pad_visible_queue.log
[ -f $STUDY/pad_visible/newton_nopad.json ] || uv run python scripts/eval_policy.py --manifest $M --sim mujoco --render newton --policy lerobot --checkpoint $CK --seed 1000 --episodes 300 --json $STUDY/pad_visible/newton_nopad.json > $STUDY/logs/pad_visible_newton_nopad.log 2>&1
echo "newton_nopad: $(grep 'policy=lerobot' $STUDY/logs/pad_visible_newton_nopad.log | cut -c1-60)" >> $STUDY/logs/pad_visible_queue.log
# the nominal model act_nominal_seed1 with the pad drawn
[ -f $STUDY/pad_visible/old_model_mujoco_pad.json ] || $PAD scripts/eval_policy.py --manifest $M --policy lerobot --checkpoint $HF/act_nominal_seed1 --seed 1000 --episodes 300 --json $STUDY/pad_visible/old_model_mujoco_pad.json > $STUDY/logs/pad_visible_nominal_seed1_pad.log 2>&1
echo "act_nominal_seed1 with the pad: $(grep 'policy=lerobot' $STUDY/logs/pad_visible_nominal_seed1_pad.log | cut -c1-60)" >> $STUDY/logs/pad_visible_queue.log
echo "pad_visible queue done $(date)" >> $STUDY/logs/pad_visible_queue.log
