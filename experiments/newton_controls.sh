#!/bin/bash
# Two controls on Newton (after the fix), seeds 1000-1299, no training (Newton controls table):
# (1) the three tile-free randomized-look models (training seeds 0-2) on the held-out look (heldout_look);
# (2) the three models trained with floor tiles (training seeds 1-3) on the look outside every range of the tile-free runs (all_unseen).
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; M=research/configs/so101_randomized_pick_place.yaml; HF=hf:rizgiak/physai-study-01-checkpoints
mkdir -p $STUDY/newton_controls
run() { # name checkpoint look
  o=$STUDY/newton_controls/$1_$3
  [ -f $o.json ] || uv run python $STUDY/newton_look.py $3 -- --manifest $M --sim mujoco --render newton --policy lerobot --checkpoint $2 --seed 1000 --episodes 300 --json $o.json > $o.log 2>&1
  echo "$1 $3: $(grep 'policy=lerobot' $o.log | cut -c1-60) $(date)" >> $STUDY/logs/newton_controls_queue.log
}
for s in 0 1 2; do run randlook_seed$s $HF/act_randomized_look_seed$s heldout_look; done
for s in 1 2 3; do run randlook_bw_seed$s $STUDY/act_randlook_bw_120k_seed$s all_unseen; done
echo "newton_controls queue done $(date)" >> $STUDY/logs/newton_controls_queue.log
