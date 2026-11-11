#!/bin/bash
# Rebuild the Isaac-frame probes of the paper on Isaac Sim 6.2.0.0 (they used frames captured on 6.1.0.0):
#   1  Isaac first frames, seeds 900-1099            -> data/study01/probe_first_frames/isaac   (MuJoCo frames are reused from v1)
#   2  first-read swap, nominal ACT on MuJoCo, 900-999 and 1000-1099 (image_edit=first_all, FirstFrameSwap)
#   3  encoder-distance AUC of the first frames against the failures on 6.2.0.0 (q1/F_validation + v2 bridge 1000-1099)
#   4  read probe (Table "reads"): Isaac render of the saved MuJoCo states, then compare
# Not run yet. The commands follow capture_isaac.sh / run_e1.sh of v1 and the docstrings of feature_shift.py and read_probe.py with the current paths;
# if one step fails because a path or flag changed, the script stops (set -e) and nothing is patched on the way.
# Start it with the full path of Git's bash.exe (plain "bash" in PowerShell is WSL, which has no uv). Run it after q1_eh.sh has finished (Isaac Sim alone on the GPU).
set -e
export PATH="/usr/bin:/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; P=$STUDY/probes62; mkdir -p $P $STUDY/logs
SO=research/configs/so101_randomized_pick_place.yaml
CK=checkpoints/rizgiak/physai-study-01-checkpoints/act_nominal_unseeded
FF=data/study01/probe_first_frames
export OMNI_KIT_ACCEPT_EULA=YES
say() { echo "$* $(date)" >> $STUDY/logs/p_probes62.log; }

# 1  Isaac first frames on 6.2 (retry like capture_isaac.sh: Isaac sometimes starts without drawing the robot)
mkdir -p $FF
[ -d $FF/mujoco ] || cp -r data/study01/probe_first_frames/mujoco $FF/mujoco
for i in 1 2 3 4 5; do
  [ -f $FF/isaac/meta.json ] && break
  rm -rf $FF/isaac
  uv run python scripts/eval_policy.py --manifest $SO --sim isaac --policy constant --seed 900 --episodes 200 --max-steps 2 --save-dataset $FF/isaac > $P/first_isaac.log 2>&1 || true
done
[ -f $FF/isaac/meta.json ] || { say "probes62 STOPPED: no Isaac first frames"; exit 1; }
say "1 first frames done"

# 2  first-read swap (the policy runs on MuJoCo)
uv run python scripts/eval_policy.py --manifest $SO --sim mujoco --policy lerobot --checkpoint $CK --seed 900 --episodes 100 \
  --policy-arg image_edit=first_all --policy-arg first_frames=$FF --policy-arg first_seed=900 --json $P/swap_val.json > $P/swap_val.log 2>&1
uv run python scripts/eval_policy.py --manifest $SO --sim mujoco --policy lerobot --checkpoint $CK --seed 1000 --episodes 100 \
  --policy-arg image_edit=first_all --policy-arg first_frames=$FF --policy-arg first_seed=1000 --json $P/swap_test.json > $P/swap_test.log 2>&1
say "2 swap done"

# 3  AUC (failures of the policy on Isaac Sim 6.2.0.0: validation 900-999 from q1/F_validation, test 1000-1099 from the v2 bridge)
uv run python research/imitation_learning/feature_shift.py --checkpoint $CK --first-frames $FF \
  --isaac-results $STUDY/q1/F_validation/act_nominal_unseeded/s*/merged.json outputs/study01/isaac_n2/bridge_act_nominal_unseeded/s*/merged.json \
  --out $P/feature_shift.json > $P/feature_shift.log 2>&1
say "3 AUC done"

# 4  read probe
python research/imitation_learning/read_probe.py render --dataset data/study01/probe_act_reads_mujoco/act200_mujoco --out $P/reads_isaac.npz > $P/reads_render.log 2>&1
python research/imitation_learning/read_probe.py compare --reads $P/reads_isaac.npz --checkpoint $CK --table $P/reads_table.md > $P/reads_compare.log 2>&1
say "probes62 all done"
