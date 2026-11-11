#!/bin/bash
# Q2b step 1/2: SO-101 camera dumps and the gripper trace on the current stack (Isaac Sim 6.2.0.0), Isaac alone on the GPU.
# Stops (set -e) if a step gives no output; nothing is patched on the way. Log: outputs/study01/logs/p_probes62.log
set -e
export PATH="/usr/bin:/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; P=$STUDY/probes62; mkdir -p $P
export OMNI_KIT_ACCEPT_EULA=YES
say() { echo "$* $(date)" >> $STUDY/logs/p_probes62.log; }
uv pip list 2>/dev/null | grep -i "^isaacsim \|^mujoco " > $P/q2b_versions.txt || true

uv run python research/scripts/compare_cameras.py --dump mujoco --out $P/q2b_cams_mujoco.npz > $P/q2b_dump_mujoco.log 2>&1
[ -f $P/q2b_cams_mujoco.npz ] || { say "Q2b STOPPED: no MuJoCo dump"; exit 1; }
say "Q2b mujoco dump done"
# Isaac sometimes starts without drawing the robot (compare_cameras exits then): retry as in the other scripts
for i in 1 2 3 4 5; do
  [ -f $P/q2b_cams_isaac.npz ] && break
  uv run python research/scripts/compare_cameras.py --dump isaac --out $P/q2b_cams_isaac.npz > $P/q2b_dump_isaac_try$i.log 2>&1 || sleep 60
done
[ -f $P/q2b_cams_isaac.npz ] || { say "Q2b STOPPED: no Isaac dump"; exit 1; }
say "Q2b isaac dump done"
uv run python research/scripts/compare_cameras.py --compare $P/q2b_cams_mujoco.npz $P/q2b_cams_isaac.npz --montage $P/q2b_montage.png > $P/q2b_compare.log 2>&1
say "Q2b compare done"
uv run python research/experiments/q2b_grip.py mujoco $P/q2b_grip_mujoco.json > $P/q2b_grip_mujoco.log 2>&1
for i in 1 2 3; do
  [ -f $P/q2b_grip_isaac.json ] && break
  uv run python research/experiments/q2b_grip.py isaac $P/q2b_grip_isaac.json > $P/q2b_grip_isaac_try$i.log 2>&1 || sleep 60
done
[ -f $P/q2b_grip_mujoco.json ] && [ -f $P/q2b_grip_isaac.json ] || { say "Q2b STOPPED: no gripper trace"; exit 1; }
say "Q2b dumps done"
