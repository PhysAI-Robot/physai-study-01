#!/bin/bash
# SO-101 camera dumps on MuJoCo and Isaac Sim (the input of image_statistics.py) and the gripper trace of both engines (gripper_trace.py),
# for the measured picture differences and the gripper reading of Section "attribution". Isaac Sim alone on the GPU.
# Stops (set -e) if a step gives no output; nothing is patched on the way. Log: outputs/study01/logs/probes.log
set -e
export PATH="/usr/bin:/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; P=$STUDY/probes; mkdir -p $P
export OMNI_KIT_ACCEPT_EULA=YES
say() { echo "$* $(date)" >> $STUDY/logs/probes.log; }
uv pip list 2>/dev/null | grep -i "^isaacsim \|^mujoco " > $P/versions.txt || true

uv run python research/scripts/compare_cameras.py --dump mujoco --out $P/cams_mujoco.npz > $P/camera_dump_mujoco.log 2>&1
[ -f $P/cams_mujoco.npz ] || { say "probes STOPPED: no MuJoCo dump"; exit 1; }
say "probes mujoco dump done"
# Isaac sometimes starts without drawing the robot (compare_cameras exits then): retry as in the other scripts
for i in 1 2 3 4 5; do
  [ -f $P/cams_isaac.npz ] && break
  uv run python research/scripts/compare_cameras.py --dump isaac --out $P/cams_isaac.npz > $P/camera_dump_isaac_try$i.log 2>&1 || sleep 60
done
[ -f $P/cams_isaac.npz ] || { say "probes STOPPED: no Isaac dump"; exit 1; }
say "probes isaac dump done"
uv run python research/scripts/compare_cameras.py --compare $P/cams_mujoco.npz $P/cams_isaac.npz --montage $P/montage.png > $P/camera_compare.log 2>&1
say "probes compare done"
uv run python research/experiments/gripper_trace.py mujoco $P/gripper_trace_mujoco.json > $P/gripper_trace_mujoco.log 2>&1
for i in 1 2 3; do
  [ -f $P/gripper_trace_isaac.json ] && break
  uv run python research/experiments/gripper_trace.py isaac $P/gripper_trace_isaac.json > $P/gripper_trace_isaac_try$i.log 2>&1 || sleep 60
done
[ -f $P/gripper_trace_mujoco.json ] && [ -f $P/gripper_trace_isaac.json ] || { say "probes STOPPED: no gripper trace"; exit 1; }
say "probes dumps done"
