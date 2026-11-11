#!/bin/bash
# V3 P1 gate: first frames of seeds 1000-1004, MuJoCo with the cube start of the SO-101 nominal_seed1 table, alone and with Isaac Sim's pictures.
# Only seed 1000 is history-free (first episode of a fresh process), so only it is compared with Isaac Sim's own run (N9's gate file).
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; export OMNI_KIT_ACCEPT_EULA=YES GATE_CUBE_START=research/experiments/cube_start_so101_nominal_seed1.json
uv run python research/experiments/n9_gate.py dump mujoco $STUDY/gate/gate_mujoco_start.npz > $STUDY/logs/p1_gate_mujoco.log 2>&1
for i in 1 2 3 4 5 6; do
  [ -f $STUDY/gate/gate_cell_start.npz ] && break
  uv run python research/experiments/n9_gate.py dump cell $STUDY/gate/gate_cell_start.npz > $STUDY/logs/p1_gate_cell.log 2>&1
  [ -f $STUDY/gate/gate_cell_start.npz ] || sleep 30
done
uv run python research/experiments/n9_gate.py check $STUDY/n9/gate_isaac_own.npz $STUDY/gate/gate_cell_start.npz $STUDY/gate/gate_mujoco_start.npz > $STUDY/gate/gate_check.txt 2>&1
echo "p1 gate done $(date)" >> $STUDY/logs/p1_queue.log
