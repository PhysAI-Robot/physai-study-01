#!/bin/bash
# Gate of the cube-start cells: first frames of seeds 1000-1004, MuJoCo with the cube start of the SO-101 nominal_seed1 table, alone and with Isaac Sim's pictures.
# Only seed 1000 is history-free (first episode of a fresh process), so only it is compared with Isaac Sim's own run (the gate file of cell_mujoco_physics_isaac_pictures.sh).
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; export OMNI_KIT_ACCEPT_EULA=YES GATE_CUBE_START=research/experiments/cube_start_so101_nominal_seed1.json
uv run python research/experiments/cell_first_frame_gate.py dump mujoco $STUDY/gate/gate_mujoco_start.npz > $STUDY/logs/cube_start_gate_mujoco.log 2>&1
for i in 1 2 3 4 5 6; do
  [ -f $STUDY/gate/gate_cell_start.npz ] && break
  uv run python research/experiments/cell_first_frame_gate.py dump cell $STUDY/gate/gate_cell_start.npz > $STUDY/logs/cube_start_gate_cell.log 2>&1
  [ -f $STUDY/gate/gate_cell_start.npz ] || sleep 30
done
uv run python research/experiments/cell_first_frame_gate.py check $STUDY/mi_cell/gate_isaac_own.npz $STUDY/gate/gate_cell_start.npz $STUDY/gate/gate_mujoco_start.npz > $STUDY/gate/gate_check.txt 2>&1
echo "cube_start gate done $(date)" >> $STUDY/logs/cube_start_queue.log
