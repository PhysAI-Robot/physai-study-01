#!/bin/bash
# The ViperX repeats the MI and IM cells (cells table). usage: viperx_cells.sh CELL   (CELL = mi: --sim mujoco --render isaac | im: --sim isaac --render mujoco)
# Run it after cell_isaac_physics_mujoco_pictures.sh (one Isaac Sim process at a time). It runs the gate of the cell once (first frames of seeds 1000-1004,
# no policy), then scores the two nominal ViperX models on seeds 1000-1299. Finished shards are skipped.
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; C=$1; N=$STUDY/viperx_cells
M=research/configs/viperx300s_single_cube_randomized_place.yaml
export GATE_MANIFEST=$M GATE_NO_SILHOUETTE=1 OMNI_KIT_ACCEPT_EULA=YES
case $C in mi) FLAGS="--sim mujoco --render isaac"; GC=cell;; im) FLAGS="--sim isaac --render mujoco"; GC=im;; *) echo "cell mi|im"; exit 1;; esac
mkdir -p $N
dump() { # CELL FILE: Isaac exits 0 without writing the file after its start-up glitch, so test for the file
  for i in 1 2 3 4 5 6; do
    [ -f $2 ] && return
    uv run python research/experiments/cell_first_frame_gate.py dump $1 $2 > $STUDY/logs/viperx_cells_gate_$1.log 2>&1
    [ -f $2 ] || sleep 30
  done
}
dump $GC $N/gate_$GC.npz
[ $C = mi ] && dump isaac_own $N/gate_isaac_own.npz
if [ $C = mi ]; then uv run python research/experiments/cell_first_frame_gate.py check $N/gate_isaac_own.npz $N/gate_cell.npz $N/gate_mujoco.npz > $N/gate_mi.txt 2>&1
else uv run python research/experiments/cell_first_frame_gate.py check_im $N/gate_im.npz $N/gate_mujoco.npz > $N/gate_im.txt 2>&1; fi
for m in nominal_seed1 nominal_seed2; do
  for k in $(seq 1000 10 1290); do
    d=$N/$C/$m/s$k; [ -f $d/merged.json ] && continue
    uv run python scripts/eval_policy_sharded.py --out $d --seed $k --episodes 10 --shard-size 10 -- --manifest $M $FLAGS --policy lerobot --checkpoint $STUDY/viperx/act_$m --video failures --camera front --out $d/videos > $STUDY/logs/viperx_cells_${C}_${m}_s$k.log 2>&1
  done
  echo "$C $m $(date)" >> $STUDY/logs/viperx_cells_queue.log
done
echo "viperx_cells $C queue done $(date)" >> $STUDY/logs/viperx_cells_queue.log
