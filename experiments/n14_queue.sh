#!/bin/bash
# V2 N14: the ViperX repeats the cells. usage: n14_queue.sh CELL   (CELL = mi: --sim mujoco --render isaac | im: --sim isaac --render mujoco)
# Waits for the N13 queue, runs the gate of the cell once (first frames of seeds 1000-1004, no policy), then scores the two nominal ViperX
# models of N4 on seeds 1000-1299. One Isaac process at a time; finished shards are skipped.
cd "$(dirname "$0")/../../.."
V2=outputs/study01/v2; C=$1; N=$V2/n14
M=research/configs/viperx300s_single_cube_randomized_place.yaml
export GATE_MANIFEST=$M GATE_NO_SILHOUETTE=1 OMNI_KIT_ACCEPT_EULA=YES
case $C in mi) FLAGS="--sim mujoco --render isaac"; GC=cell;; im) FLAGS="--sim isaac --render mujoco"; GC=im;; *) echo "cell mi|im"; exit 1;; esac
while ! grep -q "n13 queue done" $V2/logs/n13_queue.log 2>/dev/null; do sleep 120; done
mkdir -p $N
dump() { # CELL FILE: Isaac exits 0 without writing the file after its start-up glitch, so test for the file
  for i in 1 2 3 4 5 6; do
    [ -f $2 ] && return
    uv run python $V2/n9_gate.py dump $1 $2 > $V2/logs/n14_gate_$1.log 2>&1
    [ -f $2 ] || sleep 30
  done
}
dump $GC $N/gate_$GC.npz
[ $C = mi ] && dump isaac_own $N/gate_isaac_own.npz
if [ $C = mi ]; then uv run python $V2/n9_gate.py check $N/gate_isaac_own.npz $N/gate_cell.npz $N/gate_mujoco.npz > $N/gate_mi.txt 2>&1
else uv run python $V2/n9_gate.py check_im $N/gate_im.npz $N/gate_mujoco.npz > $N/gate_im.txt 2>&1; fi
for m in nominal_seed1 nominal_seed2; do
  for k in $(seq 1000 10 1290); do
    d=$N/$C/$m/s$k; [ -f $d/merged.json ] && continue
    uv run python scripts/eval_policy_sharded.py --out $d --seed $k --episodes 10 --shard-size 10 -- --manifest $M $FLAGS --policy lerobot --checkpoint $V2/n4/act_$m --video failures --camera front --out $d/videos > $V2/logs/n14_${C}_${m}_s$k.log 2>&1
  done
  echo "$C $m $(date)" >> $V2/logs/n14_queue.log
done
echo "n14 $C queue done $(date)" >> $V2/logs/n14_queue.log
