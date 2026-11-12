#!/bin/bash
# First step of the cube-start measurement (Appendix, further results): the cube pose on Isaac Sim after reset, in two orders, both robots. Isaac Sim exits 0 without its file after the start-up glitch, so the script retries.
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; export OMNI_KIT_ACCEPT_EULA=YES
SO=research/configs/so101_randomized_pick_place.yaml; VX=research/configs/viperx300s_single_cube_randomized_place.yaml
run() { # NAME MANIFEST SEEDS ORDER
  o=research/experiments/cube_$1_$4.json
  for i in 1 2 3 4 5 6; do
    [ -f $o ] && return
    uv run python research/scripts/cube_start_table.py measure --manifest $2 --seeds $3 --order $4 --out $o > $STUDY/logs/cube_start_measure_$1_$4.log 2>&1
    [ -f $o ] || sleep 30
  done
}
for ord in forward reverse; do run so101 $SO 1000-1099 $ord; done
for ord in forward reverse; do run viperx $VX 1000-1299 $ord; done
echo "cube_start measure done $(date)" >> $STUDY/logs/cube_start_queue.log
