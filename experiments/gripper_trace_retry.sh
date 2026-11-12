#!/bin/bash
# The Isaac Sim gripper trace (gripper_trace.py) with up to 10 starts 60 s apart (cameras front and wrist), because Isaac Sim sometimes exits at start-up without drawing the robot.
export PATH="/usr/bin:/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; P=$STUDY/probes
export OMNI_KIT_ACCEPT_EULA=YES
for i in 1 2 3 4 5 6 7 8 9 10; do
  [ -f $P/gripper_trace_isaac.json ] && break
  uv run python research/experiments/gripper_trace.py isaac $P/gripper_trace_isaac.json > $P/gripper_trace_isaac_retry$i.log 2>&1; [ -f $P/gripper_trace_isaac.json ] || sleep 60
done
if [ -f $P/gripper_trace_isaac.json ]; then echo "probes gripper trace done (retry $i) $(date)" >> $STUDY/logs/probes.log
else echo "probes STOPPED: no Isaac gripper trace after 10 more starts $(date)" >> $STUDY/logs/probes.log; fi
