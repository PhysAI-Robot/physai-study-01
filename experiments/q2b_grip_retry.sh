#!/bin/bash
# Q2b: the Isaac gripper trace again, up to 10 starts 60 s apart (cameras front+wrist) (Isaac start glitch); code unchanged.
export PATH="/usr/bin:/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."
STUDY=outputs/study01; P=$STUDY/probes62
export OMNI_KIT_ACCEPT_EULA=YES
for i in 1 2 3 4 5 6 7 8 9 10; do
  [ -f $P/q2b_grip_isaac.json ] && break
  uv run python research/experiments/q2b_grip.py isaac $P/q2b_grip_isaac.json > $P/q2b_grip_isaac_retry$i.log 2>&1; [ -f $P/q2b_grip_isaac.json ] || sleep 60
done
if [ -f $P/q2b_grip_isaac.json ]; then echo "Q2b gripper trace done (retry $i) $(date)" >> $STUDY/logs/p_probes62.log
else echo "Q2b STOPPED: no Isaac gripper trace after 10 more starts $(date)" >> $STUDY/logs/p_probes62.log; fi
