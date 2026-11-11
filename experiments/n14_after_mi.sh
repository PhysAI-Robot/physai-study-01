#!/bin/bash
cd "$(dirname "$0")/../.."
while ! grep -q 'n14 mi queue done' outputs/study01/logs/n14_queue.log 2>/dev/null; do sleep 120; done
bash research/experiments/n14_queue.sh im

