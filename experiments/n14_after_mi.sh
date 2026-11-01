#!/bin/bash
cd "$(dirname "$0")/../../.."
while ! grep -q 'n14 mi queue done' outputs/study01/v2/logs/n14_queue.log 2>/dev/null; do sleep 120; done
bash outputs/study01/v2/n14_queue.sh im

