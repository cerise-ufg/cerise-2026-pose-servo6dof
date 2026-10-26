#!/bin/bash
# lanca a fase 2 destacada
cd ~
setsid nohup bash ~/cerise2026/pose/src/repo/scripts/spark/final2.sh > ~/cerise2026/pose/logs/final2.log 2>&1 < /dev/null &
sleep 1
pgrep -af "bash .*final2\.sh" | grep -v lancar
