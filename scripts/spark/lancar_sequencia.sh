#!/bin/bash
cd ~
setsid nohup bash ~/cerise2026/pose/src/repo/scripts/spark/sequencia.sh > ~/cerise2026/pose/logs/sequencia.log 2>&1 < /dev/null &
sleep 1
pgrep -af 'sequencia\.sh' | grep -v lancar
