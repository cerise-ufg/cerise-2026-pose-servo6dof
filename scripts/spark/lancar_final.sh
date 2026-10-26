#!/bin/bash
# lanca o orquestrador final destacado (EPOCAS e TMAX configuraveis)
export EPOCAS=${EPOCAS:-20} TMAX=${TMAX:-70}
cd ~
setsid nohup bash ~/cerise2026/pose/src/repo/scripts/spark/final.sh > ~/cerise2026/pose/logs/final.log 2>&1 < /dev/null &
sleep 1
pgrep -af "bash .*final\.sh" | grep -v lancar
