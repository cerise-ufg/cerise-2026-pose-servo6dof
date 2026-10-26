#!/bin/bash
# estado geral do Spark (somente leitura)
date
uptime
free -g | head -2
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null | head
sudo docker ps -a --format '{{.Names}}\t{{.Status}}' | head -30
echo "dataset json: $(ls ~/braco/dataset/*.json 2>/dev/null | wc -l)"
