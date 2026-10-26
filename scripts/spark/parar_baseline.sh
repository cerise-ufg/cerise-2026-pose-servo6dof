#!/bin/bash
# para o baseline ImageNet que rodava em paralelo (memoria apertada apos a volta da producao)
sudo docker stop -t 20 cerise26_pose_full_imagenet >/dev/null 2>&1
sudo docker rm cerise26_pose_full_imagenet >/dev/null 2>&1
cd ~/cerise2026/pose/runs && [ -d final_full_imagenet ] && sudo mv final_full_imagenet final_full_imagenet_interrompido_memoria
echo "[$(date '+%Y-%m-%d %H:%M:%S')] POSE: baseline imagenet interrompido (available 15 GB, vllm-stt com 2 reinicios na partida) - sera rodado em sequencia apos o transfer" | tee -a ~/cerise2026/pose/logs/fiscalizacao_producao.log
sleep 20
free -g | head -2
