#!/bin/bash
# uso: rodar.sh <nome_container_sufixo> <comando python e argumentos...>
# Sobe um container destacado cerise26_pose_<sufixo> com GPU, montando codigo, dados, pesos e cache HF.
# Limite de RAM de CPU do container (protecao); o teto de GPU (14 GB) e controlado pelo batch e monitorado via torch.cuda.
NOME=cerise26_pose_$1; shift
MEM=${MEM_LIMITE:-32g}
sudo docker rm -f $NOME >/dev/null 2>&1
sudo docker run -d --gpus all --ipc=host --name $NOME --memory=$MEM --memory-swap=$MEM \
  -e HPE_HRNET_W32=/workspace/pose/pesos/hrnet_w32-36af842e_roc.pth -e PYTHONUNBUFFERED=1 \
  -v ~/cerise2026/hf_cache:/root/.cache/huggingface \
  -v ~/cerise2026/pose/pesos/torch_hub:/root/.cache/torch \
  -v ~/cerise2026/pose/src/repo:/workspace/pose/repo \
  -v ~/cerise2026/pose/pesos:/workspace/pose/pesos \
  -v ~/cerise2026/pose/dados:/dados \
  -v ~/braco/dataset:/dados/dataset:ro \
  -v ~/cerise2026/pose/inferencia:/workspace/pose/inferencia \
  -v ~/cerise2026/pose/runs:/runs \
  -w /workspace/pose/repo cerise26_pose:latest "$@"
