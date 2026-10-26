#!/bin/bash
# testa exatamente o comando documentado no READY.json / INFERENCIA.md (sem o rodar.sh)
sudo docker rm -f cerise26_pose_docteste >/dev/null 2>&1
sudo cp ~/braco/dataset/002079_camera_diagonal_image.png ~/cerise2026/pose/inferencia/teste_002079_diagonal.png
sudo docker run --rm --name cerise26_pose_docteste --gpus all --ipc=host -v ~/cerise2026/pose:/workspace/pose \
  -v ~/cerise2026/hf_cache:/root/.cache/huggingface -e POSE_REPO=/workspace/pose/src/repo -e HF_HUB_OFFLINE=1 cerise26_pose:latest \
  python /workspace/pose/inferencia/predict.py --ckpt /workspace/pose/inferencia/modelo/pose_servo6dof.pt \
  --imagem /workspace/pose/inferencia/teste_002079_diagonal.png --camera diagonal --saida /workspace/pose/inferencia/overlay_teste_doc.png 2>&1 | \
  grep -v -i 'copyright\|nvidia\|governing\|found at\|rights\|caffe\|idiap\|google\|yangqing\|^$\|====\|pytorch\|cuda\|shmem\|docker run\|insufficient\|driver' | head -20
python3 -c "import json; d=json.load(open('/home/energiainteligente/braco/dataset/002079.json')); import math; print('verdade (graus):', {k: round(math.degrees(v),2) for k,v in d['juntas_reais'].items()})"
