#!/bin/bash
cp ~/cerise2026/pose/dados/parcial/depth256_teste.npy /tmp/cerise26_pose_depth_old.npy
sudo docker run --rm --gpus all --ipc=host --name cerise26_pose_da \
  -v ~/cerise2026/hf_cache:/root/.cache/huggingface -v ~/cerise2026/pose/src/repo:/workspace/pose -v ~/cerise2026/pose/dados:/dados \
  cerise26_pose:latest python scripts/precomputar_profundidade.py --dados /dados/parcial --splits teste --bs 16 2>&1 | tail -1
python3 - <<'EOF'
import numpy as np
a = np.load('/tmp/cerise26_pose_depth_old.npy').astype(float)
b = np.load('/home/energiainteligente/cerise2026/pose/dados/parcial/depth256_teste.npy').astype(float)
print('MAE entre versoes (0-255):', np.abs(a - b).mean(), 'corr', np.corrcoef(a[:50].ravel(), b[:50].ravel())[0, 1])
EOF
rm -f /tmp/cerise26_pose_depth_old.npy
