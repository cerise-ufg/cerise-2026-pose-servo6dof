#!/bin/bash
# verifica o pacote de inferencia exportado (epoca/init/metricas) e roda o comando documentado
bash ~/cerise2026/pose/src/repo/scripts/spark/teste_doc_inferencia.sh 2>&1 | grep -A6 joints_graus
sudo docker run --rm -v ~/cerise2026/pose:/w cerise26_pose:latest python - <<'EOF' 2>&1 | tail -1
import torch
c = torch.load('/w/inferencia/modelo/pose_servo6dof.pt', map_location='cpu', weights_only=False)
print('pacote: epoca', c['epoch'] + 1, 'init', c['init'], 'MAE_teste', round(c['metricas_teste']['MAE_medio_graus'], 2))
EOF
