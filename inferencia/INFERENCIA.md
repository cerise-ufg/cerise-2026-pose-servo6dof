# Inferência — estimação de pose do servo6dof (frente POSE, CERISE 2026)

Modelo: HPE/HoRoPose + ramo de profundidade Depth Anything V2 (modelo de 2025, variante *MyNet*),
adaptado ao braço **servo6dof** e re-treinado no dataset sintético de 2026 (Atividade 7).
Entrada: **uma imagem RGB 1280×720** de uma das câmeras do estúdio (`frontal` ou `diagonal`).
Saída: ângulos `joint_1..joint_5` (rad), 7 keypoints 2D/3D e a latência.

## Onde está tudo (DGX Spark, usuário `energiainteligente`)

| item | caminho |
|---|---|
| imagem Docker | `cerise26_pose:latest` (FROM `nvcr.io/nvidia/pytorch:26.05-py3`) |
| módulo | `~/cerise2026/pose/inferencia/predict.py` (classe `PosePredictor`) |
| checkpoint empacotado | `~/cerise2026/pose/inferencia/modelo/pose_servo6dof.pt` (+ `crop.json`) |
| código (pacotes `pose26`, `hpe2025`, `servo6dof`) | `~/cerise2026/pose/src/repo` (o `predict.py` acha sozinho; ou `POSE_REPO=...`) |
| pesos Depth Anything V2 (vitl) | cache HF em `~/cerise2026/hf_cache` (montar em `/root/.cache/huggingface`) |
| status/métricas | `~/cerise2026/pose/READY.json` |

## Obter o modelo fora do Spark

O checkpoint empacotado (`inferencia/modelo/pose_servo6dof.pt`, 446 MB) não é versionado no git: baixe
`pose_servo6dof_modelo_final.tar` dos Releases do repositório e extraia na raiz (ver `../RELEASES.md`).
O `crop.json` correspondente já está em `inferencia/modelo/crop.json`. Os pesos do Depth Anything V2 Large
vêm do Hugging Face (`python scripts/baixar_pesos_da.py vitl`).

## Como chamar (Python, dentro do container)

```python
import sys; sys.path.insert(0, "/workspace/pose/inferencia")
from predict import PosePredictor
p = PosePredictor("/workspace/pose/inferencia/modelo/pose_servo6dof.pt", device="cuda")
out = p.predict(rgb_uint8_HxWx3, camera="frontal")   # RGB (não BGR!), 1280x720
out["joints"]        # {"joint_1": rad, ..., "joint_5": rad}
out["keypoints_2d"]  # 7 x [u, v] em pixels da imagem de entrada
out["keypoints_3d"]  # 7 x [x, y, z] em metros, frame ótico da câmera
out["keypoint_names"]# ["base","junta_1",...,"junta_5","ponta_garra"]
out["latency_ms"]    # DA-V2 + HPE, com sincronização da GPU
```

Container (montagens mínimas):

```bash
sudo docker run --rm --gpus all --ipc=host \
  -v ~/cerise2026/pose:/workspace/pose \
  -v ~/cerise2026/hf_cache:/root/.cache/huggingface \
  -e POSE_REPO=/workspace/pose/src/repo -e HF_HUB_OFFLINE=1 \
  cerise26_pose:latest \
  python /workspace/pose/inferencia/predict.py \
     --ckpt /workspace/pose/inferencia/modelo/pose_servo6dof.pt \
     --imagem /caminho/foto_1280x720.png --camera frontal --saida /workspace/pose/inferencia/overlay.png
```

O CLI imprime JSON com `joints_rad`, `joints_graus`, `keypoints_2d`, `latency_ms` e `gpu_mem_pico_GB`.

## Observações importantes

- **Câmeras fixas**: o recorte de entrada (quadrado → 256×256) é fixo por câmera (`crop.json`) e os
  intrínsecos são os do estúdio (HFOV 1,0472 rad, 1280×720 → fx = fy ≈ 1108,5, cx = 640, cy = 360).
  Imagens de outra resolução são reescaladas para 1280×720; câmeras em outras poses exigem re-treino.
- A garra (`joint_6`) **não** é estimada (keypoint "ponta_garra" é fixo no frame de `joint_5`).
- Memória de GPU medida no batch 1: ver `READY.json` (`gpu_mem_pico_GB`, ~3 GB).
- Juntas no intervalo [-π/2, π/2] (limites do URDF). Convenção de sinais = `/joint_states` do Gazebo.
