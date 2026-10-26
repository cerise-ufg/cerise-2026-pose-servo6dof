# cerise-2026-pose-servo6dof

**CERISE/EMC/UFG — Plano de Trabalho 2026, Objetivo 3 (Atividades 8 e 9).**
Aplicação do modelo de estimação de pose de braço robótico desenvolvido em 2025 (HPE/HoRoPose + ramo de
profundidade Depth Anything V2, variante *MyNet*) ao **novo dataset sintético do braço servo6dof** gerado em
2026 (Atividade 7), com avaliação de precisão e robustez, e módulo de inferência para o gêmeo digital (Objetivo 4).

> Bolsista: Eduardo Martins de Carvalho Caixeta. Execução: DGX Spark (NVIDIA GB10, aarch64), 26/10/2026.

## O que tem aqui

| pasta | conteúdo |
|---|---|
| `hpe2025/` | **porte do código de 2025** (MyNet/HPE): `models/full_net.py` (RootNetwithRegInt + fusão de profundidade), `models/depth_net.py` (DepthNet/RootNet), backbones ResNet-50/HRNet-W32, camada integral, Depth Anything V2. Alterações marcadas com `[porte 2026]`. `robot_servo6dof.py` substitui o `URDFRobot` (Panda) por uma FK diferenciável do servo6dof. |
| `servo6dof/` | URDF gerado pelo xacro (`servo6dof.urdf`), `geometria.py` (FK numpy/torch, 7 keypoints, poses/intrínsecos das câmeras do `estudio_hd.sdf`), `gerar_urdf.sh`. |
| `pose26/` | código novo de 2026: dataset/augmentations, construção do modelo + transfer learning, métricas. |
| `scripts/` | conversor do dataset, pré-cálculo do DA-V2, treino (etapas DepthNet → completo), avaliação (métricas, robustez, latência), figuras, verificação da geometria, utilitários do Spark (`scripts/spark/`). |
| `inferencia/` | `predict.py` (`PosePredictor`) + `INFERENCIA.md` — usado pela frente do gêmeo digital. |
| `docker/` | `Dockerfile` da imagem `cerise26_pose:latest` (FROM `nvcr.io/nvidia/pytorch:26.05-py3`). |
| `resultados/` | métricas medidas (`.json/.csv`) e predições do teste (`.npz`). |
| `evidencias/` | figuras e prints de terminal usados no relatório. |
| `RELATORIO_SECAO.md` | seção do relatório técnico (A8/A9). |

## Como rodar (no Spark)

```bash
# 0) imagem
sudo docker build -t cerise26_pose:latest -f docker/Dockerfile docker/
# 1) URDF a partir do xacro (imagem ROS servo6dof:humble, copia do pacote)
bash servo6dof/gerar_urdf.sh
# 2) verificacao da geometria (overlay dos keypoints projetados)
python scripts/verificar_geometria.py --dataset /dados/dataset --saida evidencias/
# 3) conversao (split 80/10/10 por id de pose, semente 2026; recorte fixo por camera) + profundidade DA-V2
python scripts/converter_dataset.py --dataset /dados/dataset --saida /dados/final
python scripts/precomputar_profundidade.py --dados /dados/final
# 4) treino: etapa 1 (DepthNet) e etapa 2 (modelo completo); --init 2025 (transfer) ou imagenet (baseline)
python scripts/treinar.py --dados /dados/final --saida /runs/dn --etapa depthnet --init 2025 --epocas 5
python scripts/treinar.py --dados /dados/final --saida /runs/full --etapa full --init 2025 --rootnet /runs/dn/melhor.pt --epocas 12
# 5) avaliacao no teste (+ robustez e latencia)
python scripts/avaliar.py --dados /dados/final --ckpt /runs/full/melhor.pt --saida /runs/full/aval --robustez --latencia
```
`scripts/spark/rodar.sh <nome> <comando>` sobe cada passo num container `cerise26_pose_<nome>` com as montagens certas.

## Onde ficam os pesos (não versionados)

- Modelo final de 2025 (Panda, etapa autossupervisionada RealSense): `G:/Meu Drive/Mestrado/MyNet/experiments/panda/realsense_full/best_model_auc.pk`
  (cópia no Spark: `~/cerise2026/pose/pesos/best_model_auc.pk`).
- HRNet-W32 ImageNet: `~/cerise2026/pose/pesos/hrnet_w32-36af842e_roc.pth`; ResNet-50 ImageNet: torchvision.
- Depth Anything V2 Large: Hugging Face `depth-anything/Depth-Anything-V2-Large` (cache `~/cerise2026/hf_cache`).
- Modelos treinados em 2026: `~/cerise2026/pose/runs/*/melhor.pt`; pacote de inferência: `~/cerise2026/pose/inferencia/modelo/pose_servo6dof.pt`.

## Resultados

Ver `RELATORIO_SECAO.md` (tabelas completas) e `resultados/`.
