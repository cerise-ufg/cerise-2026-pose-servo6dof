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
python scripts/treinar.py --dados /dados/final --saida /runs/full --etapa full --init 2025 --rootnet /runs/dn/melhor.pt --epocas 20
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

## Resultados (teste: 500 poses × 2 câmeras, nunca vistas no treino)

| | Transferência (pesos de 2025) | Baseline (ImageNet) |
|---|---|---|
| MAE médio das 5 juntas | **12,68°** (juntas 1–4: 5,4°) | 13,51° |
| Erro 3D médio / mediano dos keypoints | **58,7 / 41,1 mm** | 88,3 / 65,7 mm |
| Acurácia@60 mm (métrica da Tabela I de 2025) | **68,0%** | 32,8% |
| AUC-ADD (0–100 mm) | **0,471** | 0,265 |
| PCK 2D @5 / 10 / 20 px (1280×720) | 30,1 / **61,7** / 87,1% | 19,9 / 43,6 / 71,4% |
| Épocas | 20 | 12 |

- MAE por junta (transferência): 4,2° / 3,0° / 3,9° / 10,5° / 41,7° — a junta 5 (rotação da garra) não é observável
  pelos keypoints escolhidos (keypoint sobre o eixo).
- Referência de 2025 (Panda, DREAM real): RoboPose 74,3%, HPRE 75,2%, modelo de 2025 75,4% @60 mm — robô 2,4× menor,
  dados reais; **não é comparação direta**.
- Generalização entre câmeras: treinado só com a frontal → erro 3D 140 mm na frontal e 419 mm na diagonal (não generaliza).
- Robustez: estável a brilho/contraste; sensível a ruído (σ=0,05 → 99 mm), desfoque (7 px → 96 mm), JPEG q30 (84 mm),
  oclusão de 30% (94 mm) e resolução 25% (109 mm).
- Latência no GB10 (lote 1, GPU compartilhada): 162,5 ms ponta a ponta (6,2 FPS), 2,94 GB de GPU; o DA-V2 ViT-L é ~65% do custo.

Detalhes, figuras e discussão: `RELATORIO_SECAO.md`; números brutos em `resultados/` (`metricas_teste_*.json`,
`curvas_*.csv`, `predicoes_teste_*.npz`, `latencia_predictor.json`, `READY.json`).

## Correções feitas no código de 2025 durante o porte

- `full_net.py` (MyNet): a profundidade da raiz era calculada **depois** da camada integral, que recebia
  `root_trans = 0` (keypoints 3D integrais sem a profundidade da raiz). Restaurada a ordem do HPE original.
- `full_net.py`: faltava `import` do `DepthFeatureExtractor`; caminhos fixos do Colab trocados por relativos.
- `train_full.py` chamava `optimizer.step()` duas vezes por iteração; no treino novo é uma vez.
