# cerise-2026-pose-servo6dof

**CERISE/EMC/UFG — Plano de Trabalho 2026** ("Pesquisa, Desenvolvimento e Inovação em Gêmeos Digitais, Tecnologias
Imersivas, Robótica e Navegação Autônoma no escopo do CERISE/EMC/UFG").
**Objetivo 3, Atividades 8 e 9**, com interface para o Objetivo 4 (gêmeo digital).

- **Atividade 8:** aplicação do modelo de estimação de pose de braço robótico desenvolvido em 2025
  (HPE/HoRoPose + ramo de profundidade Depth Anything V2, variante *MyNet*) ao **novo dataset sintético do braço
  servo6dof** gerado em 2026 (Atividade 7).
- **Atividade 9:** avaliação de precisão (MAE por junta, erro 3D, ADD/AUC, PCK 2D), generalização entre câmeras,
  robustez a perturbações de imagem e latência.
- **Objetivo 4:** módulo de inferência (`inferencia/predict.py`, classe `PosePredictor`) consumido pelo gêmeo digital.

> Autor: **Eduardo Martins de Carvalho Caixeta** (bolsista, CERISE/EMC/UFG).
> Execução: DGX Spark (NVIDIA GB10, Blackwell, aarch64), 26/10/2026.

Seção completa do relatório técnico (metodologia, tabelas, figuras e discussão): [`RELATORIO_SECAO.md`](RELATORIO_SECAO.md).

## Estrutura de pastas

| pasta | conteúdo |
|---|---|
| `hpe2025/` | **porte do código de 2025** (MyNet/HPE): `models/full_net.py` (RootNetwithRegInt + fusão de profundidade), `models/depth_net.py` (DepthNet/RootNet), backbones ResNet-50/HRNet-W32, camada integral e código do Depth Anything V2. Alterações marcadas com `[porte 2026]`. `robot_servo6dof.py` substitui o `URDFRobot` (Panda) por uma cinemática direta diferenciável do servo6dof. |
| `servo6dof/` | URDF gerado pelo xacro (`servo6dof.urdf`), `geometria.py` (cinemática direta numpy/torch, 7 keypoints, poses/intrínsecos das câmeras do `estudio_hd.sdf`), `gerar_urdf.sh`. |
| `servo6dof/geometria_referencia/` | cópia de referência usada na verificação da geometria: pacotes ROS 2 `servo6dof_description`/`servo6dof_controller` (xacro original e modificado, launch, meshes STL, modelos do mundo), URDFs gerados a partir do xacro e do pacote, e `diff_xacro_vs_repo.txt`. |
| `pose26/` | código novo de 2026: dataset/augmentations, construção do modelo + transferência de pesos, métricas. |
| `scripts/` | conversor do dataset, pré-cálculo do Depth Anything V2, treino (etapa 1 DepthNet → etapa 2 completo), avaliação (métricas, robustez, latência), figuras, tabelas, verificação da geometria, inspeção dos checkpoints de 2025. |
| `scripts/spark/` | utilitários de execução no Spark: `rodar.sh` (sobe cada passo num contêiner `cerise26_pose_<nome>`), sequências de treino/avaliação (`final.sh`, `final2.sh`, `sequencia.sh`), fiscalização da produção, testes do módulo de inferência. |
| `inferencia/` | `predict.py` (`PosePredictor`), `INFERENCIA.md`, `crop.json` do pacote final (`modelo/`) e do pacote de desenvolvimento (`dev/`), latência medida. Os pesos (`modelo/pose_servo6dof.pt`) estão nos Releases. |
| `docker/` | `Dockerfile` da imagem `cerise26_pose:latest` (FROM `nvcr.io/nvidia/pytorch:26.05-py3`) e `check_env.py`. |
| `resultados/` | métricas medidas (`metricas_teste_*.json`, `curvas_*.csv`, `latencia_predictor.json`, `READY.json`), predições do teste (`predicoes_teste_*.npz`), tabelas (`tabelas_relatorio.md`), verificação da geometria, inspeção dos checkpoints de 2025. |
| `resultados/runs/` | metadados de **todas** as rodadas de treino feitas no Spark (finais e exploratórias): `treino.log`, `curvas.csv`, `transfer_info.json` e, quando avaliadas, `aval/metricas_teste.json` e `aval/*.npz`. |
| `resultados/dados_convertidos/` | metadados das três conversões do dataset (`final`, `parcial`, `parcial2`): `index.npz` (ids, splits, keypoints, ângulos, K), `crop.json` (recorte fixo por câmera) e `resumo_conversao.json`. Os memmaps de imagem/profundidade não estão aqui (ver abaixo). |
| `resultados/logs/` | logs das execuções (build da imagem, ambiente do contêiner, conversão, treinos, avaliações, sequências, fiscalização e religamento da produção). |
| `resultados/imagens_teste/` | 6 imagens do conjunto de teste (poses 004353, 002079 e 004743 — percentis 25/50/75 do erro 3D do modelo de transferência — nas duas câmeras), com `selecao.json`; usadas nas figuras de predição e nos testes do módulo de inferência. |
| `evidencias/` | figuras e prints de terminal do relatório; `evidencias/inferencia/` tem saídas adicionais do `predict.py`. |
| `legado_2025/` | código de 2025 (dissertação de mestrado) de onde o modelo foi portado: `MyNet/` (variante final, com configs, scripts e resultados por época dos experimentos) e `Holistic-Robot-Pose-Estimation/` (cópia de trabalho adaptada do HPE). Ver [`legado_2025/LEIAME.md`](legado_2025/LEIAME.md). |
| `pesos_2025/` | registros da cópia dos checkpoints de 2025 a partir do Google Drive (os checkpoints estão nos Releases). |
| `RELEASES.md` | tabela dos artefatos grandes (checkpoints) publicados como Releases, com SHA-256 e instruções. |

## Como reproduzir (DGX Spark ou outra máquina com GPU NVIDIA)

```bash
# 0) imagem (base NGC PyTorch 26.05; no Spark a imagem gerada tem ~35 GB e nao e versionada)
sudo docker build -t cerise26_pose:latest -f docker/Dockerfile docker/
# 1) pesos de terceiros (nao redistribuidos): Depth Anything V2 Large (Hugging Face do autor),
#    HRNet-W32 ImageNet (arquivo oficial hrnet_w32-36af842e) e ResNet-50 ImageNet (torchvision, baixado sozinho)
python scripts/baixar_pesos_da.py vitl          # -> cache HF (montar em /root/.cache/huggingface)
export HPE_HRNET_W32=/caminho/hrnet_w32-36af842e_roc.pth
# 2) checkpoints de 2025 e de 2026: baixar dos Releases e extrair (ver RELEASES.md)
# 3) URDF a partir do xacro (imagem ROS servo6dof:humble, copia do pacote)
bash servo6dof/gerar_urdf.sh
# 4) verificacao da geometria (overlay dos keypoints projetados)
python scripts/verificar_geometria.py --dataset /dados/dataset --saida evidencias/
# 5) conversao (split 80/10/10 por id de pose, semente 2026; recorte fixo por camera) + profundidade DA-V2
#    (--crop_json reproduz exatamente os recortes da execucao original)
python scripts/converter_dataset.py --dataset /dados/dataset --saida /dados/final --workers 6 \
  --crop_json resultados/dados_convertidos/final/crop.json
python scripts/precomputar_profundidade.py --dados /dados/final --bs 16
# 6) treino: etapa 1 (DepthNet) e etapa 2 (modelo completo); --init 2025 (transferencia) ou imagenet (baseline)
python scripts/treinar.py --dados /dados/final --saida /runs/etapa1_depthnet_transfer_v2 --etapa depthnet --init 2025 \
  --ckpt_2025 /workspace/pose/pesos/realsense_full_checkpoint.pk --epocas 5
python scripts/treinar.py --dados /dados/final --saida /runs/final_full_transfer --etapa full --init 2025 \
  --rootnet /runs/etapa1_depthnet_transfer_v2/melhor.pt --epocas 20 --bs 32 --workers 4
# 7) avaliacao no teste (+ robustez e latencia) e exportacao do pacote de inferencia
python scripts/avaliar.py --dados /dados/final --ckpt /runs/final_full_transfer/melhor.pt --saida /runs/final_full_transfer/aval \
  --robustez --n_robustez 400 --latencia
python scripts/exportar_modelo.py --ckpt /runs/final_full_transfer/melhor.pt --dados /dados/final \
  --saida inferencia/modelo/pose_servo6dof.pt --metricas /runs/final_full_transfer/aval/metricas_teste.json
# 8) figuras e tabelas do relatorio
python scripts/figuras.py --res resultados --out evidencias
python scripts/tabelas.py resultados
```

`scripts/spark/rodar.sh <nome> <comando>` sobe cada passo num contêiner `cerise26_pose_<nome>` com as montagens
usadas no Spark (código, dados, pesos, cache HF, `runs/`). A sequência exata usada nas rodadas finais está em
`scripts/spark/final.sh`, `final2.sh` e `sequencia.sh`; os argumentos de cada rodada ficam registrados na primeira
linha de `resultados/runs/<rodada>/treino.log`. Observação: na execução original, a etapa 1 usou a conversão
`parcial2` (3151 poses de treino disponíveis às 03h) e a etapa 2, a conversão `final` (dataset completo).

**Somente inferência** (sem treinar): baixe `pose_servo6dof_modelo_final.tar` dos Releases, extraia na raiz do
repositório (cria `inferencia/modelo/pose_servo6dof.pt`) e siga [`inferencia/INFERENCIA.md`](inferencia/INFERENCIA.md).

### Como regenerar os dados convertidos (`/dados/final`)

Os memmaps 256×256 (RGB e profundidade DA-V2) não são versionados nem publicados (~5,7 GB nas três conversões,
regeneráveis). Partindo do dataset sintético da Atividade 7 (5000 poses × 2 câmeras, `~/braco/dataset` no Spark), o
passo 5 acima recria `rgb256_{treino,val,teste}.npy`, `depth256_{treino,val,teste}.npy`, `index.npz`, `crop.json` e
`resumo_conversao.json`. Os `index.npz`/`crop.json`/`resumo_conversao.json` gerados na execução original estão em
`resultados/dados_convertidos/` para conferência (split final: 4000/500/500 poses = 8000/1000/1000 imagens, semente
2026; recortes `frontal` [-38,7, -162,5, 1045,0] e `diagonal` [122,5, -76,9, 846,6]).

## Resultados principais (teste: 500 poses × 2 câmeras, nunca vistas no treino)

| | Transferência (pesos de 2025) | Baseline (ImageNet) |
|---|---|---|
| MAE médio das 5 juntas | **12,68°** (juntas 1–4: 5,4°) | 13,51° |
| Erro 3D médio / mediano dos keypoints | **58,7 / 41,1 mm** | 88,3 / 65,7 mm |
| Acurácia@60 mm (métrica da Tabela I de 2025) | **68,0%** | 32,8% |
| AUC-ADD (0–100 mm) | **0,471** | 0,265 |
| PCK 2D @5 / 10 / 20 px (1280×720) | 30,1 / **61,7** / 87,1% | 19,9 / 43,6 / 71,4% |
| Épocas | 20 | 12 |

- MAE por junta (transferência): 4,19° / 3,04° / 3,91° / 10,52° / 41,73° — a junta 5 (rotação da garra) não é
  observável pelos keypoints escolhidos (keypoint sobre o eixo).
- Referência de 2025 (Panda, DREAM real): RoboPose 74,3%, HPRE 75,2%, modelo de 2025 75,4% @60 mm — robô 2,4× menor,
  dados reais; **não é comparação direta**.
- Generalização entre câmeras: treinado só com a frontal → erro 3D 140 mm na frontal e 419 mm na diagonal (não generaliza).
- Robustez: estável a brilho/contraste; sensível a ruído (σ=0,05 → 99 mm), desfoque (7 px → 96 mm), JPEG q30 (84 mm),
  oclusão de 30% (94 mm) e resolução 25% (109 mm).
- Latência no GB10 (lote 1, GPU compartilhada): 162,5 ms ponta a ponta (6,15 FPS), 2,94 GB de GPU; o DA-V2 ViT-L é ~65% do custo.

Números brutos em `resultados/` (`metricas_teste_*.json`, `curvas_*.csv`, `predicoes_teste_*.npz`,
`latencia_predictor.json`, `READY.json`) e por rodada em `resultados/runs/`.

### Rodadas de treino (todas registradas em `resultados/runs/`)

| rodada | o que é | dados | checkpoint |
|---|---|---|---|
| `final_full_transfer` | **modelo final** (etapa 2, pesos de 2025, 20 épocas) | `final` (5000 poses) | Release |
| `final_full_imagenet` | baseline ImageNet (etapa 2, 12 épocas) | `final` | Release |
| `etapa1_depthnet_transfer_v2` | etapa 1 (DepthNet) a partir de `realsense_full_checkpoint.pk`, usada pelo modelo final | `parcial2` | Release |
| `etapa1_depthnet_imagenet_v2` | etapa 1 (DepthNet) ImageNet, usada pelo baseline | `parcial2` | Release |
| `xcam_frontal_transfer` | generalização entre câmeras (treino só com a frontal, 10 épocas, melhor na 6ª) | `parcial2` | Release |
| `xcam_frontal_transfer_bug_integral`, `xcam_frontal_transfer_depthw1` | versões preliminares do experimento entre câmeras, interrompidas (6 e 3 épocas registradas) e substituídas pela rodada acima | `parcial2` | só no Spark |
| `etapa1_depthnet_transfer_v2_ckpt_ep207` | etapa 1 a partir de `best_model_auc.pk` (época 207), interrompida | `parcial2` | só no Spark |
| `etapa1_depthnet_transfer`, `etapa1_depthnet_imagenet` | primeira versão da etapa 1 (3 épocas) | `parcial` | só no Spark |
| `dev_full_transfer`, `smoke_full` | testes de desenvolvimento do treino completo (1–2 épocas) | `parcial` | só no Spark |
| `final_full_imagenet_interrompido_memoria` | primeira tentativa do baseline em paralelo, interrompida por falta de memória | `final` | — |

Os horários dentro dos logs dos contêineres estão em UTC (+3 h em relação ao horário de Brasília).

## Artefatos grandes (Releases)

Os checkpoints (> 50 MB) não ficam no git; estão publicados nos Releases do repositório. A lista completa, com
tamanho, SHA-256, conteúdo e instruções para baixar e extrair, está em [`RELEASES.md`](RELEASES.md):
modelo final `pose_servo6dof.pt` (+ `crop.json`), melhores checkpoints das rodadas finais (transferência, ImageNet,
DepthNet etapa 1, entre câmeras) e os dois checkpoints de 2025 do experimento `realsense_full`.

## O que não está aqui e por quê

| item | motivo | como obter |
|---|---|---|
| Memmaps do dataset convertido (`/dados/{final,parcial,parcial2}/*.npy`, ~5,7 GB) | regeneráveis pelo conversor | `scripts/converter_dataset.py` + `scripts/precomputar_profundidade.py` (ver acima) |
| Dataset sintético do braço (imagens RGB, profundidade, segmentação, JSON das juntas; 5000 poses) | produzido pela Atividade 7 (geração do dataset do braço), não por esta atividade | `~/braco/dataset` no Spark (`index.csv` consolidado) |
| Pesos do Depth Anything V2 Large | terceiros (modelos Base/Large sob CC-BY-NC-4.0) | Hugging Face `depth-anything/Depth-Anything-V2-Large`, arquivo `depth_anything_v2_vitl.pth` (`scripts/baixar_pesos_da.py vitl`) |
| HRNet-W32 ImageNet (`hrnet_w32-36af842e_roc.pth`, 133 MB, SHA-256 `12682d70057169f2e593b18173e5388627205e666f39bf9e082e08b95098a4c6`) | pesos de terceiros (ImageNet) | arquivo oficial `hrnet_w32-36af842e.pth` do HRNet (HRNet-Image-Classification); usado via `HPE_HRNET_W32` |
| ResNet-50 ImageNet (`resnet50-0676ba61.pth`) | pesos de terceiros (torchvision) | baixado automaticamente pelo torchvision (`IMAGENET1K_V1`) |
| Código oficial do Depth Anything V2 dentro de `legado_2025/` | clone de repositório de terceiros, sem alterações de código | `git clone https://github.com/DepthAnything/Depth-Anything-V2` no commit `e5a2732d3ea2cddc081d7bfd708fc0bf09f812f1` (ver `legado_2025/LEIAME.md`) |
| Imagem Docker `cerise26_pose:latest` (~35,5 GB) | imagem própria, porém muito grande; regenerável | `docker/Dockerfile` (base `nvcr.io/nvidia/pytorch:26.05-py3`); log do build em `resultados/logs/docker_build.log` |
| Imagem ROS 2 `servo6dof:humble` | pertence à geração do dataset (Atividade 7) | usada só por `servo6dof/gerar_urdf.sh`; o URDF resultante está versionado |
| Checkpoints das rodadas exploratórias e `ultimo.pt` de todas as rodadas (~5 GB) | intermediários, superados pelas rodadas finais | permanecem no Spark em `~/cerise2026/pose/runs/`; métricas e logs estão em `resultados/runs/` |
| Pacote de desenvolvimento `inferencia/dev/pose_dev.pt` (446 MB) | pacote de teste do `predict.py`, superado pelo pacote final | no Spark em `~/cerise2026/pose/inferencia/dev/` (`crop.json` versionado) |
| `Cópia de best_model_auc_1.pk` (2025) | duplicata exata de `best_model_auc.pk` (mesmo SHA-256 `72b6b49b…cec1`) | desnecessária |

Segredos: nenhum arquivo de credencial (keystore, token, chave de API ou senha) faz parte deste trabalho.

## Correções feitas no código de 2025 durante o porte

- `full_net.py` (MyNet): a profundidade da raiz era calculada **depois** da camada integral, que recebia
  `root_trans = 0` (keypoints 3D integrais sem a profundidade da raiz). Restaurada a ordem do HPE original.
- `full_net.py`: faltava `import` do `DepthFeatureExtractor`; caminhos fixos do Colab trocados por relativos.
- `train_full.py` chamava `optimizer.step()` duas vezes por iteração; no treino novo é uma vez.

## Pendências e próximos passos

- A etapa 3 de 2025 (autossupervisionada com imagens reais) não se aplica enquanto não houver imagens reais do braço
  servo6dof; depende do braço físico.
- A junta 5 (rotação da garra) não é observável com os keypoints atuais (MAE ~42°) e a `joint_6` (garra) não é
  estimada: é preciso adicionar keypoints nas pontas das pinças e re-treinar.
- O baseline ImageNet teve 12 épocas e a transferência 20 (limite de tempo); a comparação no mesmo orçamento está na
  validação, época 12 (65,0 × 83,7 mm). Completar o baseline até 20 épocas é possível com `--retomar`/`--epoca_inicial`
  do `treinar.py`.
- A etapa 1 (DepthNet) e o experimento entre câmeras usaram as 3151 poses de treino disponíveis às 03h; o modelo final
  e todas as avaliações usaram o dataset completo (5000 poses).
- A latência foi medida com a GPU compartilhada; para tempo real falta testar DA-V2 Small/Base ou TensorRT.
- Observados durante a execução, fora do escopo deste repositório: `~/braco/finalizar.sh` chama
  `~/braco/dataset_run_restore.sh`, que não existe (o correto é `~/dataset_run_restore.sh`), e no `gerar_hd_spark.sh`
  cada worker executa `pkill -f 'ign gazebo'` ao terminar, derrubando os simuladores dos demais.

## Código de terceiros incluído

- `hpe2025/` e `legado_2025/` derivam de *Real-time Holistic Robot Pose Estimation with Unknown States* (HPE/HoRoPose,
  ECCV 2024; repositório `Oliverbansk/Holistic-Robot-Pose-Estimation`, que não declara licença), incluindo os módulos
  auxiliares `ctrnet` e `urdfpytorch` presentes na cópia de trabalho do HPE.
- `hpe2025/depth_anything_v2/` é o código do Depth Anything V2 (Apache-2.0), com a atenção trocada por
  `scaled_dot_product_attention` (marcado `[porte 2026]`).

## Licença

Este repositório não contém arquivo `LICENSE`, por decisão do autor. Os componentes de terceiros listados acima
mantêm as licenças dos respectivos projetos.
