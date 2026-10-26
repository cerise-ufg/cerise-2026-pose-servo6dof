# Roteiro de implementação — Treino e avaliação do modelo de estimação de pose

*CERISE/EMC/UFG, Plano de Trabalho 2026. Autor: Eduardo Martins de Carvalho Caixeta.*

## O que será reproduzido

Este roteiro descreve como adaptar o modelo de estimação de pose de braço robótico desenvolvido em 2025 (HPE/HoRoPose com ramo de profundidade Depth Anything V2, variante MyNet) ao braço servo6dof, treiná-lo no dataset sintético de 2026 e avaliá-lo. O código está no repositório `cerise-2026-pose-servo6dof` da organização cerise-ufg no GitHub (privado; o acesso é concedido mediante solicitação ao bolsista). O roteiro cobre o Objetivo 3, Atividades 8 (aplicação do modelo de 2025 ao novo dataset) e 9 (avaliação de precisão e robustez), e a interface com o Objetivo 4: o módulo de inferência consumido pelo gêmeo digital na Atividade 12.

Ao final, você terá:

- a imagem Docker de treino `cerise26_pose:latest` (base NGC PyTorch 26.05);
- o dataset convertido para o formato do modelo (recortes 256×256 e profundidade relativa do Depth Anything V2);
- os modelos treinados por transferência dos pesos de 2025 (etapa 1, DepthNet; etapa 2, modelo completo) e pela linha de base ImageNet;
- as métricas no conjunto de teste: MAE por junta, erro 3D, ADD/AUC, PCK 2D, robustez a sete perturbações, generalização entre câmeras e latência;
- o pacote de inferência `pose_servo6dof.pt` e a classe `PosePredictor` funcionando sobre uma imagem do teste.

## Requisitos

- Hardware: GPU NVIDIA com pelo menos 12 GB e 32 GB de RAM para o container (limite de [`scripts/spark/rodar.sh`](scripts/spark/rodar.sh)). Picos medidos de GPU: 9,3 GB no treino completo com lote 32 ([`resultados/runs/final_full_transfer/treino.log`](resultados/runs/final_full_transfer/treino.log)), 7,6 GB na profundidade ([`resultados/logs/conversao_final.txt`](resultados/logs/conversao_final.txt)) e 2,94 GB na inferência ([`resultados/latencia_predictor.json`](resultados/latencia_predictor.json)). O projeto usou um NVIDIA DGX Spark (GPU GB10 Blackwell, sm_121, CPU aarch64, 128 GB de memória unificada).
- Sistema: Linux (o projeto usou Ubuntu 24.04) com Docker, NVIDIA Container Toolkit e driver com suporte a CUDA 13 ([`resultados/logs/check_env.txt`](resultados/logs/check_env.txt) registra o driver 580 em modo de compatibilidade).
- Software: git, GitHub CLI (`gh`), `sha256sum` e `tar` no hospedeiro. Dentro da imagem: Python 3.12.3, PyTorch 2.12, torchvision 0.27, numpy 2.1 (da base `nvcr.io/nvidia/pytorch:26.05-py3`) e, instalados pelo [`docker/Dockerfile`](docker/Dockerfile), `opencv-python-headless==4.11.0.86`, `lxml`, `trimesh`, `easydict` e `six`.
- Contas e acessos: conta no GitHub com acesso aos repositórios [`cerise-2026-pose-servo6dof`](https://github.com/cerise-ufg/cerise-2026-pose-servo6dof) e [`cerise-2026-servo6dof-gemeo-digital`](https://github.com/cerise-ufg/cerise-2026-servo6dof-gemeo-digital) (este último fornece o dataset); acesso à internet para baixar a imagem NGC, o Depth Anything V2 Large do Hugging Face e a ResNet-50 do torchvision. Nenhum token é necessário.
- Espaço em disco: cerca de 60 GB livres (imagem Docker de 35,5 GB; dataset de 2,6 GB compactado e 2,5 GB extraído; dataset convertido de 2,6 GB; Releases de até 4,9 GB; cerca de 0,9 GB por rodada de treino completo).

**Tabela: Tempos medidos no GB10 compartilhado com outros serviços (fonte: logs em [`resultados/logs/`](resultados/logs) e [`resultados/runs/`](resultados/runs))**

| Passo | Tempo medido |
|---|---|
| Conversão das imagens (6 processos) | 23 s |
| Profundidade Depth Anything V2 (10.000 recortes) | cerca de 16 imagens/s, ou 10 min |
| Etapa 1, DepthNet, 5 épocas sobre 3.151 poses | 10,5 min (transferência) e 19,2 min (ImageNet) |
| Etapa 2, modelo completo, transferência, 20 épocas | 61,8 min |
| Etapa 2, linha de base ImageNet, 12 épocas | 52,5 min |
| Avaliação no teste, robustez e latência | cerca de 25 min |

## Obtenção do código e dos dados

1. Autentique o GitHub CLI e clone o repositório:

```bash
gh auth login
gh repo clone cerise-ufg/cerise-2026-pose-servo6dof
cd cerise-2026-pose-servo6dof
```

2. Baixe os artefatos do Release `artefatos-2026-10-26`, confira os SHA-256 e extraia na raiz do repositório. Cada `.tar` já traz os caminhos relativos (`pesos/`, `runs/<rodada>/`, [`inferencia/modelo/`](inferencia/modelo)). O mínimo para treinar é `pesos2025_realsense_full_checkpoint.tar`, que contém o checkpoint de 2025 usado na transferência. Os `ckpt_*.tar` trazem os checkpoints originais, para comparação, e `pose_servo6dof_modelo_final.tar` traz o modelo final para inferência:

```bash
gh release download artefatos-2026-10-26 -R cerise-ufg/cerise-2026-pose-servo6dof -D _releases
cd _releases && sha256sum -c SHA256SUMS.txt && cd ..
for f in _releases/*.tar; do tar -xf "$f"; done
```

3. Baixe o dataset sintético (versão cinza, a usada em todos os resultados) do Release `dados-2026-10-26` do repositório do gêmeo digital. Extraia as duas partes na mesma pasta e confira arquivo por arquivo com o manifesto:

```bash
gh release download dados-2026-10-26 -R cerise-ufg/cerise-2026-servo6dof-gemeo-digital \
  -D "$HOME/dados_gemeo" -p 'dataset_cinza.*' -p SHA256SUMS.txt
cd "$HOME/dados_gemeo"
sha256sum -c --ignore-missing SHA256SUMS.txt
tar -xf dataset_cinza.part01.tar && tar -xf dataset_cinza.part02.tar
sha256sum -c --quiet dataset_cinza.manifest.sha256
cd -
```

A pasta `dataset_cinza/` deve ter 35.009 arquivos: 5.000 JSON, 30.000 PNG (RGB, profundidade e segmentação das câmeras frontal e diagonal) e os auxiliares. No Windows, use `Get-FileHash -Algorithm SHA256 <arquivo>` e o `tar` nativo.

## Preparação do container de treino

1. Construa a imagem. O [`docker/Dockerfile`](docker/Dockerfile) parte de `nvcr.io/nvidia/pytorch:26.05-py3` e instala as dependências do código de 2025. Use `sudo` se o seu usuário não estiver no grupo `docker`:

```bash
docker build -t cerise26_pose:latest -f docker/Dockerfile docker/
```

2. Abra um shell no container. As montagens reproduzem as de [`scripts/spark/rodar.sh`](scripts/spark/rodar.sh): o repositório em `/workspace/pose`, os dados convertidos em `/dados`, o dataset (somente leitura) em `/dados/dataset`, as rodadas em `/runs` e os caches do Hugging Face e do torchvision no hospedeiro, para baixar os pesos uma única vez:

```bash
mkdir -p dados runs pesos "$HOME/.cache/huggingface" "$HOME/.cache/torch"
export DATASET="$HOME/dados_gemeo/dataset_cinza"
docker run --rm -it --gpus all --ipc=host --memory=32g --memory-swap=32g \
  -e HPE_HRNET_W32=/workspace/pose/pesos/hrnet_w32-36af842e_roc.pth -e PYTHONUNBUFFERED=1 \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$HOME/.cache/torch":/root/.cache/torch \
  -v "$PWD":/workspace/pose -v "$PWD/dados":/dados -v "$DATASET":/dados/dataset:ro \
  -v "$PWD/runs":/runs -w /workspace/pose cerise26_pose:latest bash
```

3. Dentro do container, confira o ambiente. A linha `cuda True` deve mostrar a sua GPU; `timm MISSING` é esperado, porque o código não usa o `timm`:

```bash
python docker/check_env.py
```

Todos os comandos das seções seguintes rodam dentro desse shell, a partir de `/workspace/pose`. Para treinos longos, use `docker run -d --name <nome> ... cerise26_pose:latest python ...`, como faz [`scripts/spark/rodar.sh`](scripts/spark/rodar.sh), e acompanhe com `docker logs -f <nome>`.

## Pesos de terceiros e verificação da geometria

1. Baixe o Depth Anything V2 Large (arquivo `depth_anything_v2_vitl.pth` do repositório `depth-anything/Depth-Anything-V2-Large` do Hugging Face, licença CC-BY-NC-4.0). O script imprime o caminho do arquivo no cache:

```bash
python scripts/baixar_pesos_da.py vitl
```

2. Somente para a linha de base ImageNet, obtenha a HRNet-W32 pré-treinada no ImageNet: baixe o arquivo `hrnet_w32-36af842e.pth` pelo link do repositório oficial HRNet/HRNet-Image-Classification, salve-o como `pesos/hrnet_w32-36af842e_roc.pth` e compare o SHA-256 com o do arquivo usado no projeto, registrado no [`README.md`](README.md) (`12682d70…a4c6`). Ao treinar a DepthNet ImageNet, o terminal deve mostrar `Loading hrnet pretrained weights (ImageNet) from ...`; sem essa linha, a HRNet começa com pesos aleatórios. A ResNet-50 ImageNet é baixada sozinha pelo torchvision no primeiro treino.

3. Verifique a geometria. O script projeta os keypoints da cinemática direta do URDF versionado ([`servo6dof/servo6dof.urdf`](servo6dof/servo6dof.urdf)) sobre as imagens e mede quantos caem sobre a máscara do robô:

```bash
python scripts/verificar_geometria.py --dataset /dados/dataset --saida /runs/geometria
```

## Conversão do dataset

1. Converta o dataset. O conversor calcula os keypoints 2D e 3D, a matriz K e os ângulos `juntas_reais`, divide as poses em 80/10/10 por id (semente 2026) e grava os recortes 256×256. Passe sempre o `crop.json` da execução original: ele fixa os recortes por câmera, que fazem parte da entrada do modelo:

```bash
python scripts/converter_dataset.py --dataset /dados/dataset --saida /dados/final --workers 6 \
  --crop_json resultados/dados_convertidos/final/crop.json
```

2. Calcule a profundidade relativa do Depth Anything V2 para cada recorte (gera `depth256_{treino,val,teste}.npy`):

```bash
python scripts/precomputar_profundidade.py --dados /dados/final --bs 16
```

## Treino com transferência e linha de base

O [`scripts/treinar.py`](scripts/treinar.py) implementa as duas etapas de 2025. Na etapa 1, a DepthNet aprende a profundidade da raiz (perda L1). Na etapa 2, o modelo completo (ResNet-50, camada integral, DepthNet da etapa 1, regressão iterativa das juntas e cinemática direta) é treinado com as perdas do estágio sintético de 2025. Com `--init 2025`, os tensores do checkpoint de 2025 cujas formas casam são carregados. Com `--init imagenet`, só os backbones vêm do ImageNet. Os argumentos abaixo são os registrados na primeira linha de cada `resultados/runs/<rodada>/treino.log`. Use pastas `--saida` novas: o script acrescenta linhas a `treino.log` e `curvas.csv` já existentes.

1. Treine a etapa 1 (DepthNet) com os pesos de 2025 e com ImageNet:

```bash
python scripts/treinar.py --dados /dados/final --saida /runs/repro_etapa1_transfer --etapa depthnet \
  --init 2025 --ckpt_2025 /workspace/pose/pesos/realsense_full_checkpoint.pk \
  --epocas 5 --inicio_decaimento 2 --workers 3
python scripts/treinar.py --dados /dados/final --saida /runs/repro_etapa1_imagenet --etapa depthnet \
  --init imagenet --epocas 5 --inicio_decaimento 2 --workers 3
```

2. Treine a etapa 2 por transferência (20 épocas) e a linha de base (12 épocas, o orçamento obtido na execução original; use 20 para comparar no mesmo número de épocas):

```bash
python scripts/treinar.py --dados /dados/final --saida /runs/repro_full_transfer --etapa full \
  --init 2025 --rootnet /runs/repro_etapa1_transfer/melhor.pt --epocas 20 --bs 32 --workers 4
python scripts/treinar.py --dados /dados/final --saida /runs/repro_full_imagenet --etapa full \
  --init imagenet --rootnet /runs/repro_etapa1_imagenet/melhor.pt --epocas 12 --bs 32 --workers 4
```

3. Para o experimento de generalização entre câmeras, treine só com a câmera frontal:

```bash
python scripts/treinar.py --dados /dados/final --saida /runs/repro_xcam_frontal --etapa full --init 2025 \
  --rootnet /runs/repro_etapa1_transfer/melhor.pt --epocas 10 --bs 32 --workers 3 \
  --cams_treino frontal --cams_val frontal
```

Cada rodada grava `melhor.pt` (melhor época na validação), `ultimo.pt`, `curvas.csv`, `treino.log` e, na transferência, `transfer_info.json`. Para continuar um treino interrompido, use `--retomar <ultimo.pt> --epoca_inicial <épocas feitas>`.

![Curvas por época da etapa 2: perda de treino e métricas de validação da transferência e da linha de base (resultado esperado)](evidencias/curvas_treino.png)

*Figura: Curvas por época da etapa 2: perda de treino e métricas de validação da transferência e da linha de base (resultado esperado).*

## Avaliação no conjunto de teste

1. Avalie os três modelos nas 1.000 imagens do teste (500 poses × 2 câmeras). `--robustez` aplica as sete perturbações a um subconjunto fixo de 400 imagens, recalculando a profundidade sobre a imagem perturbada, e `--latencia` mede o Depth Anything V2 e o HPE com lote 1:

```bash
python scripts/avaliar.py --dados /dados/final --ckpt /runs/repro_full_transfer/melhor.pt \
  --saida /runs/repro_full_transfer/aval --robustez --n_robustez 400 --latencia
python scripts/avaliar.py --dados /dados/final --ckpt /runs/repro_full_imagenet/melhor.pt \
  --saida /runs/repro_full_imagenet/aval --robustez --n_robustez 400
python scripts/avaliar.py --dados /dados/final --ckpt /runs/repro_xcam_frontal/melhor.pt \
  --saida /runs/repro_xcam_frontal/aval
```

Cada avaliação grava `metricas_teste.json` (todas as câmeras e cada câmera separadamente) e `predicoes_teste.npz`. Para conferir os checkpoints originais sem treinar, troque `--ckpt` por `/runs/final_full_transfer/melhor.pt` (ou `final_full_imagenet`, `xcam_frontal_transfer`) e use outra pasta `--saida`.

2. Gere as tabelas no formato do relatório. Com `resultados` como argumento, o script reproduz [`resultados/tabelas_relatorio.md`](resultados/tabelas_relatorio.md). Para as suas rodadas, copie os JSON com os nomes esperados:

```bash
mkdir -p /runs/minhas_tabelas
cp /runs/repro_full_transfer/aval/metricas_teste.json /runs/minhas_tabelas/metricas_teste_transfer.json
cp /runs/repro_full_imagenet/aval/metricas_teste.json /runs/minhas_tabelas/metricas_teste_imagenet.json
cp /runs/repro_xcam_frontal/aval/metricas_teste.json /runs/minhas_tabelas/metricas_teste_xcam.json
python scripts/tabelas.py /runs/minhas_tabelas
```

O `scripts/figuras.py --res <pasta> --out <pasta>` gera as figuras, desde que a pasta siga a mesma convenção de nomes de [`resultados/`](resultados) (`curvas_full_transfer.csv`, `predicoes_teste_transfer.npz` etc.).

![Predição do modelo de transferência (vermelho) sobre a verdade (verde) em três poses do teste, nas duas câmeras](evidencias/predicoes_vs_gt_transfer.png)

*Figura: Predição do modelo de transferência (vermelho) sobre a verdade (verde) em três poses do teste, nas duas câmeras.*

## Exportação e módulo de inferência

1. Empacote o checkpoint para inferência. O pacote reúne pesos, hiperparâmetros, recortes, intrínsecos e métricas de teste:

```bash
python scripts/exportar_modelo.py --ckpt /runs/repro_full_transfer/melhor.pt --dados /dados/final \
  --saida /runs/pacote/pose_servo6dof.pt --metricas /runs/repro_full_transfer/aval/metricas_teste.json
```

2. Teste o `PosePredictor` pela linha de comando com uma imagem do teste versionada no repositório. Para usar o modelo final do Release, aponte `--ckpt` para `inferencia/modelo/pose_servo6dof.pt`:

```bash
python inferencia/predict.py --ckpt inferencia/modelo/pose_servo6dof.pt \
  --imagem resultados/imagens_teste/002079_camera_frontal_image.png --camera frontal \
  --saida /runs/overlay_002079.png --repeticoes 30
python scripts/latencia_predictor.py inferencia/modelo/pose_servo6dof.pt /dados/dataset /runs/latencia.json
```

3. Use a classe em Python. A entrada é uma imagem RGB (não BGR) 1280×720 de uma das câmeras fixas do estúdio:

```python
import sys; sys.path.insert(0, "/workspace/pose/inferencia")
import cv2
from predict import PosePredictor
p = PosePredictor("/workspace/pose/inferencia/modelo/pose_servo6dof.pt", device="cuda")
rgb = cv2.imread("resultados/imagens_teste/002079_camera_frontal_image.png")[..., ::-1].copy()
out = p.predict(rgb, camera="frontal")
print(out["joints"], out["latency_ms"])  # joint_1..joint_5 em rad; keypoints_2d/3d também
```

Para só usar o modelo, sem treinar, bastam o `pose_servo6dof_modelo_final.tar`, o Depth Anything V2 e a imagem Docker: retire a montagem do dataset do `docker run` e siga [`inferencia/INFERENCIA.md`](inferencia/INFERENCIA.md). Depois do primeiro download, `-e HF_HUB_OFFLINE=1` evita acessos ao Hugging Face.

![Saída esperada do predict.py: esqueleto previsto sobre a imagem 002079 da câmera frontal](evidencias/inferencia_exemplo_frontal_002079.png)

*Figura: Saída esperada do predict.py: esqueleto previsto sobre a imagem 002079 da câmera frontal.*

## Verificação do resultado

- Geometria: 99,9% (frontal) e 99,6% (diagonal) dos keypoints dentro do quadro caem sobre a máscara do robô ([`resultados/verificacao_geometria.json`](resultados/verificacao_geometria.json)).
- Conversão: `ids completos: 5000`, recortes `frontal [-38.7, -162.5, 1045.0]` e `diagonal [122.5, -76.9, 846.6]`, e 8.000/1.000/1.000 imagens em treino, validação e teste ([`resultados/logs/conversao_final.txt`](resultados/logs/conversao_final.txt)).
- Transferência: o log mostra `2361/2366 tensores (98.1% dos parametros) carregados`; só `fc_pose_1.weight`, `decpose.weight` e `decpose.bias` mudam de forma (8 juntas do Panda para 5).
- Etapa 1: erro de profundidade da raiz de 71 mm na validação após 5 épocas ([`resultados/runs/etapa1_depthnet_transfer_v2/curvas.csv`](resultados/runs/etapa1_depthnet_transfer_v2/curvas.csv)).
- Etapa 2, época 20: erro 3D de 54,1 mm e MAE de 11,86° na validação ([`resultados/runs/final_full_transfer/curvas.csv`](resultados/runs/final_full_transfer/curvas.csv)).
- Inferência na pose 002079 (frontal): `joints_graus` igual a -50,92; 1,0; -20,08; -78,53; 0,2, contra a verdade -43,23; 0,05; -17,5; -69,18; -7,69 ([`resultados/logs/demo_predict.log`](resultados/logs/demo_predict.log)). `PosePredictor` com 162,5 ms de média (6,15 FPS) e 2,94 GB de GPU ([`resultados/latencia_predictor.json`](resultados/latencia_predictor.json)).

**Tabela: Resultados de referência no teste ([`resultados/metricas_teste_transfer.json`](resultados/metricas_teste_transfer.json), `metricas_teste_imagenet.json` e `metricas_teste_xcam.json`)**

| Métrica | Transferência | Linha de base |
|---|---|---|
| MAE médio das 5 juntas | 12,68° | 13,51° |
| MAE por junta (1 a 5) | 4,19 / 3,04 / 3,91 / 10,52 / 41,73° | 7,20 / 3,87 / 5,41 / 9,25 / 41,83° |
| Erro 3D médio / mediano | 58,7 / 41,1 mm | 88,3 / 65,7 mm |
| Acurácia ADD a 60 mm / AUC-ADD 0–100 mm | 68,0% / 0,471 | 32,8% / 0,265 |
| PCK 2D a 5 / 10 / 20 px | 30,1 / 61,7 / 87,1% | 19,9 / 43,6 / 71,4% |
| Robustez: sem perturbação, ruído σ = 0,05, oclusão de 30% | 57,4 / 99,0 / 94,3 mm | 88,2 / 109,7 / 114,5 mm |
| Treino só com a frontal: erro 3D na frontal / diagonal | 140,4 / 418,5 mm | — |

Ao refazer as etapas com o dataset completo, espere valores próximos, mas não idênticos (ver a próxima seção). A avaliação dos checkpoints do Release só foi executada no GB10; em outra GPU, pequenas diferenças podem vir da profundidade recalculada.

## Problemas comuns e soluções

- Código de 2025 com dois erros: na variante MyNet, a profundidade da raiz era calculada depois da camada integral, que recebia `root_trans = 0`; e `train_full.py` chamava `optimizer.step()` duas vezes por iteração. Os dois já estão corrigidos no repositório ([`hpe2025/models/full_net.py`](hpe2025/models/full_net.py), trecho marcado `[porte 2026] CORRECAO`, e um único `opt.step()` em [`scripts/treinar.py`](scripts/treinar.py)). Não treine com o código de [`legado_2025/`](legado_2025), mantido sem alterações como registro.
- Depth Anything V2 Large: vem do Hugging Face no primeiro uso. Sem internet no container, rode [`scripts/baixar_pesos_da.py`](scripts/baixar_pesos_da.py) numa máquina conectada e copie o cache para `$HOME/.cache/huggingface`. O aviso de requisições não autenticadas pode ser ignorado.
- Latência maior que a de referência, ou instável: as medidas foram feitas com a GPU compartilhada com outros serviços. A decomposição de `avaliar.py --latencia` (213,8 ms: 138,9 do Depth Anything V2 e 74,9 do HPE) foi medida com um treino rodando em paralelo. Meça com a GPU livre. O Depth Anything V2 ViT-L responde por cerca de 65% do custo.
- MAE de cerca de 42° na junta 5: é esperado e não indica erro de execução. O keypoint da garra fica sobre o eixo da rotação da garra, então essa junta não é observável pelos keypoints atuais, e a `joint_6` não é estimada. Corrigir exige novos keypoints (pontas das pinças) e novo treino.
- Métricas ruins com os checkpoints do Release: confira se a conversão usou `--crop_json resultados/dados_convertidos/final/crop.json` e o dataset cinza, não a variante laranja.
- Falta de memória: reduza `--bs` e `--workers`. Não rode dois treinos completos ao mesmo tempo; na execução original, a linha de base em paralelo foi interrompida por falta de memória ([`resultados/runs/final_full_imagenet_interrompido_memoria/`](resultados/runs/final_full_imagenet_interrompido_memoria)).
- Diferenças na etapa 1 e no experimento entre câmeras: na execução original, os dois usaram a conversão `parcial2`, com as 3.151 poses de treino disponíveis naquele momento ([`resultados/dados_convertidos/parcial2/`](resultados/dados_convertidos/parcial2)). Para repetir exatamente a etapa 2, use `--rootnet /runs/etapa1_depthnet_transfer_v2/melhor.pt` (ou `etapa1_depthnet_imagenet_v2`), extraído de `ckpt_etapa1_depthnet_v2.tar`.
- Arquivos criados pelo container pertencem ao root: rode `sudo chown -R $(id -u):$(id -g) dados runs` no hospedeiro. Não é preciso rodar [`servo6dof/gerar_urdf.sh`](servo6dof/gerar_urdf.sh) (usa caminhos da máquina original e a imagem ROS 2 do gêmeo): o URDF já está versionado.
- x86_64 com GPU NVIDIA comum: a imagem base NGC também é publicada para x86_64 e o código não depende de aarch64, mas essa combinação não foi testada. Em GPUs sem bfloat16 (anteriores à Ampere), desligue a precisão mista do treino com `--sem_amp` (também não testado).

## Referências do repositório

- [`README.md`](README.md) e [`RELEASES.md`](RELEASES.md): visão geral, comandos e artefatos com SHA-256; [`RELATORIO_SECAO.md`](RELATORIO_SECAO.md): metodologia e discussão.
- [`docker/Dockerfile`](docker/Dockerfile) e [`docker/check_env.py`](docker/check_env.py): imagem e verificação do ambiente.
- [`scripts/converter_dataset.py`](scripts/converter_dataset.py), [`scripts/precomputar_profundidade.py`](scripts/precomputar_profundidade.py) e [`scripts/verificar_geometria.py`](scripts/verificar_geometria.py): dados e geometria.
- [`scripts/treinar.py`](scripts/treinar.py), [`pose26/modelo.py`](pose26/modelo.py) e [`pose26/dados.py`](pose26/dados.py): treino, modelo, transferência e augmentations.
- [`scripts/avaliar.py`](scripts/avaliar.py), [`pose26/metricas.py`](pose26/metricas.py), [`scripts/tabelas.py`](scripts/tabelas.py) e [`scripts/figuras.py`](scripts/figuras.py): avaliação, tabelas e figuras.
- [`inferencia/predict.py`](inferencia/predict.py), [`inferencia/INFERENCIA.md`](inferencia/INFERENCIA.md) e [`scripts/exportar_modelo.py`](scripts/exportar_modelo.py): módulo de inferência.
- [`hpe2025/`](hpe2025) (porte do código de 2025), [`servo6dof/geometria.py`](servo6dof/geometria.py) e [`servo6dof/servo6dof.urdf`](servo6dof/servo6dof.urdf) (cinemática e câmeras).
- [`resultados/`](resultados): métricas, curvas, logs e metadados de todas as rodadas.
