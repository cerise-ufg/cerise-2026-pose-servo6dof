# Artefatos grandes (Releases)

Os checkpoints deste trabalho (arquivos > 50 MB) não são versionados no git. Eles estão publicados como *assets* do
Release **`artefatos-2026-10-26`** deste repositório, empacotados em `.tar` (sem compressão; pesos em ponto flutuante
praticamente não comprimem), cada um com menos de 1,9 GB, junto com o arquivo `SHA256SUMS.txt`.

Os caminhos dentro de cada `.tar` reproduzem a organização usada no DGX Spark (`~/cerise2026/pose/`): `inferencia/modelo/`,
`runs/<rodada>/` e `pesos/`. Extraídos na raiz do repositório, ficam onde os scripts e o `predict.py` esperam
(`runs/` e os arquivos `*.pt`/`*.pk` estão no `.gitignore`).

| arquivo | tamanho | SHA-256 | conteúdo |
|---|---|---|---|
| `pose_servo6dof_modelo_final.tar` | 445.696.000 B (425,0 MiB) | `77e0cb21691d65009bdf1631efaac12f8fb987c4bd1f838a73076028bb8d6dee` | **modelo final para inferência**: `inferencia/modelo/pose_servo6dof.pt` (pesos + hiperparâmetros + recortes e intrínsecos das câmeras + métricas de teste; exportado de `final_full_transfer/melhor.pt` por `scripts/exportar_modelo.py`) e `inferencia/modelo/crop.json`. Usado por `inferencia/predict.py`. |
| `ckpt_final_full_transfer.tar` | 445.726.720 B (425,1 MiB) | `50a99e1725631b78b777e8f6b7594c4985408788dfe3644081a708cc3c954655` | `runs/final_full_transfer/melhor.pt` — etapa 2, transferência dos pesos de 2025, melhor época na validação (20) — com `transfer_info.json`, `curvas.csv` e `treino.log`. |
| `ckpt_final_full_imagenet.tar` | 445.716.480 B (425,1 MiB) | `637c72ff8b77a86b8b47f1a7f479ff6843aa4e6ef328097503e57afffc17deb1` | `runs/final_full_imagenet/melhor.pt` — etapa 2, baseline ImageNet (12 épocas) — com `curvas.csv` e `treino.log`. |
| `ckpt_etapa1_depthnet_v2.tar` | 458.117.120 B (436,9 MiB) | `ecf0c867f27fc587dae934c55e4d509124e67119c0b87296a696cce00fcd04b5` | etapa 1 (DepthNet/RootNet): `runs/etapa1_depthnet_transfer_v2/melhor.pt` (a partir de `realsense_full_checkpoint.pk`, usada pelo modelo final) e `runs/etapa1_depthnet_imagenet_v2/melhor.pt` (usada pelo baseline), com curvas, logs e `transfer_info.json`. |
| `ckpt_xcam_frontal_transfer.tar` | 445.716.480 B (425,1 MiB) | `c203e5e0a9d0a9fb029d3dbb53493eea384eccdc00a109d901d260627fcd455b` | `runs/xcam_frontal_transfer/melhor.pt` — experimento de generalização entre câmeras (treino só com a câmera frontal; melhor na época 6) — com `transfer_info.json`, `curvas.csv` e `treino.log`. |
| `pesos2025_realsense_full_checkpoint.tar` | 1.335.439.360 B (1,24 GiB) | `5b06311d84a6316c00337ec6d3f19f6f37b309163ff12d9a06076c8d9148fa7e` | `pesos/realsense_full_checkpoint.pk` — checkpoint de 2025 (dissertação; `MyNet/experiments/panda/realsense_full/checkpoint.pk`, estado final da etapa 3 autossupervisionada no Panda/RealSense, época 440, 111,2 M parâmetros + estado do otimizador). **É o checkpoint usado na transferência.** SHA-256 do `.pk`: `5f8c1e029c62ece2853f15c00bc835383b9a04b3387732382db317e107a00750`. |
| `pesos2025_realsense_full_best_model_auc.tar` | 1.335.449.600 B (1,24 GiB) | `87b0941e5f06874e0186fcb54f60d8111f44858ecbd9607cd58f90e24d767cf9` | `pesos/best_model_auc.pk` — checkpoint de 2025 do mesmo experimento com o melhor AUC na validação (época 207). SHA-256 do `.pk`: `72b6b49b1e3560b0f5d8345ef0cac54bb993366d62fbc41e16c262b1b510cec1`. |

Total: 4.911.861.760 B (~4,9 GB) em 7 arquivos. SHA-256 do `.pt` do modelo final:
`7e14c32286d3c743a255ca197d074b75385c16536845d8dadeb73bf6825b2104`.

## Como baixar, conferir e extrair

```bash
# na raiz do repositorio clonado (substitua <organizacao> pelo dono do repositorio)
gh release download artefatos-2026-10-26 -R <organizacao>/cerise-2026-pose-servo6dof -D _releases
cd _releases && sha256sum -c SHA256SUMS.txt && cd ..
# extrair (cada tar ja traz os caminhos relativos: inferencia/modelo/, runs/<rodada>/, pesos/)
for f in _releases/*.tar; do tar -xf "$f"; done
```

No Windows (PowerShell), a conferência pode ser feita com `Get-FileHash -Algorithm SHA256 <arquivo>` e a extração
com o `tar` nativo (`tar -xf <arquivo>`). Para usar apenas o módulo de inferência basta o
`pose_servo6dof_modelo_final.tar`; para retomar ou avaliar treinos, os `ckpt_*.tar`; para refazer a transferência a
partir de 2025, o `pesos2025_realsense_full_checkpoint.tar` (o `treinar.py` procura por padrão
`/workspace/pose/pesos/realsense_full_checkpoint.pk`; use `--ckpt_2025` para outro caminho).

## O que não está nos Releases

- Pesos de terceiros (Depth Anything V2 Large, HRNet-W32 e ResNet-50 ImageNet): ver "O que não está aqui e por quê"
  no `README.md` para fonte e forma de obtenção.
- `ultimo.pt` de cada rodada e checkpoints das rodadas exploratórias (`smoke_full`, `dev_full_transfer`,
  `etapa1_depthnet_{transfer,imagenet}`, `etapa1_depthnet_transfer_v2_ckpt_ep207`, `xcam_frontal_transfer_bug_integral`,
  `xcam_frontal_transfer_depthw1`) e o pacote de desenvolvimento `inferencia/dev/pose_dev.pt`: intermediários,
  mantidos apenas no Spark (`~/cerise2026/pose/`); as métricas e os logs dessas rodadas estão em `resultados/runs/`.
- Memmaps do dataset convertido (~5,7 GB): regeneráveis (ver `README.md`).
