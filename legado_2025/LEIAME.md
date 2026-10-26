# Código de 2025 (referência)

Cópia do código de estimação de pose de braço robótico desenvolvido em 2025 na dissertação de mestrado do autor,
trazida da pasta `Mestrado/` do Google Drive do autor (`G:/Meu Drive/Mestrado/`) em 26/10/2026. É a base a partir da
qual o modelo foi portado para o servo6dof em 2026 (pasta `hpe2025/` na raiz do repositório). O código foi escrito
para o Google Colab (caminhos fixos em `/content/drive/MyDrive/Mestrado/...`, CUDA 11) e é mantido aqui sem alterações,
como registro; o código que roda no DGX Spark é o de `hpe2025/`, `pose26/` e `scripts/`.

| pasta | conteúdo |
|---|---|
| `MyNet/` | **variante final de 2025** (HPE + ramo de profundidade Depth Anything V2): `lib/` (modelos `full_net.py`, `depth_net.py`, `depth_extractor.py`, datasets DREAM, métricas, BPnP, renderização), `configs/` (Panda, Kuka, Baxter; etapas sintética e autossupervisionada), `scripts/` (`train_depthnet.py`, `train_full.py`, `train_sim2real.py`, `train_autoss.py`, `test.py`, `save_maks.py`), `bench_colab.py`. |
| `MyNet/experiments/` | configurações (`config.yaml`) e **resultados por época** dos experimentos de 2025 (`results*.csv`) e os scripts de gráficos usados na dissertação. O valor de 75,4% de acurácia@60 mm citado no relatório de 2025 está em `experiments/panda/realsense_full/results_formatado_complete.csv` (época 355). |
| `Holistic-Robot-Pose-Estimation/` | cópia de trabalho do repositório HPE adaptada em 2025 (configs, `lib/`, `scripts/`, e a pasta `mestrado/` com `ctrnet` e `urdfpytorch`). |

Os checkpoints do experimento `MyNet/experiments/panda/realsense_full/` (`checkpoint.pk`, época 440, usado na
transferência de 2026, e `best_model_auc.pk`, época 207) estão nos Releases do repositório (ver `../RELEASES.md`).

## O que foi retirado desta cópia e como obter

As duas cópias do repositório oficial do **Depth Anything V2** que existiam dentro desta árvore não estão
versionadas, por serem clones de código de terceiros sem alterações nos módulos usados (na cópia clonada, o nome da
pasta foi trocado nos `README.md` e parte dos arquivos — imagens de exemplo e arquivos de `metric_depth/` — foi
apagada; a cópia em `MyNet/` é completa):

- `Holistic-Robot-Pose-Estimation/mestrado/Depth_Anything/` (clone git, 46 MB de histórico);
- `MyNet/lib/models/Depth_Anything/` (cópia com `assets/` e `metric_depth/`).

Ambas correspondem ao commit `e5a2732d3ea2cddc081d7bfd708fc0bf09f812f1` (22/01/2025) de
`https://github.com/DepthAnything/Depth-Anything-V2` (código sob Apache-2.0). Para restaurar:

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2 legado_2025/MyNet/lib/models/Depth_Anything
git -C legado_2025/MyNet/lib/models/Depth_Anything checkout e5a2732d3ea2cddc081d7bfd708fc0bf09f812f1
# os pesos (ex.: depth_anything_v2_vitl.pth) vem do Hugging Face depth-anything/Depth-Anything-V2-Large
# e ficam em lib/models/Depth_Anything/checkpoints/ (caminho usado por lib/dataset/dream.py)
```

Também foi omitido `MyNet/experiments/panda/realsense_full/tempCodeRunnerFile.py` (arquivo vazio gerado pelo editor).

## Origem do código

O HPE é o código de *Real-time Holistic Robot Pose Estimation with Unknown States* (ECCV 2024),
repositório `Oliverbansk/Holistic-Robot-Pose-Estimation`, que não declara licença. As alterações de 2025 (ramo de
profundidade, extrator de características de profundidade, configurações e scripts de treino) são do autor.
