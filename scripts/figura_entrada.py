"""Figura: entradas do modelo (recorte fixo 256x256 por camera + keypoints GT + profundidade Depth Anything V2)."""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from servo6dof import geometria as G  # noqa: E402
from pose26.dados import Servo6dofDataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dados", required=True)
ap.add_argument("--saida", required=True)
ap.add_argument("--split", default="teste")
a = ap.parse_args()
ds = Servo6dofDataset(a.dados, a.split)
rng = np.random.default_rng(3)
fr = [i for i in range(len(ds)) if ds.meta["cam"][ds.idx[i]] == 0]
dg = [i for i in range(len(ds)) if ds.meta["cam"][ds.idx[i]] == 1]
sel = list(rng.choice(fr, 2, replace=False)) + list(rng.choice(dg, 2, replace=False))
fig, axs = plt.subplots(2, 4, figsize=(11, 5.6))
for j, i in enumerate(sel):
    it = ds[int(i)]
    kp = it["kp2d_crop"].numpy()
    axs[0, j].imshow(it["rgb"].permute(1, 2, 0).numpy())
    for u, v in G.ARESTAS:
        axs[0, j].plot(kp[[u, v], 0], kp[[u, v], 1], "-", color="yellow", lw=1.5)
    axs[0, j].scatter(kp[:, 0], kp[:, 1], s=10, c="red", zorder=3)
    axs[0, j].set_title(f"pose {it['id']:06d} - {['frontal', 'diagonal'][it['cam']]}\nRGB 256x256 + keypoints (GT)", fontsize=8)
    axs[1, j].imshow(it["depth"][0].numpy(), cmap="inferno")
    axs[1, j].set_title("profundidade relativa (Depth Anything V2)", fontsize=8)
    for r in range(2):
        axs[r, j].set_xlim(0, 256); axs[r, j].set_ylim(256, 0); axs[r, j].axis("off")
fig.tight_layout()
fig.savefig(a.saida, dpi=150)
print("ok", a.saida)
