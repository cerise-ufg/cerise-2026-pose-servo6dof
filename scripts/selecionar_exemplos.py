"""Escolhe 3 poses do teste representativas (percentis 25/50/75 do erro 3D medio das duas cameras)."""
import json
import os
import sys

import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "resultados"
z = np.load(os.path.join(R, "predicoes_teste_transfer.npz"))
ids = np.unique(z["id"])
err = np.array([z["e3_img"][z["id"] == i].mean() for i in ids])
ordem = np.argsort(err)
sel = [int(ids[ordem[int(p * (len(ids) - 1))]]) for p in (0.25, 0.5, 0.75)]
os.makedirs(os.path.join(R, "imagens_teste"), exist_ok=True)
json.dump({"ids": sel, "criterio": "percentis 25/50/75 do erro 3D medio (2 cameras) do modelo de transferencia",
           "erros_mm": [float(err[ids == i][0] * 1000) for i in sel]}, open(os.path.join(R, "imagens_teste", "selecao.json"), "w"), indent=1)
print(" ".join(str(i) for i in sel))
