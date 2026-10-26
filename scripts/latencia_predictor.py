"""Latencia ponta a ponta do PosePredictor (imagem 1280x720 -> juntas), lote 1, no GB10."""
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, "/workspace/pose/inferencia")
from predict import PosePredictor  # noqa: E402

ck = sys.argv[1]
dataset = sys.argv[2]
saida = sys.argv[3]
z = np.load("/dados/final/index.npz")
ids = z["teste_ids"][::2][:40]
cams = ["frontal", "diagonal"]
imgs = [(cv2.imread(os.path.join(dataset, f"{int(i):06d}_camera_{cams[k % 2]}_image.png"))[..., ::-1].copy(), cams[k % 2])
        for k, i in enumerate(ids)]
p = PosePredictor(ck)
torch.cuda.reset_peak_memory_stats()
for k in range(20):
    p.predict(*imgs[k % len(imgs)])
lat = []
for k in range(100):
    out = p.predict(*imgs[k % len(imgs)])
    lat.append(out["latency_ms"])
r = {"n": len(lat), "media_ms": float(np.mean(lat)), "mediana_ms": float(np.median(lat)), "p95_ms": float(np.percentile(lat, 95)),
     "fps": 1000.0 / float(np.mean(lat)), "gpu_mem_pico_GB": torch.cuda.max_memory_allocated() / 1e9,
     "descricao": "PosePredictor.predict(rgb 1280x720) ponta a ponta: recorte+resize (CPU) + Depth Anything V2 vitl + HPE; lote 1; GB10 compartilhado",
     "data": time.strftime("%Y-%m-%d %H:%M:%S")}
json.dump(r, open(saida, "w"), indent=1)
print(json.dumps(r, indent=1))
