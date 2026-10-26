"""Empacota o checkpoint treinado para inferencia (pesos + hiperparametros + recortes das cameras + metadados)."""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from servo6dof import geometria as G  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--dados", required=True)
ap.add_argument("--saida", required=True)
ap.add_argument("--metricas", default=None)
a = ap.parse_args()
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
crop = json.load(open(os.path.join(a.dados, "crop.json")))
pac = {
    "model_state_dict": ck["model_state_dict"], "args_modelo": ck["args_modelo"], "amp": ck.get("amp", False),
    "epoch": ck.get("epoch"), "init": ck.get("init"), "metricas_val": ck.get("metricas_val"),
    "crop": crop, "K": G.K_matrix().tolist(), "juntas": G.JUNTAS, "keypoints": G.KEYPOINTS,
    "T_camera_mundo": {c: G.T_camera_mundo(c).tolist() for c in G.CAMERAS_SDF},
    "descricao": "HPE/HoRoPose + Depth Anything V2 (modelo de 2025, MyNet) adaptado ao servo6dof - CERISE 2026, frente POSE",
}
if a.metricas and os.path.exists(a.metricas):
    pac["metricas_teste"] = json.load(open(a.metricas)).get("teste_todas_cameras")
os.makedirs(os.path.dirname(os.path.abspath(a.saida)), exist_ok=True)
torch.save(pac, a.saida)
json.dump(crop, open(os.path.join(os.path.dirname(os.path.abspath(a.saida)), "crop.json"), "w"), indent=2)
print("exportado:", a.saida, f"({os.path.getsize(a.saida)/1e6:.0f} MB)")
