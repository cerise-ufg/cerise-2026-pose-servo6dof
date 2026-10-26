"""Modulo de inferencia da estimacao de pose do servo6dof (frente POSE, CERISE 2026) para o gemeo digital.

    from predict import PosePredictor
    p = PosePredictor("/workspace/pose/inferencia/modelo/pose_servo6dof.pt", device="cuda")
    out = p.predict(rgb_uint8_HxWx3, camera="frontal")   # imagem RGB 1280x720 da camera do estudio
    out["joints"]        -> {"joint_1": rad, ..., "joint_5": rad}
    out["keypoints_2d"]  -> [[u, v], ...] (7 pontos, pixels da imagem de entrada)
    out["keypoints_3d"]  -> [[x, y, z], ...] (m, frame otico da camera)
    out["latency_ms"]    -> tempo total (Depth Anything V2 + HPE), com sincronizacao da GPU

CLI de teste:
    python predict.py --ckpt modelo/pose_servo6dof.pt --imagem foto.png --camera frontal --saida overlay.png
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def _achar_repo():
    cands = [os.environ.get("POSE_REPO", ""), "/workspace/pose/repo", "/workspace/pose/src/repo",
             os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "repo"),
             os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "pose26", "modelo.py")):
            return os.path.abspath(c)
    raise RuntimeError("repositorio da frente POSE nao encontrado (defina POSE_REPO)")


REPO = _achar_repo()
for _p in (REPO, os.path.join(REPO, "hpe2025")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2  # noqa: E402
import torch  # noqa: E402

from pose26.modelo import args_2025, ativar_amp, construir_full  # noqa: E402
from pose26.dados import k_value  # noqa: E402
from depth_da import DAProfundidade  # noqa: E402
from servo6dof import geometria as G  # noqa: E402

CAMS = ["frontal", "diagonal"]


class PosePredictor:
    def __init__(self, ckpt, device="cuda", da_pesos=None):
        self.device = device
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        a = args_2025(**{k: v for k, v in ck.get("args_modelo", {}).items() if k != "aug"})
        a.imagenet_init = False
        os.environ.setdefault("HPE_HRNET_W32", "")
        os.environ["HPE_RESNET_IMAGENET"] = "0"
        self.model, _ = construir_full(a, device=device)
        self.model.load_state_dict(ck["model_state_dict"])
        if ck.get("amp", False):
            ativar_amp(self.model)
            self.model = self.model.to(memory_format=torch.channels_last)
        self.model.eval()
        crop = ck.get("crop")
        if crop is None:
            crop = json.load(open(os.path.join(os.path.dirname(os.path.abspath(ckpt)), "crop.json")))
        self.crop = {c: crop[c] for c in CAMS}
        self.res = int(crop.get("resolucao_entrada", 256))
        self.K = G.K_matrix()
        self.Kc = {}
        for c in CAMS:
            x0, y0, S = self.crop[c]
            s = self.res / S
            Kc = self.K.copy()
            Kc[0, 0] *= s; Kc[1, 1] *= s
            Kc[0, 2] = (self.K[0, 2] - x0) * s; Kc[1, 2] = (self.K[1, 2] - y0) * s
            self.Kc[c] = torch.tensor(Kc, dtype=torch.float32, device=device).unsqueeze(0)
        self.da = DAProfundidade("vitl", device, pesos=da_pesos)
        self.info = {"epoca": ck.get("epoch", -1) + 1, "init": ck.get("init"), "metricas_val": ck.get("metricas_val")}

    def _recortar(self, rgb, camera):
        H, W = rgb.shape[:2]
        if (W, H) != (G.W, G.H):  # outra resolucao: reescala para 1280x720 (intrinsecos do estudio)
            rgb = cv2.resize(rgb, (G.W, G.H), interpolation=cv2.INTER_AREA)
        x0, y0, S = [int(round(v)) for v in self.crop[camera]]
        out = np.zeros((S, S, 3), np.uint8)
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(G.W, x0 + S), min(G.H, y0 + S)
        out[ya - y0:yb - y0, xa - x0:xb - x0] = rgb[ya:yb, xa:xb]
        return cv2.resize(out, (self.res, self.res), interpolation=cv2.INTER_AREA), (W / G.W, H / G.H)

    @torch.no_grad()
    def predict(self, rgb_uint8_HxWx3, camera="frontal"):
        assert camera in CAMS, camera
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        crop, (sx, sy) = self._recortar(np.asarray(rgb_uint8_HxWx3), camera)
        x = torch.from_numpy(crop).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        d = (self.da(x) * 255).round() / 255.0
        Kc = self.Kc[camera]
        x = x.contiguous(memory_format=torch.channels_last)
        d = d.contiguous(memory_format=torch.channels_last)
        pp, prot, ptrans, puv, pdepth, puvd, pint, pfk = self.model(x, x, d, k_value(Kc), K=Kc)
        q = pp[0].float().cpu().numpy()
        kp3 = pfk[0].float().cpu().numpy()
        uv = G.projetar(self.K, kp3) * np.array([sx, sy])
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        lat = (time.perf_counter() - t0) * 1000.0
        return {"joints": {j: float(v) for j, v in zip(G.JUNTAS, q)},
                "keypoints_2d": uv.tolist(), "keypoints_3d": kp3.tolist(), "keypoint_names": list(G.KEYPOINTS),
                "camera": camera, "latency_ms": float(lat)}


def desenhar(img_rgb, uv):
    img = img_rgb.copy()
    for a, b in G.ARESTAS:
        cv2.line(img, tuple(int(v) for v in uv[a]), tuple(int(v) for v in uv[b]), (255, 60, 60), 4, cv2.LINE_AA)
    for p in uv:
        cv2.circle(img, (int(p[0]), int(p[1])), 7, (255, 255, 0), -1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--imagem", required=True)
    ap.add_argument("--camera", default="frontal", choices=CAMS)
    ap.add_argument("--saida", default=None, help="PNG com o esqueleto previsto desenhado")
    ap.add_argument("--repeticoes", type=int, default=1)
    a = ap.parse_args()
    p = PosePredictor(a.ckpt)
    rgb = cv2.imread(a.imagem, cv2.IMREAD_COLOR)[..., ::-1].copy()
    for _ in range(a.repeticoes):
        out = p.predict(rgb, a.camera)
    print(json.dumps({"joints_rad": out["joints"], "joints_graus": {k: round(np.degrees(v), 2) for k, v in out["joints"].items()},
                      "keypoints_2d": [[round(u, 1), round(v, 1)] for u, v in out["keypoints_2d"]],
                      "latency_ms": round(out["latency_ms"], 2),
                      "gpu_mem_pico_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None},
                     indent=1))
    if a.saida:
        cv2.imwrite(a.saida, desenhar(rgb, np.array(out["keypoints_2d"]))[..., ::-1])
        print("overlay salvo em", a.saida)


if __name__ == "__main__":
    main()
