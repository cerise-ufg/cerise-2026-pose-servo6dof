"""Converte o novo dataset do servo6dof (Atividade 7) para o formato usado pelo modelo HPE portado.

Para cada pose (id) e cada camera (frontal, diagonal):
  - angulos reais joint_1..joint_5 (rad) e garra joint_6 (comandada)
  - keypoints 3D no frame otico da camera (m) e 2D na imagem original (px), via FK do URDF
  - K original (1280x720), T_camera<-base (pose da base do robo no frame da camera)
  - recorte FIXO por camera (quadrado, calculado so com o split de treino) redimensionado para 256x256
    -> K do recorte; a mesma janela e usada em treino, teste e inferencia (sem usar GT no teste)
Split por id de pose 80/10/10 com semente fixa sobre os ids planejados 1..5000 (estavel mesmo com
o dataset parcial). As duas cameras de uma pose ficam sempre no mesmo split.

Saida (--saida): index.npz, crop.json, rgb256_{split}.u8 (memmap N x 256 x 256 x 3, RGB)
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from servo6dof import geometria as G  # noqa: E402

CAMS = ["frontal", "diagonal"]
RES = 256
N_PLANEJADO = 5000
SEMENTE = 2026


def split_de_ids(n_total=N_PLANEJADO, semente=SEMENTE):
    perm = np.random.default_rng(semente).permutation(np.arange(1, n_total + 1))
    n_tr, n_va = int(0.8 * n_total), int(0.1 * n_total)
    sp = {}
    for i in perm[:n_tr]:
        sp[int(i)] = "treino"
    for i in perm[n_tr:n_tr + n_va]:
        sp[int(i)] = "val"
    for i in perm[n_tr + n_va:]:
        sp[int(i)] = "teste"
    return sp


def ids_completos(d):
    fs = set(os.listdir(d))
    ids = []
    for f in fs:
        if f.endswith(".json") and len(f) == 11:
            i = int(f[:6])
            ok = all(f"{i:06d}_camera_{c}_{m}.png" in fs for c in CAMS
                     for m in ("image", "depth", "segmentation_labels_map"))
            if ok:
                ids.append(i)
    return sorted(ids)


def crop_K(K, box):
    x0, y0, S = box
    s = RES / S
    Kc = K.copy()
    Kc[0, 0] *= s; Kc[1, 1] *= s
    Kc[0, 2] = (K[0, 2] - x0) * s
    Kc[1, 2] = (K[1, 2] - y0) * s
    return Kc


def recortar(img, box):
    """recorte quadrado (pode sair da imagem: preenche com 0) e redimensiona para RES."""
    x0, y0, S = [int(round(v)) for v in box]
    H, W = img.shape[:2]
    out = np.zeros((S, S, 3), np.uint8)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + S), min(H, y0 + S)
    if xb > xa and yb > ya:
        out[ya - y0:yb - y0, xa - x0:xb - x0] = img[ya:yb, xa:xb]
    return cv2.resize(out, (RES, RES), interpolation=cv2.INTER_AREA)


_ARGS = {}


def _trabalho(item):
    i, cam, box = item
    img = cv2.imread(os.path.join(_ARGS["dataset"], f"{i:06d}_camera_{cam}_image.png"), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return recortar(img[..., ::-1], box)  # RGB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--urdf", default=G.URDF_PADRAO)
    ap.add_argument("--crop_json", default=None, help="reusar recortes ja calculados")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max_ids", type=int, default=0)
    a = ap.parse_args()
    _ARGS["dataset"] = a.dataset
    os.makedirs(a.saida, exist_ok=True)
    t0 = time.time()

    cadeia = G.ler_cadeia(a.urdf)
    K = G.K_matrix()
    T_cw = {c: G.T_camera_mundo(c) for c in CAMS}
    ids = ids_completos(a.dataset)
    if a.max_ids:
        ids = ids[:a.max_ids]
    sp = split_de_ids()
    print(f"ids completos: {len(ids)} (min {ids[0]}, max {ids[-1]})")

    reg = []
    for i in ids:
        meta = json.load(open(os.path.join(a.dataset, f"{i:06d}.json")))
        q = np.array([meta["juntas_reais"][j] for j in G.JUNTAS], dtype=np.float64)
        garra = float(meta.get("garra_comandada", {}).get("joint_6", 0.0))
        kw = G.fk_keypoints_np(q, cadeia)
        for ci, cam in enumerate(CAMS):
            kc = (np.c_[kw, np.ones(len(kw))] @ T_cw[cam].T)[:, :3]
            uv = G.projetar(K, kc)
            reg.append(dict(id=i, cam=ci, q=q, garra=garra, kp3d=kc, kp2d=uv, split=sp[i]))

    # ---- recorte fixo por camera (somente treino) ----
    if a.crop_json and os.path.exists(a.crop_json):
        crops = json.load(open(a.crop_json))
    else:
        crops = {}
        for ci, cam in enumerate(CAMS):
            uv = np.concatenate([r["kp2d"] for r in reg if r["cam"] == ci and r["split"] == "treino"])
            lo0 = np.percentile(uv, 0.5, axis=0)
            hi0 = np.percentile(uv, 99.5, axis=0)
            # limita a janela a area da imagem (keypoints fora do quadro nao sao visiveis; manter a regiao
            # preenchida so reduziria a resolucao util do robo na entrada 256x256)
            lo = np.maximum(lo0, [0, 0])
            hi = np.minimum(hi0, [G.W, G.H])
            c = (lo + hi) / 2
            S = float(max(hi - lo) * 1.08)
            crops[cam] = [float(c[0] - S / 2), float(c[1] - S / 2), S]
            crops[cam + "_info"] = {"uv_p0.5": lo0.tolist(), "uv_p99.5": hi0.tolist(), "janela_limitada_a_imagem": [lo.tolist(), hi.tolist()],
                                    "margem": 0.08, "n_amostras_treino": int(len(uv) / 7)}
        crops["resolucao_entrada"] = RES
    json.dump(crops, open(os.path.join(a.saida, "crop.json"), "w"), indent=2)
    print("recortes:", {c: [round(v, 1) for v in crops[c]] for c in CAMS})

    Kc = {cam: crop_K(K, crops[cam]) for cam in CAMS}
    for r in reg:
        cam = CAMS[r["cam"]]
        x0, y0, S = crops[cam]
        r["kp2d_crop"] = (r["kp2d"] - np.array([x0, y0])) * (RES / S)
        r["Kc"] = Kc[cam]

    idx = {}
    for split in ["treino", "val", "teste"]:
        rs = [r for r in reg if r["split"] == split]
        if not rs:
            continue
        mm = np.lib.format.open_memmap(os.path.join(a.saida, f"rgb256_{split}.npy"), mode="w+", dtype=np.uint8, shape=(len(rs), RES, RES, 3))
        itens = [(r["id"], CAMS[r["cam"]], crops[CAMS[r["cam"]]]) for r in rs]
        with Pool(a.workers) as pool:
            for k, im in enumerate(pool.imap(_trabalho, itens, chunksize=16)):
                mm[k] = im
        mm.flush()
        del mm
        idx[split] = dict(
            ids=np.array([r["id"] for r in rs]), cam=np.array([r["cam"] for r in rs]),
            q=np.stack([r["q"] for r in rs]).astype(np.float32), garra=np.array([r["garra"] for r in rs], np.float32),
            kp3d=np.stack([r["kp3d"] for r in rs]).astype(np.float32), kp2d=np.stack([r["kp2d"] for r in rs]).astype(np.float32),
            kp2d_crop=np.stack([r["kp2d_crop"] for r in rs]).astype(np.float32),
            Kc=np.stack([r["Kc"] for r in rs]).astype(np.float32))
        print(f"{split}: {len(rs)} amostras ({len(set(idx[split]['ids'].tolist()))} poses)  t={time.time()-t0:.0f}s")
    flat = {f"{s}_{k}": v for s, d in idx.items() for k, v in d.items()}
    np.savez(os.path.join(a.saida, "index.npz"), K=K, T_cw_frontal=T_cw["frontal"], T_cw_diagonal=T_cw["diagonal"],
             keypoints=np.array(G.KEYPOINTS), juntas=np.array(G.JUNTAS), **flat)
    resumo = {"n_ids": len(ids), "splits": {s: int(len(d["ids"])) for s, d in idx.items()},
              "poses_por_split": {s: int(len(set(d["ids"].tolist()))) for s, d in idx.items()},
              "semente_split": SEMENTE, "cameras": CAMS, "keypoints": G.KEYPOINTS, "tempo_s": round(time.time() - t0, 1)}
    json.dump(resumo, open(os.path.join(a.saida, "resumo_conversao.json"), "w"), indent=2)
    print(json.dumps(resumo, indent=1))


if __name__ == "__main__":
    main()
