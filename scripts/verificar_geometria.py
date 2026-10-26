"""Verifica a geometria (FK do URDF + poses/intrinsecos das cameras do SDF) projetando os keypoints
sobre as imagens RGB do novo dataset. Gera a figura de overlay e um JSON com estatisticas.

uso: python scripts/verificar_geometria.py --dataset /dados/dataset --saida /out [--ids 1 2 3]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from servo6dof import geometria as G  # noqa: E402

CORES = [(255, 255, 255), (255, 80, 80), (255, 170, 0), (240, 240, 0), (0, 220, 0), (0, 200, 255), (200, 80, 255)]


def desenhar(img, uv, raio=7, esp=3):
    for a, b in G.ARESTAS:
        cv2.line(img, tuple(int(v) for v in uv[a]), tuple(int(v) for v in uv[b]), (0, 255, 255), esp, cv2.LINE_AA)
    for i, p in enumerate(uv):
        cv2.circle(img, (int(p[0]), int(p[1])), raio, CORES[i][::-1], -1, cv2.LINE_AA)
        cv2.circle(img, (int(p[0]), int(p[1])), raio, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--urdf", default=G.URDF_PADRAO)
    ap.add_argument("--ids", type=int, nargs="*", default=None)
    ap.add_argument("--n_stats", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(a.saida, exist_ok=True)
    cadeia = G.ler_cadeia(a.urdf)
    K = G.K_matrix()
    jsons = sorted(f for f in os.listdir(a.dataset) if f.endswith(".json"))
    ids = a.ids or [int(jsons[i][:6]) for i in np.linspace(0, len(jsons) - 1, 3).astype(int)]

    # ---- overlay ----
    linhas = []
    for pid in ids:
        meta = json.load(open(os.path.join(a.dataset, f"{pid:06d}.json")))
        q = np.array([meta["juntas_reais"][j] for j in G.JUNTAS])
        kw = G.fk_keypoints_np(q, cadeia)
        lin = []
        for cam in ["frontal", "diagonal"]:
            img = cv2.imread(os.path.join(a.dataset, f"{pid:06d}_camera_{cam}_image.png"))
            T = G.T_camera_mundo(cam)
            kc = (np.c_[kw, np.ones(len(kw))] @ T.T)[:, :3]
            uv = G.projetar(K, kc)
            seg = cv2.imread(os.path.join(a.dataset, f"{pid:06d}_camera_{cam}_segmentation_labels_map.png"), cv2.IMREAD_UNCHANGED)
            over = img.copy()
            if seg is not None:
                lab = seg[..., 0] if seg.ndim == 3 else seg
                rob = lab == 0  # rotulos: 1 = chao, 2 = fundo, 0 = robo (sem rotulo)
                over[rob] = (0.55 * over[rob] + 0.45 * np.array([255, 0, 255])).astype(np.uint8)
            desenhar(over, uv)
            cv2.putText(over, f"pose {pid:06d} - camera {cam}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
            lin.append(over)
        linhas.append(np.concatenate(lin, 1))
    grade = np.concatenate(linhas, 0)
    grade = cv2.resize(grade, (grade.shape[1] // 2, grade.shape[0] // 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(a.saida, "verificacao_geometria_overlay.png"), grade)

    # ---- estatisticas: keypoint dentro da mascara do robo / profundidade ----
    stats = {"ids_overlay": ids, "K": K.tolist(), "T_camera_mundo": {c: G.T_camera_mundo(c).tolist() for c in G.CAMERAS_SDF}}
    rng = np.random.default_rng(0)
    amostra = rng.choice(len(jsons), size=min(a.n_stats, len(jsons)), replace=False)
    for cam in ["frontal", "diagonal"]:
        dentro, total, dz, labels = 0, 0, [], {}
        uv_all = []
        for i in amostra:
            pid = int(jsons[i][:6])
            meta = json.load(open(os.path.join(a.dataset, jsons[i])))
            q = np.array([meta["juntas_reais"][j] for j in G.JUNTAS])
            kw = G.fk_keypoints_np(q, cadeia)
            kc = (np.c_[kw, np.ones(len(kw))] @ G.T_camera_mundo(cam).T)[:, :3]
            uv = G.projetar(K, kc)
            uv_all.append(uv)
            seg = cv2.imread(os.path.join(a.dataset, f"{pid:06d}_camera_{cam}_segmentation_labels_map.png"), cv2.IMREAD_UNCHANGED)
            dep = cv2.imread(os.path.join(a.dataset, f"{pid:06d}_camera_{cam}_depth.png"), cv2.IMREAD_UNCHANGED)
            if seg is None or dep is None:
                continue
            lab = seg[..., 0] if seg.ndim == 3 else seg
            for v, c in zip(*np.unique(lab, return_counts=True)):
                labels[int(v)] = labels.get(int(v), 0) + int(c)
            rob = lab == 0  # rotulos: 1 = chao, 2 = fundo, 0 = robo (sem rotulo)
            # dilata 3px: keypoints internos ficam na borda da silhueta em vistas de perfil
            robd = cv2.dilate(rob.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            for k, (u, v) in enumerate(uv):
                ui, vi = int(round(u)), int(round(v))
                if 0 <= ui < G.W and 0 <= vi < G.H:
                    total += 1
                    if robd[vi, ui]:
                        dentro += 1
                        d = float(dep[vi, ui])
                        if d > 0:
                            dz.append(kc[k, 2] * 1000.0 - d)  # (z do keypoint) - (profundidade da superficie)
        uv_all = np.concatenate(uv_all)
        stats[cam] = {
            "keypoints_no_quadro": total,
            "frac_keypoints_sobre_mascara_robo": dentro / max(total, 1),
            "z_keypoint_menos_profundidade_superficie_mm": {
                "mediana": float(np.median(dz)) if dz else None,
                "p10": float(np.percentile(dz, 10)) if dz else None,
                "p90": float(np.percentile(dz, 90)) if dz else None},
            "labels_segmentacao_pixels": labels,
            "uv_min": uv_all.min(0).tolist(), "uv_max": uv_all.max(0).tolist(),
        }
    json.dump(stats, open(os.path.join(a.saida, "verificacao_geometria.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k in ("frontal", "diagonal")}, indent=2))


if __name__ == "__main__":
    main()
