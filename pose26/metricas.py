"""Metricas (mesma definicao do codigo de 2025 para ADD/AUC; PCK por keypoint e por imagem)."""
import numpy as np

ADD_LIMIARES_MM = [1, 5, 10, 20, 40, 60, 80, 100]
PCK_LIMIARES_PX = [2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]


def auc_add(dis3d_m, limite=0.1, passo=1e-5):
    """AUC da curva ADD (fracao de imagens com erro 3D medio <= t), t em [0, 100 mm] (lib/utils/metrics.py)."""
    dis3d_m = np.asarray(dis3d_m)
    ts = np.arange(0.0, limite, passo)
    cont = [(dis3d_m <= t).mean() for t in ts]
    return float(np.trapezoid(cont, dx=passo) / limite)


def resumo(pred_kp3d, gt_kp3d, pred_kp2d, gt_kp2d, pred_q=None, gt_q=None, W=1280, H=720):
    """pred/gt kp3d: (N,7,3) m no frame da camera; kp2d: (N,7,2) px na imagem ORIGINAL 1280x720."""
    e3 = np.linalg.norm(pred_kp3d - gt_kp3d, axis=2)  # (N,7) m
    e3_img = e3.mean(1)
    e2 = np.linalg.norm(pred_kp2d - gt_kp2d, axis=2)  # (N,7) px
    vis = (gt_kp2d[..., 0] >= 0) & (gt_kp2d[..., 0] <= W) & (gt_kp2d[..., 1] >= 0) & (gt_kp2d[..., 1] <= H)
    e2_img = (e2 * vis).sum(1) / np.maximum(vis.sum(1), 1)
    r = {
        "n_imagens": int(len(e3)),
        "erro3d_kp_medio_mm": float(e3.mean() * 1000), "erro3d_kp_mediana_mm": float(np.median(e3) * 1000),
        "erro3d_por_keypoint_mm": (e3.mean(0) * 1000).tolist(),
        "AUC_ADD_0_100mm": auc_add(e3_img),
        "erro2d_kp_medio_px": float(e2[vis].mean()), "erro2d_kp_mediana_px": float(np.median(e2[vis])),
        "erro2d_por_keypoint_px": [float(e2[:, k][vis[:, k]].mean()) if vis[:, k].any() else None for k in range(e2.shape[1])],
    }
    for t in ADD_LIMIARES_MM:
        r[f"ADD_{t}mm"] = float((e3_img <= t / 1000).mean())
    for t in (150, 175, 200):  # limiares extras (alcance do servo6dof ~2,4x o do Panda)
        r[f"ADD_{t}mm"] = float((e3_img <= t / 1000).mean())
    for t in (5, 10, 20):
        r[f"PCK_kp_{t}px"] = float((e2[vis] <= t).mean())       # por keypoint (padrao em pose 2D)
        r[f"PCK_img_{t}px"] = float((e2_img <= t).mean())       # por imagem (como o codigo de 2025)
    if pred_q is not None:
        ej = np.abs(pred_q - gt_q) * 180 / np.pi
        r["MAE_juntas_graus"] = ej.mean(0).tolist()
        r["MAE_medio_graus"] = float(ej.mean())
        r["mediana_erro_juntas_graus"] = np.median(ej, 0).tolist()
    return r, e3_img, e2_img
