"""Gera as figuras do relatorio (A8/A9) a partir de resultados/ (rodar no Windows ou no Spark).

uso: python scripts/figuras.py --res resultados --out evidencias
"""
import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from servo6dof import geometria as G  # noqa: E402

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
COR = {"transfer": "#1f6fb4", "imagenet": "#d9822b", "frontal": "#1f6fb4", "diagonal": "#2a9d55"}
ROT = {"transfer": "Transfer. (pesos 2025, Panda)", "imagenet": "Baseline (ImageNet)"}
JUNTAS_PT = ["junta 1 (base)", "junta 2 (ombro)", "junta 3 (cotovelo)", "junta 4 (punho)", "junta 5 (rot. garra)"]
KP_PT = ["base", "j1", "j2", "j3", "j4", "j5", "garra"]


def ler_csv(p):
    if not os.path.exists(p):
        return None
    with open(p) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows]) for k in rows[0]}


def ler_json(p):
    return json.load(open(p)) if os.path.exists(p) else None


def fig_curvas(res, out):
    C = {m: ler_csv(os.path.join(res, f"curvas_full_{m}.csv")) for m in ("transfer", "imagenet")}
    D = {m: ler_csv(os.path.join(res, f"curvas_depthnet_{m}.csv")) for m in ("transfer", "imagenet")}
    if not any(v is not None for v in C.values()):
        return
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.0))
    for m, c in C.items():
        if c is None:
            continue
        e = c["epoca"]
        ax[0].plot(e, c["treino_total"], "-o", ms=3, color=COR[m], label=ROT[m])
        ax[1].plot(e, c["val_erro3d_mm"], "-o", ms=3, color=COR[m], label=ROT[m])
        ax[2].plot(e, c["val_MAE_graus"], "-o", ms=3, color=COR[m], label=ROT[m])
        ax[3].plot(e, 100 * c["val_PCK_kp_10px"], "-o", ms=3, color=COR[m], label=ROT[m])
    ax[0].set_title("Perda total (treino)"); ax[0].set_yscale("log")
    ax[1].set_title("Erro 3D medio dos keypoints (val, mm)")
    ax[2].set_title("MAE das juntas (val, graus)")
    ax[3].set_title("PCK@10 px por keypoint (val, %)")
    for a in ax:
        a.set_xlabel("epoca")
    ax[0].legend(fontsize=7)
    fig.suptitle("Etapa 2 (modelo completo) - curvas por epoca no novo dataset do servo6dof", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "curvas_treino.png")); plt.close(fig)
    if any(v is not None for v in D.values()):
        fig, ax = plt.subplots(1, 2, figsize=(7.5, 2.8))
        for m, c in D.items():
            if c is None:
                continue
            ax[0].plot(c["epoca"], c["treino_depth_mm"], "-o", ms=3, color=COR[m], label=ROT[m])
            ax[1].plot(c["epoca"], c["val_erro_prof_raiz_mm"], "-o", ms=3, color=COR[m], label=ROT[m])
        ax[0].set_title("Etapa 1 (DepthNet): perda L1 treino (mm)"); ax[1].set_title("Erro de profundidade da raiz (val, mm)")
        for a in ax:
            a.set_xlabel("epoca")
        ax[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out, "curvas_depthnet.png")); plt.close(fig)


def fig_erros(res, out):
    M = {m: ler_json(os.path.join(res, f"metricas_teste_{m}.json")) for m in ("transfer", "imagenet")}
    M = {k: v for k, v in M.items() if v}
    if not M:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    w = 0.38
    for i, (m, r) in enumerate(M.items()):
        t = r["teste_todas_cameras"]
        x = np.arange(5)
        b = ax[0].bar(x + (i - 0.5) * w, t["MAE_juntas_graus"], w, color=COR[m], label=ROT[m])
        ax[0].bar_label(b, fmt="%.1f", fontsize=7)
        x = np.arange(7)
        b = ax[1].bar(x + (i - 0.5) * w, t["erro3d_por_keypoint_mm"], w, color=COR[m], label=ROT[m])
        ax[1].bar_label(b, fmt="%.0f", fontsize=6)
    ax[0].set_xticks(np.arange(5)); ax[0].set_xticklabels(JUNTAS_PT, fontsize=7, rotation=10)
    ax[0].set_ylabel("erro absoluto medio (graus)"); ax[0].set_title("Erro por junta (teste)")
    ax[1].set_xticks(np.arange(7)); ax[1].set_xticklabels(KP_PT)
    ax[1].set_ylabel("erro 3D medio (mm)"); ax[1].set_title("Erro 3D por keypoint (teste)")
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "erro_por_junta.png")); plt.close(fig)


def fig_add_pck(res, out):
    P = {m: os.path.join(res, f"predicoes_teste_{m}.npz") for m in ("transfer", "imagenet")}
    P = {k: np.load(v) for k, v in P.items() if os.path.exists(v)}
    if not P:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    ts = np.linspace(0, 400, 801)
    for m, z in P.items():
        e3 = z["e3_img"] * 1000
        ax[0].plot(ts, [100 * (e3 <= t).mean() for t in ts], color=COR[m], label=ROT[m])
        e2 = np.linalg.norm(z["p2"] - z["g2"], axis=2)
        vis = (z["g2"][..., 0] >= 0) & (z["g2"][..., 0] <= 1280) & (z["g2"][..., 1] >= 0) & (z["g2"][..., 1] <= 720)
        e2 = e2[vis]
        tp = np.linspace(0, 60, 601)
        ax[1].plot(tp, [100 * (e2 <= t).mean() for t in tp], color=COR[m], label=ROT[m])
    ax[0].axvline(60, color="k", ls="--", lw=0.8); ax[0].text(62, 5, "60 mm (limiar de 2025)", fontsize=7)
    ax[0].axvline(100, color="gray", ls=":", lw=0.8)
    ax[0].set_xlabel("limiar t (mm)"); ax[0].set_ylabel("% de imagens com erro 3D medio <= t")
    ax[0].set_title("Curva ADD (acuracia por limiar) - teste")
    for t in (5, 10, 20):
        ax[1].axvline(t, color="gray", ls=":", lw=0.8)
    ax[1].set_xlabel("limiar (px, imagem 1280x720)"); ax[1].set_ylabel("% de keypoints com erro 2D <= limiar")
    ax[1].set_title("Curva PCK 2D - teste")
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "curvas_add_pck.png")); plt.close(fig)


NOMES_PERT = {"ruido_gaussiano": ("Ruido gaussiano", "desvio (0-1)"), "desfoque_movimento": ("Desfoque de movimento", "kernel (px na entrada 256)"),
              "brilho": ("Brilho", "fator"), "contraste": ("Contraste", "fator"), "oclusao": ("Oclusao (cutout)", "fracao da caixa do robo"),
              "jpeg": ("Compressao JPEG", "qualidade"), "resolucao": ("Resolucao reduzida", "escala")}


def fig_robustez(res, out):
    M = {m: ler_json(os.path.join(res, f"metricas_teste_{m}.json")) for m in ("transfer", "imagenet")}
    M = {k: v for k, v in M.items() if v and "robustez" in v}
    if not M:
        return
    tipos = list(next(iter(M.values()))["robustez"]["condicoes"].keys())
    fig, axs = plt.subplots(2, 4, figsize=(13, 5.6))
    axs = axs.ravel()
    for i, tp in enumerate(tipos):
        a = axs[i]
        a2 = a.twinx()
        for m, r in M.items():
            rb = r["robustez"]
            pts = [(0 if tp not in ("brilho", "contraste", "jpeg", "resolucao") else (1.0 if tp in ("brilho", "contraste", "resolucao") else 100),
                    rb["limpa"])] + [(c["severidade"], c) for c in rb["condicoes"][tp]]
            pts.sort(key=lambda p: p[0])
            xs = [p[0] for p in pts]
            a.plot(xs, [p[1]["erro3d_kp_medio_mm"] for p in pts], "-o", ms=3, color=COR[m], label=f"{ROT[m]} - erro 3D")
            a2.plot(xs, [p[1]["MAE_medio_graus"] for p in pts], "--s", ms=3, color=COR[m], alpha=0.6, label=f"{ROT[m]} - MAE")
        t, xl = NOMES_PERT.get(tp, (tp, ""))
        a.set_title(t); a.set_xlabel(xl); a.set_ylabel("erro 3D medio (mm)"); a2.set_ylabel("MAE juntas (graus)", fontsize=7)
        a2.grid(False)
        if tp == "jpeg":
            a.invert_xaxis()
        if tp == "resolucao":
            a.invert_xaxis()
    h1, l1 = axs[0].get_legend_handles_labels()
    axs[-1].axis("off")
    axs[-1].legend(h1 + [plt.Line2D([], [], ls="--", marker="s", color="gray")], l1 + ["tracejado: MAE (eixo direito)"], loc="center", fontsize=8)
    fig.suptitle("Robustez sob perturbacoes da imagem de entrada (subconjunto fixo do teste; DA-V2 recalculado)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, "curvas_robustez.png")); plt.close(fig)


def fig_perturbacoes(res, out):
    p = os.path.join(res, "amostras_perturbacoes.npz")
    if not os.path.exists(p):
        return
    z = np.load(p)
    ks = ["limpa"] + [k for k in z.files if k != "limpa"]
    fig, axs = plt.subplots(1, len(ks), figsize=(1.75 * len(ks), 2.1))
    for a, k in zip(axs, ks):
        a.imshow(z[k]); a.set_title(NOMES_PERT.get(k, ("Original", ""))[0] if k != "limpa" else "Original", fontsize=7)
        a.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(out, "amostras_perturbacoes.png")); plt.close(fig)


def fig_predicoes(res, out, modelo="transfer"):
    p = os.path.join(res, f"predicoes_teste_{modelo}.npz")
    pasta = os.path.join(res, "imagens_teste")
    if not os.path.exists(p) or not os.path.isdir(pasta):
        return
    import cv2
    z = np.load(p)
    sel = json.load(open(os.path.join(pasta, "selecao.json")))
    fig, axs = plt.subplots(2, len(sel["ids"]), figsize=(4.2 * len(sel["ids"]), 5.4))
    for j, pid in enumerate(sel["ids"]):
        for ci, cam in enumerate(["frontal", "diagonal"]):
            a = axs[ci, j]
            im = cv2.imread(os.path.join(pasta, f"{pid:06d}_camera_{cam}_image.png"))[..., ::-1]
            k = np.nonzero((z["id"] == pid) & (z["cam"] == ci))[0][0]
            g, pr = z["g2"][k], z["p2"][k]
            a.imshow(im)
            for (u, v) in G.ARESTAS:
                a.plot([g[u, 0], g[v, 0]], [g[u, 1], g[v, 1]], "-", color="#00e676", lw=2.2)
                a.plot([pr[u, 0], pr[v, 0]], [pr[u, 1], pr[v, 1]], "-", color="#ff1744", lw=1.6)
            a.scatter(g[:, 0], g[:, 1], s=12, c="#00e676", zorder=3)
            a.scatter(pr[:, 0], pr[:, 1], s=10, c="#ff1744", zorder=4)
            lo = np.minimum(g.min(0), pr.min(0)) - 80; hi = np.maximum(g.max(0), pr.max(0)) + 80
            a.set_xlim(max(lo[0], 0), min(hi[0], 1280)); a.set_ylim(min(hi[1], 720), max(lo[1], 0))
            e3 = z["e3_img"][k] * 1000
            ej = np.abs(z["pq"][k] - z["gq"][k]).mean() * 180 / np.pi
            a.set_title(f"pose {pid:06d} - {cam}\nerro 3D {e3:.0f} mm, MAE {ej:.1f} graus", fontsize=8)
            a.axis("off")
    fig.legend([plt.Line2D([], [], color="#00e676", lw=2), plt.Line2D([], [], color="#ff1744", lw=2)],
               ["verdade (FK do URDF com as juntas reais)", f"predicao ({ROT[modelo]})"], loc="lower center", ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(out, f"predicoes_vs_gt_{modelo}.png")); plt.close(fig)


def fig_tabela(res, out):
    M = {m: ler_json(os.path.join(res, f"metricas_teste_{m}.json")) for m in ("transfer", "imagenet")}
    linhas = [["RoboPose (literatura) - Panda/DREAM real", "74,3", "-", "-", "relatorio dez/2025"],
              ["HPRE (literatura) - Panda/DREAM real", "75,2", "-", "-", "relatorio dez/2025"],
              ["Modelo 2025 (Eduardo) - Panda/DREAM real", "75,4", "0,547", "-", "relatorio dez/2025 (ep. 355)"]]
    for m in ("transfer", "imagenet"):
        if M.get(m):
            t = M[m]["teste_todas_cameras"]
            linhas.append([f"{ROT[m]} - servo6dof/sintetico 2026", f"{100*t['ADD_60mm']:.1f}".replace(".", ","),
                           f"{t['AUC_ADD_0_100mm']:.3f}".replace(".", ","), f"{t['MAE_medio_graus']:.2f}".replace(".", ","), "medido (teste)"])
    fig, ax = plt.subplots(figsize=(10.5, 0.5 + 0.38 * len(linhas)))
    ax.axis("off")
    tb = ax.table(cellText=linhas, colLabels=["Modelo / robo / dados", "Acuracia @60 mm (%)", "AUC ADD (0-100 mm)", "MAE juntas (graus)", "Fonte"],
                  loc="center", cellLoc="center", colWidths=[0.38, 0.14, 0.14, 0.13, 0.21])
    tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.35)
    for (r, c), cell in tb.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e8eef6"); cell.set_text_props(weight="bold")
        elif r >= 4:
            cell.set_facecolor("#f6f9f2")
    ax.set_title("Acuracia @60 mm: 2025 (Panda) x 2026 (servo6dof) - robos, escalas e datasets diferentes; nao e comparacao direta", fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(out, "tabela_comparativa.png")); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in (fig_curvas, fig_erros, fig_add_pck, fig_robustez, fig_perturbacoes, fig_tabela):
        try:
            f(a.res, a.out)
            print("ok", f.__name__)
        except Exception as e:  # noqa: BLE001
            print("falhou", f.__name__, e)
    for m in ("transfer", "imagenet"):
        try:
            fig_predicoes(a.res, a.out, m)
            print("ok fig_predicoes", m)
        except Exception as e:  # noqa: BLE001
            print("falhou fig_predicoes", m, e)


if __name__ == "__main__":
    main()
