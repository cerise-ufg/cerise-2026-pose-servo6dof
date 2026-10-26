"""Gera as tabelas (markdown) do relatorio a partir de resultados/*.json."""
import json
import os
import sys

R = sys.argv[1] if len(sys.argv) > 1 else "resultados"


def J(n):
    p = os.path.join(R, n)
    return json.load(open(p)) if os.path.exists(p) else None


def f(v, d=1, pct=False):
    if v is None:
        return "-"
    v = 100 * v if pct else v
    return f"{v:.{d}f}".replace(".", ",")


T = J("metricas_teste_transfer.json")
B = J("metricas_teste_imagenet.json")
X = J("metricas_teste_xcam.json")
out = []
out.append("| Métrica (teste, 500 poses × 2 câmeras) | Transferência (pesos 2025) | Baseline (ImageNet) |")
out.append("|---|---|---|")
lin = [("MAE médio das juntas (°)", "MAE_medio_graus", 2, False), ("Erro 3D médio dos keypoints (mm)", "erro3d_kp_medio_mm", 1, False),
       ("Erro 3D mediano dos keypoints (mm)", "erro3d_kp_mediana_mm", 1, False), ("AUC ADD 0–100 mm", "AUC_ADD_0_100mm", 3, False),
       ("Acurácia ADD @60 mm (%)", "ADD_60mm", 1, True), ("Acurácia ADD @100 mm (%)", "ADD_100mm", 1, True),
       ("Acurácia ADD @150 mm (%) ≈ 60 mm × 2,4 (escala do robô)", "ADD_150mm", 1, True), ("Acurácia ADD @175 mm (%)", "ADD_175mm", 1, True),
       ("Erro 2D médio dos keypoints (px)", "erro2d_kp_medio_px", 1, False), ("Erro 2D mediano (px)", "erro2d_kp_mediana_px", 1, False),
       ("PCK@5 px por keypoint (%)", "PCK_kp_5px", 1, True), ("PCK@10 px por keypoint (%)", "PCK_kp_10px", 1, True),
       ("PCK@20 px por keypoint (%)", "PCK_kp_20px", 1, True), ("PCK@10 px por imagem (%) (critério 2025)", "PCK_img_10px", 1, True)]
for nome, k, d, pct in lin:
    t = T["teste_todas_cameras"].get(k) if T else None
    b = B["teste_todas_cameras"].get(k) if B else None
    out.append(f"| {nome} | {f(t, d, pct)} | {f(b, d, pct)} |")
out.append(f"| Épocas (melhor checkpoint na validação) | {T['epoca'] if T else '-'} | {B['epoca'] if B else '-'} |")
print("\n".join(out)); print()

out = ["| Junta | Transferência MAE (°) | Transferência mediana (°) | Baseline MAE (°) |", "|---|---|---|---|"]
nomes = ["joint_1 (base)", "joint_2 (ombro)", "joint_3 (cotovelo)", "joint_4 (punho)", "joint_5 (rotação da garra)"]
for i, n in enumerate(nomes):
    t = T["teste_todas_cameras"] if T else None
    b = B["teste_todas_cameras"] if B else None
    out.append(f"| {n} | {f(t['MAE_juntas_graus'][i], 2) if t else '-'} | {f(t['mediana_erro_juntas_graus'][i], 2) if t else '-'} | {f(b['MAE_juntas_graus'][i], 2) if b else '-'} |")
print("\n".join(out)); print()

out = ["| Câmera de teste | Modelo | MAE (°) | Erro 3D médio (mm) | AUC ADD | ADD@60 mm (%) | PCK@10 px (%) |", "|---|---|---|---|---|---|---|"]
for nome, M in (("Transferência (2 câmeras)", T), ("Baseline (2 câmeras)", B), ("Treino só frontal (xcam)", X)):
    if not M:
        continue
    for c in ("frontal", "diagonal"):
        r = M.get(f"teste_camera_{c}")
        if r:
            out.append(f"| {c} | {nome} | {f(r['MAE_medio_graus'], 2)} | {f(r['erro3d_kp_medio_mm'])} | {f(r['AUC_ADD_0_100mm'], 3)} | {f(r['ADD_60mm'], 1, True)} | {f(r['PCK_kp_10px'], 1, True)} |")
print("\n".join(out)); print()

for nome, M in (("Transferência", T), ("Baseline", B)):
    if not M or "robustez" not in M:
        continue
    rb = M["robustez"]
    out = [f"**{nome}** — subconjunto fixo de {rb['n_imagens']} imagens do teste",
           "", "| Perturbação | Severidade | Erro 3D médio (mm) | MAE (°) | PCK@10 px (%) | AUC ADD |", "|---|---|---|---|---|---|"]
    L = rb["limpa"]
    out.append(f"| (sem perturbação) | - | {f(L['erro3d_kp_medio_mm'])} | {f(L['MAE_medio_graus'], 2)} | {f(L['PCK_kp_10px'], 1, True)} | {f(L['AUC_ADD_0_100mm'], 3)} |")
    for tp, lst in rb["condicoes"].items():
        for c in lst:
            out.append(f"| {tp} | {c['severidade']} | {f(c['erro3d_kp_medio_mm'])} | {f(c['MAE_medio_graus'], 2)} | {f(c['PCK_kp_10px'], 1, True)} | {f(c['AUC_ADD_0_100mm'], 3)} |")
    print("\n".join(out)); print()

if T and "latencia" in T:
    L = T["latencia"]
    print("| Etapa (batch 1, GB10) | Média (ms) | Mediana (ms) | p95 (ms) |\n|---|---|---|---|")
    for k, n in (("da", "Depth Anything V2 (ViT-L, 518 px)"), ("hpe", "HPE (DepthNet + integral + regressão + FK)"), ("total", "Total")):
        print(f"| {n} | {f(L[k]['media_ms'])} | {f(L[k]['mediana_ms'])} | {f(L[k]['p95_ms'])} |")
    print(f"\nFPS total: {f(L['fps_total'], 2)}; FPS só HPE: {f(L['fps_so_hpe'], 2)}; memória de pico: {f(L['mem_pico_GB'], 2)} GB")
