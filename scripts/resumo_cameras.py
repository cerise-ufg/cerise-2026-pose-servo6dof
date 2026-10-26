"""Resumo rapido por camera dos JSONs de metricas (uso: python scripts/resumo_cameras.py resultados)."""
import json
import os
import sys

R = sys.argv[1] if len(sys.argv) > 1 else "resultados"
for nome in ("transfer", "imagenet", "xcam"):
    p = os.path.join(R, f"metricas_teste_{nome}.json")
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    for c in ("todas_cameras", "camera_frontal", "camera_diagonal"):
        r = d.get(f"teste_{c}")
        if r:
            print(f"{nome:9s} {c:16s} MAE={r['MAE_medio_graus']:.2f} e3d={r['erro3d_kp_medio_mm']:.1f} (med {r['erro3d_kp_mediana_mm']:.1f}) "
                  f"AUC={r['AUC_ADD_0_100mm']:.3f} ADD60={100*r['ADD_60mm']:.1f}% ADD175={100*r['ADD_175mm']:.1f}% "
                  f"PCK10={100*r['PCK_kp_10px']:.1f}% juntas={[round(x, 1) for x in r['MAE_juntas_graus']]}")
