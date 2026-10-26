"""Grava ~/cerise2026/pose/READY.json (consumido pela frente do gemeo digital)."""
import argparse
import json
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--saida", required=True)
ap.add_argument("--metricas", required=True)
ap.add_argument("--teste_predict", default=None, help="JSON impresso pelo CLI do predict.py")
a = ap.parse_args()
m = json.load(open(a.metricas))
t = m["teste_todas_cameras"]
pred = None
if a.teste_predict and os.path.exists(a.teste_predict):
    txt = open(a.teste_predict).read()
    try:
        pred = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
    except Exception:  # noqa: BLE001
        pred = None
ready = {
    "status": "pronto",
    "gerado_em": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    "ckpt": "/home/energiainteligente/cerise2026/pose/inferencia/modelo/pose_servo6dof.pt",
    "ckpt_no_container": "/workspace/pose/inferencia/modelo/pose_servo6dof.pt",
    "imagem_docker": "cerise26_pose:latest",
    "montagens_docker": ["-v ~/cerise2026/pose:/workspace/pose", "-v ~/cerise2026/hf_cache:/root/.cache/huggingface",
                         "-e POSE_REPO=/workspace/pose/src/repo", "--gpus all --ipc=host"],
    "python_path": ["/workspace/pose/inferencia", "/workspace/pose/src/repo", "/workspace/pose/src/repo/hpe2025"],
    "import": "from predict import PosePredictor",
    "uso": "p = PosePredictor(ckpt, device='cuda'); out = p.predict(rgb_uint8_HxWx3, camera='frontal'|'diagonal')",
    "saida": {"joints": "dict joint_1..joint_5 (rad)", "keypoints_2d": "7 x [u,v] px", "keypoints_3d": "7 x [x,y,z] m (frame otico)",
              "latency_ms": "float"},
    "entrada": "RGB uint8 1280x720 das cameras fixas do estudio_hd (frontal/diagonal); outras resolucoes sao reescaladas",
    "doc": "/home/energiainteligente/cerise2026/pose/INFERENCIA.md",
    "modelo": "HPE/HoRoPose + Depth Anything V2 (modelo 2025 MyNet), transfer learning dos pesos Panda -> servo6dof",
    "epoca": m.get("epoca"),
    "metricas_teste": {
        "n_imagens": t["n_imagens"], "MAE_juntas_graus": t["MAE_juntas_graus"], "MAE_medio_graus": t["MAE_medio_graus"],
        "erro3d_kp_medio_mm": t["erro3d_kp_medio_mm"], "erro3d_kp_mediana_mm": t["erro3d_kp_mediana_mm"],
        "AUC_ADD_0_100mm": t["AUC_ADD_0_100mm"], "ADD_60mm": t["ADD_60mm"], "PCK_kp_10px": t["PCK_kp_10px"]},
    "latencia_gb10_batch1": m.get("latencia"),
    "teste_cli": pred,
}
json.dump(ready, open(a.saida, "w"), indent=1)
print(json.dumps(ready, indent=1))
