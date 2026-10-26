"""Inspeciona o checkpoint final do modelo de 2025 (MyNet / realsense_full / best_model_auc.pk)."""
import json
import sys
from collections import OrderedDict

import types

import torch

# o checkpoint de 2025 guarda objetos torchnet.meter.AverageValueMeter em 'all_metrics':
# stub minimo so para desserializar (nao e preciso instalar o torchnet).
_tn = types.ModuleType("torchnet"); _tm = types.ModuleType("torchnet.meter")


class AverageValueMeter:  # noqa: D401
    pass


_tm.AverageValueMeter = AverageValueMeter
_tn.meter = _tm
sys.modules.setdefault("torchnet", _tn); sys.modules.setdefault("torchnet.meter", _tm)
# tambem existe torchnet.meter.averagevaluemeter em versoes antigas
_tav = types.ModuleType("torchnet.meter.averagevaluemeter"); _tav.AverageValueMeter = AverageValueMeter
sys.modules.setdefault("torchnet.meter.averagevaluemeter", _tav)

p = sys.argv[1]
ck = torch.load(p, map_location="cpu", weights_only=False)
print("chaves do checkpoint:", list(ck.keys()))
info = {k: (v if isinstance(v, (int, float, str)) else str(type(v))) for k, v in ck.items() if k not in ("model_state_dict", "optimizer_state_dict", "all_metrics")}
print(json.dumps(info, indent=1, default=str))
sd = ck["model_state_dict"]
grupos = OrderedDict()
for k, v in sd.items():
    g = k.split(".")[0]
    n = grupos.setdefault(g, [0, 0])
    n[0] += 1
    n[1] += v.numel()
for g, (n, p_) in grupos.items():
    print(f"{g:32s} tensores={n:5d} params={p_/1e6:8.3f}M")
print("total params (M):", sum(v.numel() for v in sd.values()) / 1e6)
for k in ["final_layer.weight", "fc_pose_1.weight", "decpose.weight", "init_pose", "init_rot", "decrot.weight",
          "fc_rot_1.weight", "depth_layer.weight", "depth_backbone.encoder.0.weight", "fusion_fc.weight"]:
    if k in sd:
        print(k, tuple(sd[k].shape))
print("init_pose", sd.get("init_pose"))
am = ck.get("all_metrics")
if isinstance(am, dict):
    sel = {k: (float(v) if not hasattr(v, "mean") else float(v.mean) if not callable(v.mean) else float(v.mean())) for k, v in am.items()
           if any(s in k for s in ["ADD_60", "AUC_ADD", "mean_joint_error"])}
    print(json.dumps(sel, indent=1))
