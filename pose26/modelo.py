"""Construcao do modelo de 2025 (HPE/HoRoPose + ramo Depth Anything V2 da variante MyNet) para o servo6dof
e carregamento dos pesos do modelo de 2025 (Panda) onde as formas casam (transfer learning).
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
HPE = os.path.join(RAIZ, "hpe2025")
for p in (RAIZ, HPE):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.full_net import get_rootNetwithRegInt_model  # noqa: E402
from models.depth_net import get_rootnet  # noqa: E402
from robot_servo6dof import Servo6dofRobot  # noqa: E402

REF_KP = 3  # keypoint de referencia (raiz) = junta_3, como reference_keypoint_id: 3 do Panda em 2025


def args_2025(**kw):
    """Hiperparametros do modelo final de 2025 (MyNet configs/panda/full.yaml + defaults de lib/core/config.py)."""
    a = SimpleNamespace(
        backbone_name="resnet50", rootnet_backbone_name="hrnet32", other_image_size=256.0, image_size=256.0,
        rootnet_image_size=256.0, use_rpmg=False, n_iter=4, p_dropout=0.5, reg_joint_map=False, rotation_dim=6,
        direct_reg_rot=False, rot_iterative_matmul=False, fix_root=True, add_fc=False, multi_kp=False,
        kps_need_depth=None, reference_keypoint_id=REF_KP,
        # servo6dof: ~2,0 m de alcance do ombro a garra, 2,6 m de altura (Panda ~0,85 m): faixa de profundidade relativa do heatmap 3D ampliada
        bbox_3d_shape=[3000, 3000, 3000],
        pretrained_rootnet=None, imagenet_init=True,
        # perdas: estagio completo sintetico de 2025 (configs/panda/full.yaml do HPE: ramo integral com peso 10,
        # alinhamento 0). A variante MyNet usava integral 0 + alinhamento 1 partindo de um ramo integral ja
        # treinado; aqui o ramo integral precisa ser readaptado ao novo robo/faixa de profundidade.
        pose_loss_weight=1.0, rot_loss_weight=1.0, trans_loss_weight=1.0, depth_loss_weight=10.0, uv_loss_weight=1.0,
        kp2d_loss_weight=10.0, kp3d_loss_weight=10.0, kp2d_int_loss_weight=10.0, kp3d_int_loss_weight=10.0,
        align_3d_loss_weight=0.0, clip_gradient=5.0, lr=1e-4, batch_size=32,
    )
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def construir_full(args, robot=None, device="cuda"):
    robot = robot or Servo6dofRobot(device=device)
    init = {"robot_type": "servo6dof", "robot": robot, "pose_params": np.zeros(5, np.float32),
            "cam_params": np.eye(4, dtype=float), "init_pose_from_mean": True}
    m = get_rootNetwithRegInt_model(init, args)
    return m.to(device), robot


def construir_depthnet(device="cuda"):
    return get_rootnet("hrnet32", pred_xy=False, use_offset=False, add_fc=False).to(device)


def ativar_amp(model, nomes=("reg_backbone", "deconv_layers", "final_layer", "rootnet_backbone", "depth_backbone", "backbone")):
    """Precisao mista bf16 so nos extratores convolucionais (backbones); camada integral, regressao
    iterativa, FK e perdas continuam em fp32. Nao muda nomes/pesos (state_dict identico)."""
    for n in nomes:
        mod = getattr(model, n, None)
        if mod is None or getattr(mod, "_amp", False):
            continue
        orig = mod.forward

        def f(*args, _orig=orig, **kw):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = _orig(*args, **kw)
            return tuple(o.float() for o in out) if isinstance(out, tuple) else out.float()
        mod.forward = f
        mod._amp = True
    return model


def carregar_parcial(model, state_dict, prefixo_map=None, log=print):
    """Carrega os tensores cujo nome e forma casam; retorna estatisticas (transfer learning)."""
    prefixo_map = prefixo_map or {}
    alvo = model.state_dict()
    usados, forma_dif, ausentes = {}, [], []
    for k, v in state_dict.items():
        k2 = k
        for a, b in prefixo_map.items():
            if k2.startswith(a):
                k2 = b + k2[len(a):]
        if k2 in alvo:
            if alvo[k2].shape == v.shape:
                usados[k2] = v
            else:
                forma_dif.append((k2, tuple(v.shape), tuple(alvo[k2].shape)))
        else:
            ausentes.append(k2)
    model.load_state_dict(usados, strict=False)
    n_alvo = sum(t.numel() for t in alvo.values())
    n_usado = sum(t.numel() for t in usados.values())
    st = {"tensores_carregados": len(usados), "tensores_modelo": len(alvo), "params_carregados": int(n_usado),
          "params_modelo": int(n_alvo), "frac_params": n_usado / n_alvo,
          "forma_diferente": forma_dif, "nao_existem_no_modelo": ausentes[:20]}
    log(f"[transfer] {len(usados)}/{len(alvo)} tensores ({100*n_usado/n_alvo:.1f}% dos parametros) carregados; "
        f"forma diferente: {[f[0] for f in forma_dif]}")
    return st


def ler_ckpt(path):
    import types
    if "torchnet" not in sys.modules:  # o checkpoint de 2025 guarda torchnet.AverageValueMeter em 'all_metrics'
        tn, tm = types.ModuleType("torchnet"), types.ModuleType("torchnet.meter")

        class AverageValueMeter:
            pass
        tm.AverageValueMeter = AverageValueMeter
        tm.__path__ = []  # pacote
        tav = types.ModuleType("torchnet.meter.averagevaluemeter")
        tav.AverageValueMeter = AverageValueMeter
        tn.meter = tm
        sys.modules["torchnet"], sys.modules["torchnet.meter"] = tn, tm
        sys.modules["torchnet.meter.averagevaluemeter"] = tav
    return torch.load(path, map_location="cpu", weights_only=False)
