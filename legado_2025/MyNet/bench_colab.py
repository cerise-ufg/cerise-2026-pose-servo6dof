# -*- coding: utf-8 -*-
u"""
bench_colab.py  --  mede FPS (latencia) e FLOPs/parametros da rede proposta,
para preencher a Secao 7.8 da dissertacao.

Usa ENTRADAS FICTICIAS (nao depende do dataset) e espelha EXATAMENTE a chamada
de forward que o scripts/test.py faz. Reaproveita o make_cfg e a construcao de
modelo do proprio projeto, de modo que o numero saia da sua rede real.

COMO USAR NO COLAB: cole as celulas do notebook (ver instrucoes no chat) OU,
apos montar o Drive e entrar na pasta MyNet, rode:
    !python bench_colab.py
Ajuste EXP e MODEL abaixo se necessario.
"""
import os
import sys
import time
import argparse
import importlib.util

import numpy as np
import torch

# ------------------------------------------------------------------ ajuste aqui
ROOT = os.path.dirname(os.path.abspath(__file__))          # .../MyNet
EXP = os.path.join(ROOT, 'experiments', 'panda_realsense_full')   # tem config.yaml
# checkpoint real do modelo proposto (o mesmo que deu 75,4%):
CKPT = os.path.join(ROOT, 'experiments', 'panda', 'realsense_full', 'best_model_auc.pk')
BATCHES = [1, 8]        # B=1 -> comparavel aos 22,6 FPS do HRPE; B=8 -> vazao
N_ITER = 100
N_WARMUP = 15
# -----------------------------------------------------------------------------

sys.path.append(ROOT)
os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')   # pyrender headless


def _install_pytorch3d_shim():
    u"""Satisfaz os imports de pytorch3d SEM instala-lo.

    O pytorch3d so e usado pelo renderizador de mascara (mesh_renderer), que
    o forward de inferencia nunca chama. Fabricamos modulos-fantasma para
    'pytorch3d.*' de modo que os imports de topo passem; como as funcoes de
    renderizacao nao sao chamadas na medicao de FPS/FLOPs, os fantasmas nunca
    sao executados.
    """
    import importlib.util
    import importlib.abc
    import importlib.machinery
    import types
    if importlib.util.find_spec('pytorch3d') is not None:
        print('  pytorch3d instalado; usando o real.')
        return False

    class _Dummy(object):
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return None

        def __getattr__(self, k):
            return _Dummy()

    class _Mod(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith('__') and name.endswith('__'):
                raise AttributeError(name)
            return _Dummy

    class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'pytorch3d' or fullname.startswith('pytorch3d.'):
                return importlib.machinery.ModuleSpec(fullname, self)
            return None

        def create_module(self, spec):
            return _Mod(spec.name)

        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, _Finder())
    print('  pytorch3d ausente: usando stub (renderizador nao e necessario '
          'para FPS/FLOPs).')
    return True


_install_pytorch3d_shim()


def load_make_cfg():
    u"""Importa make_cfg do scripts/test.py sem executar o __main__."""
    spec = importlib.util.spec_from_file_location(
        'proj_test', os.path.join(ROOT, 'scripts', 'test.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.make_cfg


def build_model():
    from lib.dataset.const import INITIAL_JOINT_ANGLE
    from lib.models.full_net import get_rootNetwithRegInt_model

    make_cfg = load_make_cfg()
    args = argparse.Namespace(exp_path=EXP, known_joint=False,
                              model_name=os.path.splitext(os.path.basename(CKPT))[0],
                              vis_skeleton=False, dataset='panda-3cam_realsense')
    cfg = make_cfg(args)

    init_param_dict = {
        'robot_type': cfg.urdf_robot_name,
        'pose_params': INITIAL_JOINT_ANGLE,
        'cam_params': np.eye(4, dtype=float),
        'init_pose_from_mean': True,
    }
    model = get_rootNetwithRegInt_model(init_param_dict, cfg)
    ckpt = torch.load(CKPT, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print('  [aviso] load_state_dict nao estrito: %d faltando, %d inesperadas'
              % (len(missing), len(unexpected)))
    import inspect
    try:
        print('  forward(...) =', inspect.signature(model.forward))
    except (TypeError, ValueError):
        pass
    return model.cuda().eval(), cfg


def _call(model, reg, root, k_values, K):
    u"""Chama o forward espelhando o test.py; em caso de divergencia de
    assinatura, tenta a variante com depth_feats e reporta o problema."""
    try:
        return model(reg, root, k_values, K=K, test_fps=False)
    except TypeError as e:
        print('  [aviso] chamada padrao falhou (%s); tentando com depth_feats=None' % e)
        return model(reg, root, None, k_values, K=K, test_fps=False)


def dummy_inputs(cfg, B):
    S = int(cfg.other_image_size)
    Sr = int(cfg.rootnet_image_size)
    reg = torch.rand(B, 3, S, S, device='cuda')
    root = torch.rand(B, 3, Sr, Sr, device='cuda')
    k_values = torch.full((B,), 2000.0, device='cuda')
    K = torch.tensor([[500.0, 0.0, S / 2.0],
                      [0.0, 500.0, S / 2.0],
                      [0.0, 0.0, 1.0]], device='cuda').unsqueeze(0).repeat(B, 1, 1)
    return reg, root, k_values, K


def measure_fps(model, cfg):
    print('\n=== FPS / latencia (entradas ficticias) ===')
    for B in BATCHES:
        reg, root, k_values, K = dummy_inputs(cfg, B)
        with torch.no_grad():
            for _ in range(N_WARMUP):
                _call(model, reg, root, k_values, K)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(N_ITER):
                _call(model, reg, root, k_values, K)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / N_ITER
        print('  B=%-2d  latencia/lote=%.1f ms  |  FPS=%.1f imgs/s  (%.1f lotes/s)'
              % (B, dt * 1000, B / dt, 1.0 / dt))


def measure_flops(model, cfg):
    print('\n=== FLOPs / parametros (thop) ===')
    try:
        from thop import profile
    except Exception as e:
        print('  thop indisponivel:', e)
        return
    reg, root, k_values, K = dummy_inputs(cfg, 1)

    class Wrap(torch.nn.Module):
        def __init__(self, m, K):
            super().__init__()
            self.m = m
            self.K = K

        def forward(self, a, b, c):
            return _call(self.m, a, b, c, self.K)

    macs, params = profile(Wrap(model, K).cuda(), inputs=(reg, root, k_values),
                           verbose=False)
    print('  MACs   = %.2f G' % (macs / 1e9))
    print('  FLOPs ~= %.2f G   (2 x MACs)' % (2 * macs / 1e9))
    print('  params = %.2f M' % (params / 1e6))
    print('  (o HRPE reporta 25,2 GFLOPs; use a mesma convencao thop=MACs para comparar)')


def main():
    print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(),
          '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
    print('EXP =', EXP)
    print('CKPT =', CKPT)
    model, cfg = build_model()
    print('modelo construido; other_image_size=%s rootnet_image_size=%s'
          % (cfg.other_image_size, cfg.rootnet_image_size))
    measure_fps(model, cfg)
    measure_flops(model, cfg)
    print('\nPronto. Copie estas linhas e me envie para eu preencher a Secao 7.8.')


if __name__ == '__main__':
    main()
