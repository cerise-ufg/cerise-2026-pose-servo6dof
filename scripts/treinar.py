"""Treino do modelo de 2025 (HPE + Depth Anything V2) no novo dataset do servo6dof.

Etapas (metodologia de 2025):
  depthnet : RootNet/DepthNet (HRNet-32 + ramo de profundidade DA-V2) -> profundidade da raiz (L1, mm)
  full     : modelo completo (ResNet-50 integral + RootNet + regressao iterativa de juntas/rotacao + FK)
Inicializacao:
  --init 2025     : pesos do modelo final de 2025 (Panda) onde as formas casam (transfer learning)
  --init imagenet : so ImageNet (ResNet-50 torchvision, HRNet-W32 oficial) -> baseline
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pose26.modelo import (REF_KP, args_2025, ativar_amp, carregar_parcial, construir_depthnet,  # noqa: E402
                           construir_full, ler_ckpt)
from pose26.dados import Servo6dofDataset, augmentar_2025, k_value  # noqa: E402
from pose26 import metricas as M  # noqa: E402
from utils.geometries import rotmat_to_rot6d  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def projetar(K, p):
    uvw = torch.einsum("bij,bkj->bki", K, p)
    return uvw[..., :2] / uvw[..., 2:3].clamp(min=0.05)


def para_dev(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def gt_rot_raiz(robot, q, T_cw):
    TWL = robot.get_TWL(q)
    R = T_cw[:, :3, :3] @ TWL[:, REF_KP, :3, :3]
    return rotmat_to_rot6d(R)


def passo_full(model, robot, b, a, T_cw_cams, treino):
    rgb = b["rgb"].float() / 255.0
    if treino and a.aug:
        rgb = augmentar_2025(rgb)
    dep = b["depth"].float() / 255.0
    rgb = rgb.contiguous(memory_format=torch.channels_last)
    dep = dep.contiguous(memory_format=torch.channels_last)
    Kc, q, kp3d, kp2dc = b["Kc"], b["q"], b["kp3d"], b["kp2d_crop"]
    T_cw = T_cw_cams[b["cam"]]
    kv = k_value(Kc)
    pp, prot, ptrans, puv, pdepth, puvd, pint, pfk = model(rgb, rgb, dep, kv, K=Kc)
    vis = ((kp2dc >= 0) & (kp2dc < 256)).all(-1).float()
    gt_trans = kp3d[:, REF_KP]
    gt_rot = gt_rot_raiz(robot, q, T_cw)
    L = {}
    L["pose"] = torch.nn.functional.mse_loss(pp, q)
    L["rot"] = torch.nn.functional.mse_loss(prot, gt_rot)
    L["depth"] = torch.nn.functional.l1_loss(pdepth, gt_trans[:, 2:3])
    e_uv = torch.norm((puv - kp2dc[:, REF_KP]) / 256.0, dim=1) * vis[:, REF_KP]
    L["uv"] = e_uv.sum() / vis[:, REF_KP].sum().clamp(min=1)
    e_t = torch.norm(ptrans - gt_trans, dim=1)
    L["trans"] = e_t.mean()
    if L["trans"] > 0.5:
        L["trans"] = (e_t * torch.exp(-20.0 * e_t).detach()).mean()
    L["kp3d"] = torch.norm(pfk - kp3d, dim=2).mean()
    e2 = torch.norm(projetar(Kc, pfk) / 256.0 - kp2dc / 256.0, dim=2) * vis
    L["kp2d"] = e2.sum() / vis.sum().clamp(min=1)
    L["kp3d_int"] = torch.norm(pint - kp3d, dim=2).mean()
    e2i = torch.norm(projetar(Kc, pint) / 256.0 - kp2dc / 256.0, dim=2) * vis
    L["kp2d_int"] = e2i.sum() / vis.sum().clamp(min=1)
    L["align"] = torch.norm(pfk - pint, dim=2).mean()
    tot = (a.pose_loss_weight * L["pose"] + a.rot_loss_weight * L["rot"] + a.uv_loss_weight * L["uv"]
           + a.depth_loss_weight * L["depth"] + a.trans_loss_weight * L["trans"] + a.kp2d_loss_weight * L["kp2d"]
           + a.kp3d_loss_weight * L["kp3d"] + a.kp2d_int_loss_weight * L["kp2d_int"]
           + a.kp3d_int_loss_weight * L["kp3d_int"] + a.align_3d_loss_weight * L["align"])
    return tot, L, (pp, pfk)


def passo_depthnet(model, b, treino, aug):
    rgb = b["rgb"].float() / 255.0
    if treino and aug:
        rgb = augmentar_2025(rgb)
    dep = b["depth"].float() / 255.0
    rgb = rgb.contiguous(memory_format=torch.channels_last)
    dep = dep.contiguous(memory_format=torch.channels_last)
    kv = k_value(b["Kc"])
    pd = model(rgb, dep, kv)  # mm
    gt = b["kp3d"][:, REF_KP, 2:3] * 1000.0
    return torch.nn.functional.l1_loss(pd, gt), pd, gt


@torch.no_grad()
def avaliar_full(model, robot, loader, a, T_cw_cams, dev, max_lotes=0):
    model.eval()
    P3, G3, P2, G2, PQ, GQ = [], [], [], [], [], []
    for i, b in enumerate(loader):
        if max_lotes and i >= max_lotes:
            break
        b = para_dev(b, dev)
        _, _, (pp, pfk) = passo_full(model, robot, b, a, T_cw_cams, treino=False)
        P3.append(pfk.cpu().numpy()); G3.append(b["kp3d"].cpu().numpy())
        P2.append(projetar(b["K"], pfk).cpu().numpy()); G2.append(b["kp2d"].cpu().numpy())
        PQ.append(pp.cpu().numpy()); GQ.append(b["q"].cpu().numpy())
    r, _, _ = M.resumo(np.concatenate(P3), np.concatenate(G3), np.concatenate(P2), np.concatenate(G2),
                       np.concatenate(PQ), np.concatenate(GQ))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--etapa", choices=["depthnet", "full"], required=True)
    ap.add_argument("--init", choices=["2025", "imagenet"], required=True)
    # estado final do treino de 2025 (MyNet/experiments/panda/realsense_full/checkpoint.pk, epoca 440;
    # ADD@60mm RealSense = 83,1%). O best_model_auc.pk do mesmo treino e da epoca 207 (69,4%).
    ap.add_argument("--ckpt_2025", default="/workspace/pose/pesos/realsense_full_checkpoint.pk")
    ap.add_argument("--rootnet", default=None, help="checkpoint da etapa depthnet para o modelo completo")
    ap.add_argument("--epocas", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--decaimento", type=float, default=0.9, help="decaimento exponencial do lr por epoca")
    ap.add_argument("--inicio_decaimento", type=int, default=3)
    ap.add_argument("--cams_treino", nargs="*", default=["frontal", "diagonal"])
    ap.add_argument("--cams_val", nargs="*", default=["frontal", "diagonal"])
    ap.add_argument("--sem_aug", action="store_true")
    ap.add_argument("--sem_amp", action="store_true", help="desliga bf16 nos backbones")
    ap.add_argument("--max_iters", type=int, default=0, help="limite de iteracoes por epoca (teste rapido)")
    ap.add_argument("--max_val", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tempo_max_min", type=float, default=0, help="para o treino apos N minutos")
    ap.add_argument("--retomar", default=None, help="checkpoint (ultimo.pt) para continuar o treino (pesos do modelo)")
    ap.add_argument("--epoca_inicial", type=int, default=0, help="epocas ja feitas (continua numeracao e agenda do lr)")
    a_cli = ap.parse_args()
    os.makedirs(a_cli.saida, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(808); np.random.seed(808)  # mesma semente do train_full.py de 2025
    a = args_2025(lr=a_cli.lr, batch_size=a_cli.bs)
    a.aug = not a_cli.sem_aug
    log = open(os.path.join(a_cli.saida, "treino.log"), "a")

    def P(*s):
        msg = " ".join(str(x) for x in s)
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    P(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')}  etapa={a_cli.etapa} init={a_cli.init}  args={vars(a_cli)}")
    ds_tr = Servo6dofDataset(a_cli.dados, "treino", cams=a_cli.cams_treino)
    ds_va = Servo6dofDataset(a_cli.dados, "val", cams=a_cli.cams_val)
    P(f"treino: {len(ds_tr)} imagens | val: {len(ds_va)} imagens")
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=a_cli.bs, shuffle=True, num_workers=a_cli.workers,
                                        drop_last=True, pin_memory=True, persistent_workers=True)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=64, shuffle=False, num_workers=a_cli.workers, pin_memory=True)
    z = np.load(os.path.join(a_cli.dados, "index.npz"))
    T_cw_cams = torch.tensor(np.stack([z["T_cw_frontal"], z["T_cw_diagonal"]]), dtype=torch.float32, device=dev)

    info_transfer = None
    if a_cli.etapa == "depthnet":
        model = construir_depthnet(dev)
        if a_cli.init == "2025":
            sd = ler_ckpt(a_cli.ckpt_2025)["model_state_dict"]
            # rootnet_backbone.* (modelo completo de 2025) -> backbone.* (RootNet)
            info_transfer = carregar_parcial(model, sd, {"rootnet_backbone.": "backbone."}, log=P)
    else:
        a.imagenet_init = True
        model, robot = construir_full(a, device=dev)
        if a_cli.init == "2025":
            sd = ler_ckpt(a_cli.ckpt_2025)["model_state_dict"]
            sd = {k: v for k, v in sd.items() if k not in ("init_pose", "init_rot")}
            info_transfer = carregar_parcial(model, sd, log=P)
        if a_cli.rootnet:
            sdr = torch.load(a_cli.rootnet, map_location="cpu", weights_only=False)["model_state_dict"]
            # mesma regra do get_rootNetwithRegInt_model de 2025: backbone.* -> rootnet_backbone.*
            st = carregar_parcial(model, sdr, {"backbone.": "rootnet_backbone."}, log=P)
            P(f"[rootnet etapa 1] {st['tensores_carregados']} tensores do DepthNet carregados")
    if info_transfer:
        json.dump(info_transfer, open(os.path.join(a_cli.saida, "transfer_info.json"), "w"), indent=1, default=str)
    if a_cli.retomar:
        sdr = torch.load(a_cli.retomar, map_location="cpu", weights_only=False)["model_state_dict"]
        model.load_state_dict(sdr)
        P(f"[retomada] pesos de {a_cli.retomar}; continuando da epoca {a_cli.epoca_inicial} (estado do Adam reiniciado)")
    model.to(dev)
    if not a_cli.sem_amp:
        ativar_amp(model)
        model = model.to(memory_format=torch.channels_last)
        P("precisao mista bf16 + channels_last ativados nos backbones")
    n_par = sum(p.numel() for p in model.parameters())
    P(f"parametros treinaveis: {n_par/1e6:.2f} M")
    opt = torch.optim.Adam(model.parameters(), lr=a_cli.lr)
    e0 = a_cli.epoca_inicial
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: 1.0 if e + e0 < a_cli.inicio_decaimento else a_cli.decaimento ** (e + e0 - a_cli.inicio_decaimento + 1))
    csv_path = os.path.join(a_cli.saida, "curvas.csv")
    melhor = -1e9
    t_ini = time.time()
    for ep in range(e0, a_cli.epocas):
        model.train()
        t0 = time.time()
        soma, n = {}, 0
        for it, b in enumerate(dl_tr):
            if a_cli.max_iters and it >= a_cli.max_iters:
                break
            b = para_dev(b, dev)
            opt.zero_grad(set_to_none=True)
            if a_cli.etapa == "full":
                loss, L, _ = passo_full(model, robot, b, a, T_cw_cams, treino=True)
                ks = list(L.keys())
                vals = torch.stack([L[k].detach().float().reshape(()) for k in ks]).cpu().tolist()
                Ld = dict(zip(ks, vals))
            else:
                loss, _, _ = passo_depthnet(model, b, True, a.aug)
                Ld = {"depth_mm": float(loss)}
            if not torch.isfinite(loss):
                P(f"[aviso] perda nao finita no it {it}; lote ignorado")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip_gradient if a_cli.etapa == "full" else 1.0)
            opt.step()
            Ld["total"] = float(loss.detach())
            for k, v in Ld.items():
                soma[k] = soma.get(k, 0.0) + v
            n += 1
            if it % 50 == 0:
                P(f"ep {ep+1} it {it}/{len(dl_tr)} loss={float(loss):.4f} {time.time()-t0:.0f}s mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB")
        sched.step()
        tr = {f"treino_{k}": v / max(n, 1) for k, v in soma.items()}
        t_tr = time.time() - t0
        if a_cli.etapa == "full":
            r = avaliar_full(model, robot, dl_va, a, T_cw_cams, dev, a_cli.max_val)
            linha = {"epoca": ep + 1, **tr, "val_AUC_ADD": r["AUC_ADD_0_100mm"], "val_ADD_60mm": r["ADD_60mm"],
                     "val_erro3d_mm": r["erro3d_kp_medio_mm"], "val_erro2d_px": r["erro2d_kp_medio_px"],
                     "val_MAE_graus": r["MAE_medio_graus"], "val_PCK_kp_10px": r["PCK_kp_10px"]}
            score = r["AUC_ADD_0_100mm"] - 1e-4 * r["erro3d_kp_medio_mm"]
        else:
            model.eval()
            errs = []
            with torch.no_grad():
                for i, b in enumerate(dl_va):
                    if a_cli.max_val and i >= a_cli.max_val:
                        break
                    b = para_dev(b, dev)
                    _, pd, gt = passo_depthnet(model, b, False, False)
                    errs.append((pd - gt).abs().cpu().numpy())
            e = np.concatenate(errs)
            linha = {"epoca": ep + 1, **tr, "val_erro_prof_raiz_mm": float(e.mean()), "val_mediana_prof_raiz_mm": float(np.median(e))}
            score = -float(e.mean())
        linha.update({"lr": opt.param_groups[0]["lr"], "tempo_treino_s": round(t_tr, 1), "tempo_total_s": round(time.time() - t0, 1)})
        novo = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(linha.keys()))
            if novo:
                w.writeheader()
            w.writerow(linha)
        P("EPOCA", json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in linha.items()}))
        ck = {"model_state_dict": model.state_dict(), "epoch": ep, "etapa": a_cli.etapa, "init": a_cli.init,
              "metricas_val": linha, "args_modelo": vars(a), "amp": not a_cli.sem_amp}
        torch.save(ck, os.path.join(a_cli.saida, "ultimo.pt"))
        if score > melhor:
            melhor = score
            torch.save(ck, os.path.join(a_cli.saida, "melhor.pt"))
            P(f"  -> novo melhor (epoca {ep+1})")
        if a_cli.tempo_max_min and (time.time() - t_ini) / 60 > a_cli.tempo_max_min:
            P(f"tempo maximo atingido ({a_cli.tempo_max_min} min) - parando apos a epoca {ep+1}")
            break
    P(f"=== fim: {time.strftime('%H:%M:%S')}  total {(time.time()-t_ini)/60:.1f} min")


if __name__ == "__main__":
    main()
