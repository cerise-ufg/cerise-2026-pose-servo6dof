"""Avaliacao no conjunto de TESTE: metricas (A8/A9), predicoes salvas, robustez sob perturbacoes e latencia.

uso:
  python scripts/avaliar.py --dados /dados/final --ckpt /runs/x/melhor.pt --saida /runs/x/aval [--robustez] [--latencia]
"""
import argparse
import io
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pose26.modelo import args_2025, ativar_amp, construir_full  # noqa: E402
from pose26.dados import Servo6dofDataset, k_value, CAMS  # noqa: E402
from pose26 import metricas as M  # noqa: E402
from depth_da import DAProfundidade  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def projetar(K, p):
    uvw = torch.einsum("bij,bkj->bki", K, p)
    return uvw[..., :2] / uvw[..., 2:3].clamp(min=0.05)


def carregar_modelo(ckpt, dev="cuda"):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = args_2025(**{k: v for k, v in ck.get("args_modelo", {}).items() if k not in ("aug",)})
    a.imagenet_init = False
    os.environ.setdefault("HPE_HRNET_W32", "")
    model, robot = construir_full(a, device=dev)
    model.load_state_dict(ck["model_state_dict"])
    if ck.get("amp", False):
        ativar_amp(model)
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model, robot, ck


@torch.no_grad()
def inferir(model, rgb01, dep01, Kc):
    kv = k_value(Kc)
    rgb01 = rgb01.contiguous(memory_format=torch.channels_last)
    dep01 = dep01.contiguous(memory_format=torch.channels_last)
    pp, prot, ptrans, puv, pdepth, puvd, pint, pfk = model(rgb01, rgb01, dep01, kv, K=Kc)
    return pp, pfk


# ---------------- perturbacoes (aplicadas na imagem de entrada 256x256, antes do DA e do modelo) -------------
def perturbar(x, tipo, s, rng, kp2dc=None):
    """x: (B,3,H,W) float [0,1] na GPU."""
    B, _, H, W = x.shape
    if tipo == "limpa":
        return x
    if tipo == "ruido_gaussiano":
        g = torch.Generator(device=x.device).manual_seed(int(rng.integers(1 << 30)))
        return (x + s * torch.randn(x.shape, device=x.device, generator=g)).clamp(0, 1)
    if tipo == "desfoque_movimento":
        k = int(s)
        ker = torch.zeros((k, k), device=x.device)
        ker[k // 2, :] = 1.0 / k  # movimento horizontal
        return F.conv2d(F.pad(x, (k // 2, k // 2, k // 2, k // 2), mode="replicate"), ker.view(1, 1, k, k).repeat(3, 1, 1, 1), groups=3)
    if tipo == "brilho":
        return (x * s).clamp(0, 1)
    if tipo == "contraste":
        m = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).mean(dim=(1, 2)).view(B, 1, 1, 1)
        return (m + s * (x - m)).clamp(0, 1)
    if tipo == "oclusao":
        y = x.clone()
        for b in range(B):
            kp = kp2dc[b].cpu().numpy()
            lo = np.clip(kp.min(0), 0, W - 1); hi = np.clip(kp.max(0), 1, W)
            bw, bh = max(hi[0] - lo[0], 8), max(hi[1] - lo[1], 8)
            area = s * bw * bh
            asp = rng.uniform(0.5, 2.0)
            w = int(min(W, max(2, np.sqrt(area * asp)))); h = int(min(H, max(2, np.sqrt(area / asp))))
            cx = rng.uniform(lo[0], hi[0]); cy = rng.uniform(lo[1], hi[1])
            x0 = int(np.clip(cx - w / 2, 0, W - w)); y0 = int(np.clip(cy - h / 2, 0, H - h))
            y[b, :, y0:y0 + h, x0:x0 + w] = 0.5  # recorte cinza (cutout)
        return y
    if tipo == "jpeg":
        arr = (x.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
        out = []
        for im in arr:
            ok, enc = cv2.imencode(".jpg", im[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(s)])
            out.append(cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1])
        return torch.from_numpy(np.stack(out)).to(x.device).permute(0, 3, 1, 2).float() / 255.0
    if tipo == "resolucao":
        r = max(8, int(round(H * s)))
        y = F.interpolate(x, size=(r, r), mode="area")
        return F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False).clamp(0, 1)
    raise ValueError(tipo)


ROBUSTEZ = {
    "ruido_gaussiano": [0.02, 0.05, 0.10, 0.20],
    "desfoque_movimento": [3, 7, 11, 15],
    "brilho": [0.4, 0.6, 1.4, 1.8],
    "contraste": [0.3, 0.6, 1.5, 2.0],
    "oclusao": [0.10, 0.20, 0.30],
    "jpeg": [50, 30, 15, 8],
    "resolucao": [0.5, 0.35, 0.25],
}


def avaliar_lista(model, ds, dev, da=None, tipo="limpa", s=None, sub=None, bs=32, seed=0, guardar=False):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ds)) if sub is None else sub
    P3, G3, P2, G2, PQ, GQ, CAM, IDS = [], [], [], [], [], [], [], []
    for i in range(0, len(idx), bs):
        itens = [ds[int(j)] for j in idx[i:i + bs]]
        rgb = torch.stack([t["rgb"] for t in itens]).to(dev).float() / 255.0
        Kc = torch.stack([t["Kc"] for t in itens]).to(dev)
        kp2dc = torch.stack([t["kp2d_crop"] for t in itens])
        x = perturbar(rgb, tipo, s, rng, kp2dc)
        if tipo == "limpa" and "depth" in itens[0] and da is None:
            dep = torch.stack([t["depth"] for t in itens]).to(dev).float() / 255.0
        else:  # profundidade DA recalculada sobre a imagem perturbada (pipeline completo)
            dep = (da(x) * 255).round() / 255.0
        pp, pfk = inferir(model, x, dep, Kc)
        K = torch.stack([t["K"] for t in itens]).to(dev)
        P3.append(pfk.cpu().numpy()); G3.append(torch.stack([t["kp3d"] for t in itens]).numpy())
        P2.append(projetar(K, pfk).cpu().numpy()); G2.append(torch.stack([t["kp2d"] for t in itens]).numpy())
        PQ.append(pp.cpu().numpy()); GQ.append(torch.stack([t["q"] for t in itens]).numpy())
        CAM += [t["cam"] for t in itens]; IDS += [t["id"] for t in itens]
    arr = {k: np.concatenate(v) for k, v in zip(["p3", "g3", "p2", "g2", "pq", "gq"], [P3, G3, P2, G2, PQ, GQ])}
    arr["cam"] = np.array(CAM); arr["id"] = np.array(IDS)
    r, e3i, e2i = M.resumo(arr["p3"], arr["g3"], arr["p2"], arr["g2"], arr["pq"], arr["gq"])
    arr["e3_img"], arr["e2_img"] = e3i, e2i
    return r, arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--split", default="teste")
    ap.add_argument("--cams", nargs="*", default=["frontal", "diagonal"])
    ap.add_argument("--robustez", action="store_true")
    ap.add_argument("--n_robustez", type=int, default=400, help="imagens do teste usadas na robustez (subconjunto fixo)")
    ap.add_argument("--latencia", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(a.saida, exist_ok=True)
    dev = "cuda"
    model, robot, ck = carregar_modelo(a.ckpt, dev)
    print(f"checkpoint: {a.ckpt} (etapa={ck.get('etapa')}, init={ck.get('init')}, epoca={ck.get('epoch', -1)+1})", flush=True)
    res = {"ckpt": a.ckpt, "epoca": ck.get("epoch", -1) + 1, "init": ck.get("init"), "split": a.split}
    ds = Servo6dofDataset(a.dados, a.split, cams=a.cams)
    t0 = time.time()
    r, arr = avaliar_lista(model, ds, dev)
    res["teste_todas_cameras"] = r
    for c in a.cams:
        sel = arr["cam"] == CAMS.index(c)
        rc, _, _ = M.resumo(arr["p3"][sel], arr["g3"][sel], arr["p2"][sel], arr["g2"][sel], arr["pq"][sel], arr["gq"][sel])
        res[f"teste_camera_{c}"] = rc
    np.savez_compressed(os.path.join(a.saida, f"predicoes_{a.split}{a.tag}.npz"), **arr)
    print(f"[{a.split}] {len(ds)} imagens em {time.time()-t0:.0f}s | AUC_ADD={r['AUC_ADD_0_100mm']:.4f} ADD@60mm={100*r['ADD_60mm']:.2f}% "
          f"erro3d={r['erro3d_kp_medio_mm']:.1f}mm (mediana {r['erro3d_kp_mediana_mm']:.1f}) MAE={r['MAE_medio_graus']:.2f} graus "
          f"PCK10px={100*r['PCK_kp_10px']:.1f}%", flush=True)
    print("MAE por junta (graus):", [round(v, 2) for v in r["MAE_juntas_graus"]], flush=True)

    da = None
    if a.robustez or a.latencia:
        da = DAProfundidade("vitl", dev)
    if a.robustez:
        sub = np.random.default_rng(123).choice(len(ds), size=min(a.n_robustez, len(ds)), replace=False)
        sub.sort()
        rob = {"n_imagens": int(len(sub)), "condicoes": {}}
        t0 = time.time()
        rl, _ = avaliar_lista(model, ds, dev, da=da, tipo="limpa", sub=sub)
        rob["limpa"] = {k: rl[k] for k in ("AUC_ADD_0_100mm", "ADD_60mm", "erro3d_kp_medio_mm", "MAE_medio_graus", "PCK_kp_10px", "erro2d_kp_medio_px")}
        print(f"[robustez] limpa (DA recalculado): AUC={rl['AUC_ADD_0_100mm']:.4f} MAE={rl['MAE_medio_graus']:.2f}", flush=True)
        for tipo, sevs in ROBUSTEZ.items():
            rob["condicoes"][tipo] = []
            for s in sevs:
                rr, _ = avaliar_lista(model, ds, dev, da=da, tipo=tipo, s=s, sub=sub, seed=7)
                ent = {"severidade": s, **{k: rr[k] for k in rob["limpa"]}}
                rob["condicoes"][tipo].append(ent)
                print(f"[robustez] {tipo:20s} s={s:<6} AUC={rr['AUC_ADD_0_100mm']:.4f} ADD60={100*rr['ADD_60mm']:.1f}% "
                      f"erro3d={rr['erro3d_kp_medio_mm']:.1f}mm MAE={rr['MAE_medio_graus']:.2f} PCK10={100*rr['PCK_kp_10px']:.1f}%  ({time.time()-t0:.0f}s)", flush=True)
        res["robustez"] = rob
        # amostras visuais das perturbacoes (primeira imagem do subconjunto)
        it = ds[int(sub[0])]
        x = it["rgb"].unsqueeze(0).to(dev).float() / 255.0
        rng = np.random.default_rng(7)
        mos = []
        for tipo, sevs in ROBUSTEZ.items():
            y = perturbar(x, tipo, sevs[-1] if tipo not in ("brilho", "contraste") else sevs[0], rng, it["kp2d_crop"].unsqueeze(0))
            mos.append((tipo, (y[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)))
        np.savez_compressed(os.path.join(a.saida, "amostras_perturbacoes.npz"), **{t: im for t, im in mos},
                            limpa=(x[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

    if a.latencia:
        it = ds[0]
        x = it["rgb"].unsqueeze(0).to(dev).float() / 255.0
        Kc = it["Kc"].unsqueeze(0).to(dev)
        tempos = {"da": [], "hpe": [], "total": []}
        for i in range(130):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            d = da(x)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            inferir(model, x, d, Kc)
            torch.cuda.synchronize(); t2 = time.perf_counter()
            if i >= 30:
                tempos["da"].append((t1 - t0) * 1000); tempos["hpe"].append((t2 - t1) * 1000); tempos["total"].append((t2 - t0) * 1000)
        lat = {k: {"media_ms": float(np.mean(v)), "mediana_ms": float(np.median(v)), "p95_ms": float(np.percentile(v, 95))} for k, v in tempos.items()}
        lat["fps_total"] = 1000.0 / lat["total"]["media_ms"]
        lat["fps_so_hpe"] = 1000.0 / lat["hpe"]["media_ms"]
        lat["config"] = "GB10, batch 1, 256x256, DA-V2 vitl (entrada 518, fp16 autocast) + HPE (backbones bf16 channels_last, cabecas/FK fp32); 100 medidas apos 30 de aquecimento"
        lat["mem_pico_GB"] = torch.cuda.max_memory_allocated() / 1e9
        res["latencia"] = lat
        print("[latencia]", json.dumps(lat), flush=True)
    json.dump(res, open(os.path.join(a.saida, f"metricas_{a.split}{a.tag}.json"), "w"), indent=1)
    print("salvo:", os.path.join(a.saida, f"metricas_{a.split}{a.tag}.json"))


if __name__ == "__main__":
    main()
