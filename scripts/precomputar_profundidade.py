"""Pre-computa a profundidade Depth Anything V2 (vitl) dos recortes 256x256 de cada split.
Saida: depth256_{split}.npy (N,256,256) uint8 = profundidade relativa normalizada * 255.
--reusar <dir>: copia a profundidade ja calculada de outra conversao (mesmos recortes, chave (id, camera)).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hpe2025"))
from depth_da import DAProfundidade  # noqa: E402


def mapa_existente(d):
    m = {}
    if not d or not os.path.exists(os.path.join(d, "index.npz")):
        return m
    z = np.load(os.path.join(d, "index.npz"))
    for split in ["treino", "val", "teste"]:
        p = os.path.join(d, f"depth256_{split}.npy")
        if f"{split}_ids" not in z.files or not os.path.exists(p):
            continue
        dep = np.load(p, mmap_mode="r")
        for r, (i, c) in enumerate(zip(z[f"{split}_ids"], z[f"{split}_cam"])):
            m[(int(i), int(c))] = (dep, r)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", required=True)
    ap.add_argument("--enc", default="vitl")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--splits", nargs="*", default=["treino", "val", "teste"])
    ap.add_argument("--reusar", default=None)
    a = ap.parse_args()
    velho = mapa_existente(a.reusar)
    print(f"profundidades reaproveitaveis: {len(velho)}", flush=True)
    da = DAProfundidade(a.enc)
    z = np.load(os.path.join(a.dados, "index.npz"))
    for split in a.splits:
        p = os.path.join(a.dados, f"rgb256_{split}.npy")
        if not os.path.exists(p):
            continue
        rgb = np.load(p, mmap_mode="r")
        ids, cams = z[f"{split}_ids"], z[f"{split}_cam"]
        out = np.lib.format.open_memmap(os.path.join(a.dados, f"depth256_{split}.npy"), mode="w+", dtype=np.uint8, shape=rgb.shape[:3])
        falta = []
        for r in range(len(rgb)):
            k = (int(ids[r]), int(cams[r]))
            if k in velho:
                dep, rr = velho[k]
                out[r] = dep[rr]
            else:
                falta.append(r)
        t0 = time.time()
        for i in range(0, len(falta), a.bs):
            rs = falta[i:i + a.bs]
            d = da(torch.from_numpy(np.array(rgb[rs])))
            out[rs] = (d[:, 0].clamp(0, 1) * 255).round().byte().cpu().numpy()
        out.flush()
        dt = time.time() - t0
        print(f"{split}: {len(rgb)} imagens ({len(rgb)-len(falta)} reaproveitadas, {len(falta)} calculadas em {dt:.1f}s"
              f"{'' if not falta else f', {len(falta)/max(dt,1e-6):.1f} img/s'})  mem_pico={torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
