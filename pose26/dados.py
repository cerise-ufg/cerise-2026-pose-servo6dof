"""Dataset do servo6dof (formato convertido) + augmentations do pipeline de 2025 (feitas na GPU)."""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

CAMS = ["frontal", "diagonal"]


class Servo6dofDataset(torch.utils.data.Dataset):
    def __init__(self, dados, split, cams=("frontal", "diagonal"), com_profundidade=True):
        self.dados = dados
        z = np.load(os.path.join(dados, "index.npz"))
        pre = f"{split}_"
        self.meta = {k[len(pre):]: z[k] for k in z.files if k.startswith(pre)}
        self.K_full = z["K"].astype(np.float32)
        sel = np.isin(self.meta["cam"], [CAMS.index(c) for c in cams])
        self.idx = np.nonzero(sel)[0]
        self.rgb = np.load(os.path.join(dados, f"rgb256_{split}.npy"), mmap_mode="r")
        pdep = os.path.join(dados, f"depth256_{split}.npy")
        self.dep = np.load(pdep, mmap_mode="r") if (com_profundidade and os.path.exists(pdep)) else None
        self.crop = json.load(open(os.path.join(dados, "crop.json")))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        i = self.idx[j]
        m = self.meta
        out = {
            "rgb": torch.from_numpy(np.array(self.rgb[i])).permute(2, 0, 1),  # uint8 3x256x256
            "q": torch.from_numpy(m["q"][i]), "kp3d": torch.from_numpy(m["kp3d"][i]),
            "kp2d_crop": torch.from_numpy(m["kp2d_crop"][i]), "kp2d": torch.from_numpy(m["kp2d"][i]),
            "Kc": torch.from_numpy(m["Kc"][i]), "K": torch.from_numpy(self.K_full),
            "cam": int(m["cam"][i]), "id": int(m["ids"][i]), "garra": float(m["garra"][i]),
        }
        if self.dep is not None:
            out["depth"] = torch.from_numpy(np.array(self.dep[i])).unsqueeze(0)  # uint8 1x256x256
        return out


def k_value(Kc, lado=256.0):
    """k = sqrt(fx*fy*A_real/A_img) do RootNet (A_real = 1 m x 1 m). Com o recorte fixo, a 'bbox' e o proprio
    recorte 256x256 (nao usa keypoints GT para recortar, nem no teste)."""
    return torch.sqrt(Kc[:, 0, 0] * Kc[:, 1, 1] * 1000.0 * 1000.0 / (lado * lado))


# ------------- augmentations de 2025 (DreamDataset), versao vetorizada na GPU -------------
def _cinza(x):
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])


def augmentar_2025(x, g=None):
    """x: (B,3,H,W) float [0,1]. Reproduz: color jitter por canal (p=0.4), oclusao com ruido (p=0.5, ate 30%
    da area), nitidez (p=0.3, 0-50), contraste (p=0.3, 0.7-1.8), brilho (p=0.3, 0.7-1.8), cor (p=0.3, 0-4)."""
    B, _, H, W = x.shape
    dev = x.device
    r = lambda *s: torch.rand(*s, device=dev, generator=g)  # noqa: E731
    # color jitter
    m = (r(B) < 0.4).float().view(B, 1, 1, 1)
    f = 2 * r(B, 1, 1, 1)
    fac = (1 - f) + 2 * f * r(B, 3, 1, 1)
    x = m * (x * fac).clamp(0, 1) + (1 - m) * x
    # oclusao (retangulo de ruido uniforme)
    for b in torch.nonzero(r(B) < 0.5).flatten().tolist():
        area = float(r(1)) * 0.3 * H * W
        asp = float(r(1)) * 1.7 + 0.3
        h = int(min(H, max(4, (area * asp) ** 0.5)))
        w = int(min(W, max(4, (area / asp) ** 0.5)))
        y0 = int(r(1) * (H - h + 1))
        x0 = int(r(1) * (W - w + 1))
        x[b, :, y0:y0 + h, x0:x0 + w] = r(3, h, w)
    # nitidez: blend com versao suavizada
    m = (r(B) < 0.3).float().view(B, 1, 1, 1)
    k = torch.tensor([[1, 1, 1], [1, 5, 1], [1, 1, 1]], device=dev, dtype=x.dtype) / 13.0
    suave = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), k.view(1, 1, 3, 3).repeat(3, 1, 1, 1), groups=3)
    fs = 50 * r(B, 1, 1, 1)
    x = m * (suave + fs * (x - suave)).clamp(0, 1) + (1 - m) * x
    # contraste
    m = (r(B) < 0.3).float().view(B, 1, 1, 1)
    fc = 0.7 + 1.1 * r(B, 1, 1, 1)
    media = _cinza(x).mean(dim=(2, 3), keepdim=True)
    x = m * (media + fc * (x - media)).clamp(0, 1) + (1 - m) * x
    # brilho
    m = (r(B) < 0.3).float().view(B, 1, 1, 1)
    fb = 0.7 + 1.1 * r(B, 1, 1, 1)
    x = m * (x * fb).clamp(0, 1) + (1 - m) * x
    # cor (saturacao)
    m = (r(B) < 0.3).float().view(B, 1, 1, 1)
    fcor = 4 * r(B, 1, 1, 1)
    gr = _cinza(x)
    x = m * (gr + fcor * (x - gr)).clamp(0, 1) + (1 - m) * x
    return x
