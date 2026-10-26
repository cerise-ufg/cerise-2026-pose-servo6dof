"""Profundidade monocular com Depth Anything V2 (como na variante MyNet de 2025), em lote na GPU.

Equivale ao DepthAnythingV2.infer_image (redimensiona para multiplo de 14 com lado >= input_size,
normaliza ImageNet, interpola a saida de volta), mas em lote e com autocast fp16.
Saida por imagem: profundidade relativa (disparidade) normalizada min-max para [0,1].
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402

CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def achar_pesos(enc="vitl"):
    from huggingface_hub import hf_hub_download
    repo = {"vits": "Depth-Anything-V2-Small", "vitb": "Depth-Anything-V2-Base", "vitl": "Depth-Anything-V2-Large"}[enc]
    return hf_hub_download(f"depth-anything/{repo}", f"depth_anything_v2_{enc}.pth")


class DAProfundidade:
    def __init__(self, enc="vitl", device="cuda", input_size=518, pesos=None):
        self.model = DepthAnythingV2(**CONFIGS[enc])
        sd = torch.load(pesos or achar_pesos(enc), map_location="cpu")
        self.model.load_state_dict(sd)
        self.model = self.model.to(device).eval()
        self.device = device
        self.input_size = input_size

    @torch.no_grad()
    def __call__(self, rgb_u8):
        """rgb_u8: tensor (B,H,W,3) uint8 RGB (ou (B,3,H,W) float 0-1) -> (B,1,H,W) float [0,1]"""
        if rgb_u8.dtype == torch.uint8:
            x = rgb_u8.to(self.device).permute(0, 3, 1, 2).float() / 255.0
        else:
            x = rgb_u8.to(self.device)
        B, _, H, W = x.shape
        s = self.input_size / min(H, W)
        h = int(max(round(H * s / 14), 1) * 14)
        w = int(max(round(W * s / 14), 1) * 14)
        if h < self.input_size:
            h += 14
        if w < self.input_size:
            w += 14
        xi = F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1)
        xi = (xi - MEAN.to(x.device)) / STD.to(x.device)
        with torch.autocast("cuda", dtype=torch.float16):
            d = self.model(xi)  # (B,h,w)
        d = F.interpolate(d.float().unsqueeze(1), size=(H, W), mode="bilinear", align_corners=True)
        mn = d.amin(dim=(2, 3), keepdim=True)
        mx = d.amax(dim=(2, 3), keepdim=True)
        return (d - mn) / (mx - mn + 1e-6)
