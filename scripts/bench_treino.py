"""Micro-benchmark do passo de treino (fwd+bwd) do modelo completo: fp32/TF32 vs bf16 vs bf16+channels_last."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pose26.modelo import args_2025, ativar_amp, construir_full  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
bs = int(sys.argv[1]) if len(sys.argv) > 1 else 32
dev = "cuda"
for modo in ["fp32", "bf16", "bf16_cl"]:
    a = args_2025()
    a.imagenet_init = False
    os.environ["HPE_HRNET_W32"] = ""
    m, robot = construir_full(a, device=dev)
    if modo.startswith("bf16"):
        ativar_amp(m)
    if modo.endswith("cl"):
        m = m.to(memory_format=torch.channels_last)
    opt = torch.optim.Adam(m.parameters(), 1e-4)
    x = torch.rand(bs, 3, 256, 256, device=dev)
    d = torch.rand(bs, 1, 256, 256, device=dev)
    if modo.endswith("cl"):
        x = x.contiguous(memory_format=torch.channels_last)
        d = d.contiguous(memory_format=torch.channels_last)
    K = torch.tensor([[250.0, 0, 128], [0, 250.0, 128], [0, 0, 1]], device=dev).expand(bs, 3, 3).contiguous()
    kv = torch.full((bs,), 250.0 * 1000 / 256, device=dev)
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for i in range(14):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = m(x, x, d, kv, K=K)
        loss = out[0].pow(2).mean() + out[-1].pow(2).mean() + out[-2].pow(2).mean() + out[4].mean()
        opt.zero_grad(); loss.backward(); opt.step()
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    print(f"{modo:8s} bs={bs}: {sum(ts[4:])/len(ts[4:]):.3f} s/it  mem_pico={torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
    del m, opt
    torch.cuda.empty_cache()
