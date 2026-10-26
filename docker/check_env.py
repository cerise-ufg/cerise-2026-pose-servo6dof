import sys
import torch, torchvision, numpy
print("python", sys.version.split()[0], "torch", torch.__version__, "torchvision", torchvision.__version__, "numpy", numpy.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
for m in ["cv2", "pandas", "scipy", "matplotlib", "yaml", "tqdm", "lxml", "networkx", "trimesh",
          "easydict", "huggingface_hub", "PIL", "timm", "six"]:
    try:
        mod = __import__(m)
        print(m, getattr(mod, "__version__", "ok"))
    except Exception as e:
        print(m, "MISSING", type(e).__name__)
