"""Baixa os pesos oficiais do Depth Anything V2 (Hugging Face do autor: depth-anything/*)."""
import sys
from huggingface_hub import hf_hub_download

enc = sys.argv[1] if len(sys.argv) > 1 else "vitl"
repo = {"vits": "Depth-Anything-V2-Small", "vitb": "Depth-Anything-V2-Base", "vitl": "Depth-Anything-V2-Large"}[enc]
p = hf_hub_download(f"depth-anything/{repo}", f"depth_anything_v2_{enc}.pth")
print(p)
