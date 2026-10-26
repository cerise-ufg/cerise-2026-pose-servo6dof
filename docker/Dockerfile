# Imagem da frente POSE (CERISE 2026) - DGX Spark / GB10 (aarch64, sm_121)
# Base NGC PyTorch 26.05 (torch 2.12 / CUDA para Blackwell) + dependencias do codigo de 2025
# (HoRoPose/HPE + variante MyNet com Depth Anything V2).
FROM nvcr.io/nvidia/pytorch:26.05-py3

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
# opencv (I/O e augmentations), lxml/trimesh/six (urdfpytorch - cinematica direta diferenciavel),
# easydict (config do HRNet). O resto (pandas, scipy, matplotlib, networkx, PIL, hf_hub) ja vem na base.
RUN pip install opencv-python-headless==4.11.0.86 lxml trimesh easydict six \
 && python -c "import cv2, lxml, trimesh, easydict; print('ok', cv2.__version__)"

WORKDIR /workspace/pose
