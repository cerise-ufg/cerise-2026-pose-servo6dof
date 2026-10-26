#!/bin/bash
# inspeciona os checkpoints de 2025 copiados para ~/cerise2026/pose/pesos
for f in best_model_auc.pk realsense_full_checkpoint.pk realsense_full_best_auc_copia1.pk; do
  echo "== $f"
  sudo docker run --rm -v ~/cerise2026/pose:/workspace/pose cerise26_pose:latest \
    python src/repo/scripts/inspecionar_ckpt_2025.py pesos/$f 2>&1 | grep -E '"epoch"|"auc_add"|ADD_60_mm_realsense"|AUC_ADD_realsense"|mean_joint_error_realsense"'
done
