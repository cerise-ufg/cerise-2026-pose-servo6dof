#!/bin/bash
# exporta um checkpoint e testa o modulo de inferencia numa imagem do teste
# uso: teste_predict.sh <ckpt_em_/runs> <dir_dados_em_/dados> <saida_em_/workspace/pose/inferencia/...> <id_pose> <camera>
CK=$1; DADOS=$2; OUT=$3; ID=$4; CAM=${5:-frontal}
S=~/cerise2026/pose/src/repo/scripts/spark
bash $S/rodar.sh predict bash -c "python scripts/exportar_modelo.py --ckpt $CK --dados $DADOS --saida $OUT && \
  python /workspace/pose/inferencia/predict.py --ckpt $OUT --imagem /dados/dataset/$(printf %06d $ID)_camera_${CAM}_image.png \
  --camera $CAM --saida /workspace/pose/inferencia/exemplo_${CAM}_$(printf %06d $ID).png --repeticoes 20"
bash $S/esperar.sh cerise26_pose_predict 300 40
