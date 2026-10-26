#!/bin/bash
# uso: converter.sh <saida_em_~/cerise2026/pose/dados> [crop_json_reuso (caminho no container)] [dir_reuso_profundidade (no container)]
OUT=$1; CROP=$2; REUSO=$3
S=~/cerise2026/pose/src/repo/scripts/spark
ARGS=""
[ -n "$CROP" ] && ARGS="--crop_json $CROP"
RARG=""
[ -n "$REUSO" ] && RARG="--reusar $REUSO"
bash $S/rodar.sh conv bash -c "nice -n 10 python scripts/converter_dataset.py --dataset /dados/dataset --saida /dados/$OUT --workers 6 $ARGS && \
  python scripts/precomputar_profundidade.py --dados /dados/$OUT --bs 16 $RARG"
bash $S/esperar.sh cerise26_pose_conv 1200 14
