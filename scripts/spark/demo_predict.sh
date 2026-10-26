#!/bin/bash
# demonstracao do modulo de inferencia numa imagem do TESTE + latencia ponta a ponta do PosePredictor
S=~/cerise2026/pose/src/repo/scripts/spark
ID=${1:-002079}
cp ~/cerise2026/pose/src/repo/scripts/latencia_predictor.py ~/cerise2026/pose/inferencia/ 2>/dev/null
bash $S/rodar.sh demo bash -c "python /workspace/pose/inferencia/predict.py --ckpt /workspace/pose/inferencia/modelo/pose_servo6dof.pt \
   --imagem /dados/dataset/${ID}_camera_frontal_image.png --camera frontal --saida /workspace/pose/inferencia/exemplo_frontal_${ID}.png --repeticoes 30 && \
   echo '--- verdade (juntas_reais, graus):' && python -c \"import json,math; d=json.load(open('/dados/dataset/${ID}.json')); print({k: round(math.degrees(v),2) for k,v in d['juntas_reais'].items()})\" && \
   python /workspace/pose/inferencia/latencia_predictor.py /workspace/pose/inferencia/modelo/pose_servo6dof.pt /dados/dataset /workspace/pose/inferencia/latencia_predictor.json"
bash $S/esperar.sh cerise26_pose_demo 600 60
