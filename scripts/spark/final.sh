#!/bin/bash
# Orquestrador: espera o braco_gen terminar, fiscaliza a volta da producao, converte o dataset completo e
# dispara os treinos finais (etapa 2) depois que o vllm-server estiver saudavel.
# Rodar destacado: setsid nohup bash final.sh > ~/cerise2026/pose/logs/final.log 2>&1 < /dev/null &
S=~/cerise2026/pose/src/repo/scripts/spark
LOG=~/cerise2026/pose/logs
FIS=$LOG/fiscalizacao_producao.log
echo "[final] inicio $(date)"
until [ "$(sudo docker inspect -f '{{.State.Running}}' braco_gen 2>/dev/null)" != "true" ]; do sleep 30; done
echo "[final] braco_gen saiu: $(sudo docker ps -a --filter name=braco_gen --format '{{.Status}}') em $(date)"
echo "[final] jsons: $(ls ~/braco/dataset/*.json | wc -l)"
bash $S/fiscalizar_producao.sh >> $FIS 2>&1
# conversao do dataset completo (mesmos recortes do parcial2; reaproveita profundidades ja calculadas)
bash $S/rodar.sh conv bash -c "nice -n 10 python scripts/converter_dataset.py --dataset /dados/dataset --saida /dados/final --workers 6 --crop_json /dados/parcial2/crop.json && \
  python scripts/precomputar_profundidade.py --dados /dados/final --bs 16 --reusar /dados/parcial2"
# fiscaliza a producao enquanto converte
sleep 120; bash $S/fiscalizar_producao.sh >> $FIS 2>&1
sudo docker wait cerise26_pose_conv
sudo docker logs cerise26_pose_conv 2>&1 | grep -E 'ids completos|treino:|val:|teste:|recortes' | tee -a $LOG/conversao_final.txt
# espera o vllm-server responder (ate 15 min) antes de subir os treinos pesados
for i in $(seq 1 30); do
  c=$(curl -s -o /dev/null -m 5 -w '%{http_code}' http://localhost:8000/health)
  [ "$c" = "200" ] && break
  sleep 30
done
echo "[final] vllm-server /health = $c em $(date)"
bash $S/fiscalizar_producao.sh >> $FIS 2>&1
AV=$(free -g | awk '/Mem:/{print $7}')
echo "[final] memoria disponivel: ${AV} GB"
# etapa 2 - transfer learning (pesos de 2025) e baseline (ImageNet), cada um com o seu DepthNet da etapa 1
bash $S/rodar.sh full_transfer python scripts/treinar.py --dados /dados/final --saida /runs/final_full_transfer --etapa full --init 2025 \
  --rootnet /runs/etapa1_depthnet_transfer_v2/melhor.pt --epocas ${EPOCAS:-12} --bs 32 --workers 4 --tempo_max_min ${TMAX:-75}
sleep 90
AV=$(free -g | awk '/Mem:/{print $7}')
echo "[final] memoria disponivel apos subir transfer: ${AV} GB"
if [ "$AV" -ge 22 ]; then
  bash $S/rodar.sh full_imagenet python scripts/treinar.py --dados /dados/final --saida /runs/final_full_imagenet --etapa full --init imagenet \
    --rootnet /runs/etapa1_depthnet_imagenet_v2/melhor.pt --epocas ${EPOCAS:-12} --bs 32 --workers 4 --tempo_max_min ${TMAX:-75}
  echo "[final] baseline iniciado em paralelo $(date)"
else
  echo "[final] memoria apertada: baseline sera iniciado depois do transfer"
  sudo docker wait cerise26_pose_full_transfer
  bash $S/rodar.sh full_imagenet python scripts/treinar.py --dados /dados/final --saida /runs/final_full_imagenet --etapa full --init imagenet \
    --rootnet /runs/etapa1_depthnet_imagenet_v2/melhor.pt --epocas ${EPOCAS:-12} --bs 32 --workers 4 --tempo_max_min ${TMAX:-75}
fi
sleep 600; bash $S/fiscalizar_producao.sh >> $FIS 2>&1
echo "[final] orquestrador terminou $(date)"
