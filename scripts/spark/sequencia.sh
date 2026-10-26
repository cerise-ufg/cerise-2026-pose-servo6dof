#!/bin/bash
# Pipeline SEQUENCIAL (memoria apertada com a producao de volta): nunca dois jobs pesados da POSE ao mesmo tempo.
# transfer (ja rodando) -> avaliacao transfer (+robustez, latencia) -> baseline imagenet -> avaliacao baseline
# -> avaliacao entre cameras -> exportacao + teste do modulo de inferencia -> READY.json
S=~/cerise2026/pose/src/repo/scripts/spark
LOG=~/cerise2026/pose/logs
FIS=$LOG/fiscalizacao_producao.log
P() { echo "[seq $(date '+%H:%M:%S')] $*"; }
espera_mem() {  # espera ter >= $1 GB "available" e registra reinicios do vllm
  for i in $(seq 1 60); do
    av=$(free -g | awk '/Mem:/{print $7}')
    [ "$av" -ge "$1" ] && break
    P "aguardando memoria: available=${av} GB (< $1)"; sleep 30
  done
  P "available=$(free -g | awk '/Mem:/{print $7}') GB; reinicios: $(sudo docker inspect -f '{{.Name}}={{.RestartCount}}' vllm-server vllm-embed vllm-stt qdrant | tr '\n' ' ')"
}
roda() {  # roda um container e espera terminar
  nome=$1; shift
  bash $S/rodar.sh $nome "$@" >/dev/null
  sudo docker wait cerise26_pose_$nome >/dev/null
  P "$nome terminou: $(sudo docker inspect -f '{{.State.ExitCode}}' cerise26_pose_$nome)"
}
P "inicio"
sudo docker wait cerise26_pose_full_transfer >/dev/null
P "transfer terminou"; bash $S/fiscalizar_producao.sh >> $FIS 2>&1
espera_mem 22
roda aval_transfer python scripts/avaliar.py --dados /dados/final --ckpt /runs/final_full_transfer/melhor.pt \
  --saida /runs/final_full_transfer/aval --robustez --n_robustez 400 --latencia
espera_mem 22
roda full_imagenet python scripts/treinar.py --dados /dados/final --saida /runs/final_full_imagenet --etapa full --init imagenet \
  --rootnet /runs/etapa1_depthnet_imagenet_v2/melhor.pt --epocas 20 --bs 32 --workers 4 --tempo_max_min ${TMAX_BASE:-50}
bash $S/fiscalizar_producao.sh >> $FIS 2>&1
espera_mem 22
roda aval_imagenet python scripts/avaliar.py --dados /dados/final --ckpt /runs/final_full_imagenet/melhor.pt \
  --saida /runs/final_full_imagenet/aval --robustez --n_robustez 400
espera_mem 18
roda aval_xcam python scripts/avaliar.py --dados /dados/final --ckpt /runs/xcam_frontal_transfer/melhor.pt --saida /runs/xcam_frontal_transfer/aval
espera_mem 18
roda export bash -c "python scripts/exportar_modelo.py --ckpt /runs/final_full_transfer/melhor.pt --dados /dados/final \
   --saida /workspace/pose/inferencia/modelo/pose_servo6dof.pt --metricas /runs/final_full_transfer/aval/metricas_teste.json"
P "fim"; bash $S/fiscalizar_producao.sh >> $FIS 2>&1
