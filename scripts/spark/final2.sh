#!/bin/bash
# Segunda fase: espera o treino de transferencia terminar; sobe o baseline (ImageNet) com orcamento menor
# e, em paralelo, as avaliacoes do modelo de transferencia (teste + robustez + latencia) e do xcam,
# exportacao do pacote de inferencia e teste do predict.py.
S=~/cerise2026/pose/src/repo/scripts/spark
LOG=~/cerise2026/pose/logs
echo "[final2] inicio $(date)"
sudo docker wait cerise26_pose_full_transfer
echo "[final2] transfer terminou $(date)"; free -g | head -2
AV=$(free -g | awk '/Mem:/{print $7}')
echo "[final2] memoria disponivel: ${AV} GB"
bash $S/rodar.sh full_imagenet python scripts/treinar.py --dados /dados/final --saida /runs/final_full_imagenet --etapa full --init imagenet \
  --rootnet /runs/etapa1_depthnet_imagenet_v2/melhor.pt --epocas 20 --bs 32 --workers 4 --tempo_max_min ${TMAX_B:-45}
sleep 60
AV=$(free -g | awk '/Mem:/{print $7}')
echo "[final2] memoria disponivel apos baseline: ${AV} GB"
for i in $(seq 1 40); do [ "$(free -g | awk '/Mem:/{print $7}')" -ge 18 ] && break; echo "[final2] aguardando memoria ($(date +%T))"; sleep 60; done
bash $S/rodar.sh aval_transfer bash -c "python scripts/avaliar.py --dados /dados/final --ckpt /runs/final_full_transfer/melhor.pt --saida /runs/final_full_transfer/aval --robustez --n_robustez 400 --latencia && \
  python scripts/avaliar.py --dados /dados/final --ckpt /runs/xcam_frontal_transfer/melhor.pt --saida /runs/xcam_frontal_transfer/aval && \
  python scripts/exportar_modelo.py --ckpt /runs/final_full_transfer/melhor.pt --dados /dados/final --saida /workspace/pose/inferencia/modelo/pose_servo6dof.pt --metricas /runs/final_full_transfer/aval/metricas_teste.json && \
  python /workspace/pose/inferencia/predict.py --ckpt /workspace/pose/inferencia/modelo/pose_servo6dof.pt --imagem /dados/dataset/004999_camera_diagonal_image.png --camera diagonal --saida /workspace/pose/inferencia/exemplo_diagonal_004999.png --repeticoes 30"
sudo docker wait cerise26_pose_aval_transfer
sudo docker logs cerise26_pose_aval_transfer > $LOG/aval_transfer.log 2>&1
echo "[final2] avaliacao transfer terminou $(date)"
sudo docker wait cerise26_pose_full_imagenet
echo "[final2] baseline terminou $(date)"
bash $S/rodar.sh aval_imagenet python scripts/avaliar.py --dados /dados/final --ckpt /runs/final_full_imagenet/melhor.pt --saida /runs/final_full_imagenet/aval --robustez --n_robustez 400
sudo docker wait cerise26_pose_aval_imagenet
sudo docker logs cerise26_pose_aval_imagenet > $LOG/aval_imagenet.log 2>&1
bash $S/fiscalizar_producao.sh >> $LOG/fiscalizacao_producao.log 2>&1
echo "[final2] fim $(date)"
