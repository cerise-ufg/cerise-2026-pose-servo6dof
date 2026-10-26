#!/bin/bash
# uso: esperar.sh <container> [timeout_s] [linhas]  -- espera o container sair (ou timeout) e mostra o fim do log
C=$1; T=${2:-540}; N=${3:-25}
timeout $T bash -c "until [ \"\$(sudo docker inspect -f '{{.State.Running}}' $C 2>/dev/null)\" != true ]; do sleep 15; done"
echo "--- estado: $(sudo docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}}' $C 2>/dev/null)"
sudo docker logs $C 2>&1 | grep -v -i 'copyright\|nvidia corporation\|governing terms\|found at\|rights reserved\|caffe\|idiap\|google inc\|yangqing\|product-specific\|software license' | tail -n $N
