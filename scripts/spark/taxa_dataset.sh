#!/bin/bash
# taxa de geracao do dataset (somente leitura): jsons por intervalo de 10 min
cd ~/braco/dataset
now=$(date +%s)
for m in 10 20 30 40 60; do
  n=$(find . -maxdepth 1 -name '*.json' -newermt "@$((now - m*60))" | wc -l)
  echo "ultimos ${m} min: $n jsons"
done
echo "total: $(ls *.json | wc -l)"
uptime
