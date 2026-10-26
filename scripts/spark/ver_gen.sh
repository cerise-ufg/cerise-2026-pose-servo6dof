#!/bin/bash
# leitura (somente) do andamento da geracao do dataset por worker
for w in 0 1 2 3 4 5 6 7; do
  echo "== w$w ($(stat -c %y ~/braco/gen_logs/w$w.log | cut -c12-19))"
  tail -n 3 ~/braco/gen_logs/w$w.log | cut -c1-200
done
cd ~/braco/dataset
echo "jsons por worker (id % 8):"
ls *.json | sed 's/\.json//' | awk '{c[$1%8]++} END{for(i=0;i<8;i++) printf "w%d=%d ", i, c[i]; print ""}'
echo "ultimo json: $(ls -t *.json | head -1) $(stat -c %y $(ls -t *.json | head -1) | cut -c12-19)"
pgrep -af 'gz sim|ign gazebo|gazebo' | wc -l
