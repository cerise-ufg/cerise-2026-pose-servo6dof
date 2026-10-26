#!/bin/bash
# memoria de GPU por processo -> container (somente leitura)
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | while IFS=', ' read pid mem; do
  cid=$(grep -o 'docker-[0-9a-f]\{64\}' /proc/$pid/cgroup 2>/dev/null | head -1 | sed 's/docker-//')
  nome=$( [ -n "$cid" ] && sudo docker inspect -f '{{.Name}}' $cid 2>/dev/null || echo host)
  echo "$pid ${mem}MiB $nome"
done
free -g | head -2
