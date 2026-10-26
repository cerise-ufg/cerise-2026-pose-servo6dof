#!/bin/bash
# Fiscalizacao (somente leitura) da volta da producao do Spark apos o braco_gen terminar.
# Registra estado dos containers/servicos de producao e faz GETs de saude locais.
echo "=== $(date '+%Y-%m-%d %H:%M:%S %z') ==="
echo "--- braco_gen: $(sudo docker ps -a --filter name=braco_gen --format '{{.Status}}')"
echo "--- containers de producao:"
sudo docker ps -a --filter name=vllm-server --filter name=vllm-embed --filter name=vllm-stt --filter name=qdrant \
  --format '{{.Names}}\t{{.Status}}'
echo "--- servicos systemd:"
for s in vania-brain ollama; do
  st=$(systemctl is-active $s 2>/dev/null); printf "%s\t%s\n" "$s" "${st:-?}"
done
echo "--- saude HTTP (GET local):"
for u in http://localhost:8000/health http://localhost:8001/health http://localhost:8002/health \
         http://localhost:6333/healthz http://localhost:8080/ http://localhost:11434/api/version; do
  code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "$u")
  printf "%-36s HTTP %s\n" "$u" "$code"
done
echo "--- memoria:"
free -g | head -2
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | head -12
echo "--- finalizador:"
ls -la ~/braco/finalizar.sh 2>/dev/null; ls -t ~/braco/*.log 2>/dev/null | head -3
