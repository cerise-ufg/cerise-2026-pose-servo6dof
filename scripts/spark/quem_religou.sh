#!/bin/bash
# horarios de (re)inicio da producao (somente leitura)
for c in vllm-server vllm-embed vllm-stt qdrant braco_gen; do
  echo "$c: $(sudo docker inspect -f 'StartedAt={{.State.StartedAt}} FinishedAt={{.State.FinishedAt}} Restart={{.HostConfig.RestartPolicy.Name}}' $c)"
done
for s in vania-brain ollama; do
  echo "$s: $(systemctl show -p ActiveEnterTimestamp --value $s)"
done
ls -la --time-style=full-iso ~/dataset_run_restore.sh ~/braco/dataset_run_restore.sh 2>&1
grep -l dataset_run_restore ~/braco/*.sh ~/*.sh 2>/dev/null
ls -la --time-style=full-iso ~/braco/*.log | tail -5
tail -5 ~/braco/finalize.log
