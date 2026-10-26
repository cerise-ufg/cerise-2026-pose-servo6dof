#!/bin/bash
# Gera o URDF do servo6dof a partir do xacro ORIGINAL (copiado, ~/braco e' somente leitura),
# usando a imagem ROS 2 Humble ja existente no Spark (servo6dof:humble).
# Monta um "prefixo ament" falso com copias dos pacotes, para o $(find ...) do xacro funcionar.
set -e
BASE=~/cerise2026/pose/geometria
PFX=$BASE/prefix
rm -rf "$PFX"; mkdir -p "$PFX/share/ament_index/resource_index/packages"
for p in servo6dof_description servo6dof_controller; do
  cp -r ~/braco/src/servo6dof/$p "$PFX/share/$p"
  touch "$PFX/share/ament_index/resource_index/packages/$p"
done
sudo docker rm -f cerise26_pose_xacro >/dev/null 2>&1 || true
sudo docker run --rm --name cerise26_pose_xacro -v "$PFX":/prefix -v "$BASE":/out servo6dof:humble \
  bash -c "source /opt/ros/humble/setup.bash && export AMENT_PREFIX_PATH=/prefix:\$AMENT_PREFIX_PATH && \
           xacro /prefix/share/servo6dof_description/urdf/servo6dof.urdf.xacro > /out/servo6dof_xacro.urdf && \
           echo XACRO_OK && wc -l /out/servo6dof_xacro.urdf"
# o URDF que esta no repositorio do braco (gerado anteriormente) - para comparar
cp ~/braco/src/servo6dof/servo6dof_description/urdf/servo6dof.urdf $BASE/servo6dof_repo.urdf
diff <(grep -v '^<!--\|^ *<!--\||' $BASE/servo6dof_xacro.urdf) <(grep -v '^<!--\|^ *<!--\||' $BASE/servo6dof_repo.urdf) > $BASE/diff_xacro_vs_repo.txt && echo "URDF identico ao do repo" || echo "URDF difere do repo (ver diff_xacro_vs_repo.txt)"
# copia das malhas para renderizacao/overlay
mkdir -p $BASE/meshes && cp ~/braco/src/servo6dof/servo6dof_description/meshes/*.stl $BASE/meshes/
sudo chown -R $(id -u):$(id -g) $BASE
