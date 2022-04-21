#!/usr/bin/env bash
if [[ "$(< /proc/version)" == *@(Microsoft|WSL)* ]]; then
  if [ "$(sudo service docker status)" = " * Docker is not running" ]; then
    sudo service docker start
  fi
else
  if [ "$(sudo systemctl is-active docker)" = "inactive" ]; then
    sudo systemctl restart docker
  fi
fi
cd ./docker_build_context
docker build --rm -t axd465/final_proj_wire_comp_dl:gpu-2.5.0 .
cd ..
port_number=8888 # Starting Port
while nc -z localhost $port_number ; do
  ((port_number++))
done

docker run -it --rm "$@" -u root -p $port_number:$port_number -e \
JUPYTER_PORT=$port_number -v "${PWD}":/tf axd465/final_proj_wire_comp_dl:gpu-2.5.0
