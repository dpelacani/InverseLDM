#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

echo "starting ray head node"
# Launch the head node
ray start --head --node-ip-address=$1 --port=$2 --include-dashboard=False
echo "finished starting ray head node"
sleep infinity