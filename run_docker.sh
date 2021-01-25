#!/bin/bash

DATA_PATH='/home/anna/Grasping/data'
EXPER_PATH='/home/anna/Grasping/results'

echo Run docker for cpu [0] or gpu [1]?
read cond

if [ $cond == 0 ]
then
	docker run -it \
	-v ${PWD}:${PWD} \
	-v $DATA_PATH:/data \
	-v $EXPER_PATH:/results \
	-w ${PWD} \
	--gpus all \
	--name gqcnn_cpu \
	gqcnn:cpu
else
	docker run -it \
	-v ${PWD}:${PWD} \
	-v $DATA_PATH:/data \
	-v $EXPER_PATH:/results \
	-w ${PWD} \
	--gpus all \
	--name gqcnn_gpu \
	gqcnn:gpu
fi

