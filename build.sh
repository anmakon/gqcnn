#!/bin/bash

echo Build dockerfile for cpu [0] or gpu [1]?
read cond

tar -cvf docker/gqcnn.tar ../gqcnn

if [ $cond == 0 ]
then
	docker build -t gqcnn:cpu -f docker/cpu/Dockerfile .
else
	docker build -t gqcnn:gpu -f docker/gpu/Dockerfile .
fi

