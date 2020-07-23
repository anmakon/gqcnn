#!/bin/bash

# Script to train a GQCNN multiple times. Used for evaluating the influence of
# optimisation parameters on the training results.

for VARIABLE in 1 2 3 4 5 6 7 8 9 10
do

	python3 ./tools/train.py /dataset/Cornell --config_filename cfg/train_dexnet_2.0.yaml --seed 0 --name GQCNN-2.0_Cornell_benchmark_split7_"$VARIABLE"
	python3 ./tools/analyze_gqcnn_performance.py GQCNN-2.0_Cornell_benchmark_split7_"$VARIABLE"

done
