#!/bin/bash

input="./data/training/csv_files/DexNet_SingleFiles.csv"
while IFS=',' read -r file array
do
	echo "File $file Array $array"
	python ./tools/dataset_manipulation/generating_depth_histograms.py DexNet --file "$file" --array "$array"
done < "$input"

input="./data/training/csv_files/Cornell_SingleFiles.csv"
while IFS=',' read -r file array
do
	echo "File $file Array $array"
	python ./tools/dataset_manipulation/generating_depth_histograms.py Cornell --file "$file" --array "$array"
done < "$input"
