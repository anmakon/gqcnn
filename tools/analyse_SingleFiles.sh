#!/bin/bash

input="./data/training/csv_files/DexNet_SingleFiles.csv"
while IFS=',' read -r file array
do
	echo "File $file Array $array"
	python ./tools/dataset_manipulation/extract_perturbated_grasps.py DexNet --file "$file" --array "$array" --type rotation

	python ./tools/detailed_analysis.py GQCNN-2.0_benchmark data/training/Subset_datasets/DexNet_SinglePerturb/ --analysis single

	python ./tools/dataset_manipulation/extract_perturbated_grasps.py DexNet --file "$file" --array "$array" --type rotation

	python ./tools/detailed_analysis.py GQCNN-2.0_benchmark data/training/Subset_datasets/DexNet_SinglePerturb/ --analysis single
done < "$input"

