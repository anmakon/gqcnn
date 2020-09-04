#!/bin/bash

# Script to analyse a GQCNN trained on Cornell on Cornell data and a GQCNN 
# trained on DexNet on DexNet and Cornell data. Grasps to analyse are given
# in the corresponding csv files.



input="../dex-net/data/generated_val_indices.txt"
while IFS=',' read -r file array _ _ _ _
do
	if [[ "$file" != "Tensor" ]]; then
		python ./tools/dataset_manipulation/extract_modified_grasps.py --selection manual --file "$file" --array "$array"
		python ./tools/detailed_analysis.py GQCNN-2.0_benchmark data/training/Subset_datasets/DexNet_/ --output_dir "./analysis/SingleFiles/Dataset_Generation/Original/"
	fi
done < "$input"

