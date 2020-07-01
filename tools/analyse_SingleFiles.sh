#!/bin/bash

cornell=false
dexnet=true

if [ "$dexnet" = true ] ; then
	input="./data/training/csv_files/DexNet_SingleFiles.csv"
	while IFS=',' read -r file array
	do
		echo "File $file Array $array"
		python ./tools/dataset_manipulation/extract_perturbated_grasps.py DexNet --file "$file" --array "$array" --type rotation
		python ./tools/detailed_analysis.py GQCNN-2.0_benchmark data/training/Subset_datasets/Cornell_SinglePerturb/ --analysis single

		python ./tools/dataset_manipulation/extract_perturbated_grasps.py DexNet --file "$file" --array "$array" --type translation
		python ./tools/detailed_analysis.py GQCNN-2.0_benchmark data/training/Subset_datasets/Cornell_SinglePerturb/ --analysis single

		python ./tools/dataset_manipulation/extract_perturbated_grasps.py DexNet --file "$file" --array "$array" --type translationy
		python ./tools/detailed_analysis.py GQCNN-2.0_benchmark data/training/Subset_datasets/Cornell_SinglePerturb/ --analysis single
	done < "$input"

	rm -rf analysis/SingleFiles/DexNet_Single_Analysis
	mv analysis/SingleFiles/Single_Analysis analysis/SingleFiles/DexNet_Single_Analysis
fi

if [ "$cornell" = true ] ; then
	input="./data/training/csv_files/Cornell_SingleFiles.csv"
	while IFS=',' read -r file array
	do
		echo "File $file Array $array"
		python ./tools/dataset_manipulation/extract_perturbated_grasps.py Cornell --file "$file" --array "$array" --type rotation
		python ./tools/detailed_analysis.py GQCNN-2.0_benchmark data/training/Subset_datasets/Cornell_SinglePerturb/ --analysis single
	
		python ./tools/dataset_manipulation/extract_perturbated_grasps.py Cornell --file "$file" --array "$array" --type translation
		python ./tools/detailed_analysis.py GQCNN-2.0_benchmark data/training/Subset_datasets/Cornell_SinglePerturb/ --analysis single

		python ./tools/dataset_manipulation/extract_perturbated_grasps.py Cornell --file "$file" --array "$array" --type translationy
		python ./tools/detailed_analysis.py GQCNN-2.0_benchmark data/training/Subset_datasets/Cornell_SinglePerturb/ --analysis single
	done < "$input"

	rm -rf analysis/SingleFiles/Cornell_on_DexNet_Single_Analysis
	mv analysis/SingleFiles/Single_Analysis analysis/SingleFiles/Cornell_on_DexNet_Single_Analysis

	while IFS=',' read -r file array
	do
		echo "File $file Array $array"
		python ./tools/dataset_manipulation/extract_perturbated_grasps.py Cornell --file "$file" --array "$array" --type rotation
		python ./tools/detailed_analysis.py GQCNN-2.0_Cornell_benchmark data/training/Subset_datasets/Cornell_SinglePerturb/ --analysis single
	
		python ./tools/dataset_manipulation/extract_perturbated_grasps.py Cornell --file "$file" --array "$array" --type translation
		python ./tools/detailed_analysis.py GQCNN-2.0_Cornell_benchmark data/training/Subset_datasets/Cornell_SinglePerturb/ --analysis single
	
		python ./tools/dataset_manipulation/extract_perturbated_grasps.py Cornell --file "$file" --array "$array" --type translationy
		python ./tools/detailed_analysis.py GQCNN-2.0_benchmark data/training/Subset_datasets/Cornell_SinglePerturb/ --analysis single
	done < "$input"
	
	rm -rf analysis/SingleFiles/Cornell_Single_Analysis
	mv analysis/SingleFiles/Single_Analysis analysis/SingleFiles/Cornell_Single_Analysis
fi
