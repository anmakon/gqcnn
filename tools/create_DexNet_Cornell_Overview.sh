#!/bin/bash

cornell=true
dexnet=true
rm -rf analysis/SingleFiles/Single_Analysis

if [ "$dexnet" = true ] ; then
	input="./data/training/csv_files/DexNet_GB_Single.csv"
	while IFS=',' read -r file array
	do
		echo "File $file Array $array"
		python ./tools/dataset_manipulation/extract_perturbated_grasps.py DexNet --file "$file" --array "$array" --type translation
		python ./tools/detailed_analysis.py GQCNN-2.0_benchmark data/training/Subset_datasets/DexNet_SinglePerturb/ --analysis single
	done < "$input"

	rm -rf analysis/Dset_Overview/DexNet_Single_Analysis
	mv analysis/SingleFiles/Single_Analysis analysis/Dset_Overview/DexNet_Single_Analysis

	while IFS=',' read -r file array
	do
		echo "File $file Array $array"
		python ./tools/dataset_manipulation/extract_perturbated_grasps.py DexNet --file "$file" --array "$array" --type translation
		python ./tools/detailed_analysis.py GQCNN-2.0_Cornell_benchmark data/training/Subset_datasets/DexNet_SinglePerturb/ --analysis single
	done < "$input"

	rm -rf analysis/Dset_Overview/DexNet_on_Cornell_Single_Analysis
	mv analysis/SingleFiles/Single_Analysis analysis/Dset_Overview/DexNet_on_Cornell_Single_Analysis
fi

if [ "$cornell" = true ] ; then
	input="./data/training/csv_files/Cornell_GB_Single.csv"
	while IFS=',' read -r file array
	do
		echo "File $file Array $array"
		python ./tools/dataset_manipulation/extract_perturbated_grasps.py Cornell --file "$file" --array "$array" --type translation
		python ./tools/detailed_analysis.py GQCNN-2.0_benchmark data/training/Subset_datasets/Cornell_SinglePerturb/ --analysis single
	done < "$input"

	rm -rf analysis/Dset_Overview/Cornell_on_DexNet_Single_Analysis
	mv analysis/SingleFiles/Single_Analysis analysis/Dset_Overview/Cornell_on_DexNet_Single_Analysis

	while IFS=',' read -r file array
	do
		echo "File $file Array $array"
		python ./tools/dataset_manipulation/extract_perturbated_grasps.py Cornell --file "$file" --array "$array" --type translation
		python ./tools/detailed_analysis.py GQCNN-2.0_Cornell_benchmark data/training/Subset_datasets/Cornell_SinglePerturb/ --analysis single
	done < "$input"
	
	rm -rf analysis/Dset_Overview/Cornell_Single_Analysis
	mv analysis/SingleFiles/Single_Analysis analysis/Dset_Overview/Cornell_Single_Analysis
fi

python ./tools/dataset_manipulation/generating_dataset_overview.py
