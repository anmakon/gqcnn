#!/bin/bash

read -p 'File: ' spec_file
read -p 'Array: ' spec_array

python tools/dataset_manipulation/extract_perturbated_grasps.py Cornell --file $spec_file --array $spec_array --selection manual --type mixed --value 0
python tools/detailed_analysis.py GQCNN-2.0_benchmark data/training/Subset_datasets/Cornell_SinglePerturb/ --analysis single
python tools/dataset_manipulation/visualise_crosssection.py $spec_file $spec_array --single True
