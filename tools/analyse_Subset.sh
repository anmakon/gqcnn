#!/bin/bash

# Script to analyse a GQCNN trained on Cornell on Cornell data and a GQCNN 
# trained on DexNet on DexNet and Cornell data. Grasps to analyse are given
# in the corresponding csv files.

CurrentDate=`date +"%Y%m%d"`

python ./tools/dataset_manipulation/extract_modified_grasps.py --selection txt
python ./tools/detailed_analysis.py GQCNN-2.0_benchmark data/training/Subset_datasets/DexNet_/ --output_dir "./analysis/SingleFiles/Dataset_Generation/"$CurrentDate"_original/"

