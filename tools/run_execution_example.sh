#!/bin/bash

input="data/training/csv_files/Cornell_Execution_objects.csv"
depth="data/training/Grasp_plan_data/depth_0.npy"
binary="data/training/Grasp_plan_data/binary_0.png"
camera="data/calib/primesense/primesense.intr"
config="cfg/examples/replication/dex-net_2.0.yaml"

while IFS=',' read -r file
do
	python ./tools/dataset_manipulation/Convert_Cornell_to_dexnet_format.py --create_orig True --Cornell_num "$file"

	save_path="./analysis/execution_real_grasps/Cornell_obj_${file}"
	
	echo "Model trained on DexNet"

	python examples/policy.py --depth_image "$depth" --segmask "$binary" --camera_intr "$camera" --config_filename "$config" --save_path "${save_path}_model_trained_on_DexNet" GQCNN-2.0_benchmark

	echo "Model trained on Cornell"

	python examples/policy.py --depth_image "$depth" --segmask "$binary" --camera_intr "$camera" --config_filename "$config" --save_path "${save_path}_model_trained_on_Cornell" GQCNN-2.0_Cornell_benchmark

done < "$input"
