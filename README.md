## Installation

1. Make sure you have a working version of nvidia-docker or docker installed on your computer. 
2. Clone this [gqcnn repository](https://github.com/anmakon/gqcnn) to a directory of your choice: `$WORKING_DIR/gqcnn`
3. Create a directory to store your results in, e.g. `$RESULTS`.
4. Download the trained [model](https://maynoothuniversity.sharepoint.com/:u:/r/sites/AnnaPhD/Shared%20Documents/General/models.zip?csf=1&web=1&e=VNE7zO)
   from the SharePoint and extract the zip file in `$WORKING_DIR/gqcnn/.`.
5. Run `./build.sh` from `$WORKING_DIR/gqcnn/`. Choose 1 if you have nvidia-docker and would like to use the GPU, 0 otherwise.
6. Set the variables `DATA_PATH` and `EXPER_PATH` in `gqcnn/run_docker.sh` to your `$DATA_DIR` and 
   `$RESULTS` directories. `$DATA_DIR` should be pointing to the directory where your DexNet datasets are stored.
7. Run `./run_docker.sh` from `$WORKING_DIR/gqcnn/`. 

### Evaluate dataset

In order to evaluate a dataset, run
```
python3 tools/detailed_analysis.py GQCNN-2.0_benchmark $DATASET_DIR --output_dir $DATASET_OUTPUT_DIR
```

For example, if you want to evaluate a recreated PerfectPredictions dataset, run:
```
python3 tools/detailed_analysis.py GQCNN-2.0_benchmark Recreated_grasps/tensors/ --output_dir Recreated_grasps
```

The logfile and the images with prediction values are stored in `$RESULTS/$DATASET_OUTPUT_DIR/`.

------------------------
-----------------------

# Berkeley AUTOLAB's GQCNN Package
<p>
   <a href="https://travis-ci.org/BerkeleyAutomation/gqcnn/">
       <img alt="Build Status" src="https://travis-ci.org/BerkeleyAutomation/gqcnn.svg?branch=master">
   </a>
   <a href="https://github.com/BerkeleyAutomation/gqcnn/releases/latest">
       <img alt="Release" src="https://img.shields.io/github/release/BerkeleyAutomation/gqcnn.svg?style=flat">
   </a>
   <a href="https://github.com/BerkeleyAutomation/gqcnn/blob/master/LICENSE">
       <img alt="Software License" src="https://img.shields.io/badge/license-REGENTS-brightgreen.svg">
   </a>
   <a>
       <img alt="Python 3 Versions" src="https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-yellow.svg">
   </a>
</p>

## Package Overview
The gqcnn Python package is for training and analysis of Grasp Quality Convolutional Neural Networks (GQ-CNNs). It is part of the ongoing [Dexterity-Network (Dex-Net)](https://berkeleyautomation.github.io/dex-net/) project created and maintained by the [AUTOLAB](https://autolab.berkeley.edu) at UC Berkeley.

## Installation and Usage
Please see the [docs](https://berkeleyautomation.github.io/gqcnn/) for installation and usage instructions.

## Citation
If you use any part of this code in a publication, please cite [the appropriate Dex-Net publication](https://berkeleyautomation.github.io/gqcnn/index.html#academic-use).

