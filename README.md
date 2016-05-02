# 3D_features_classification_CNN

The following work was part of my lab rotation at the lab of Prof. Obermayer at the TU Berlin.

-------

Working Flow:
1. Download data-set from http://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_pcd_ascii/

2. create fpfh files with the script fpfh.cpp after changing the necessary paths in the code.

Create training and testing datasets of RGB and FPFH features:

Option A - use generate_experiment_files.py from the generate_experiment_files package:
input:
source_dir – directory for the csv files.
lmbd_dir - path to save the json and lmdb files.
mean_dir - path to save the mean files
labels_dict - where to find the dictionary of mapping categories to labels.


Option B - run each script separately:

1. create training and testing sets with the script prepare_args_all_instances.py. 
This script results in a json file containing the same split of train and test with different subsampling of the views (every second, third, fifth view). All other scripts rely on the specific structure of this json file.
Input:
source_dir – directory for the csv files.
results_dir – where to save the arguments files.
file_name – how to call the arguments file (should end with .json!!).


2. generate lmdb files using read_csv_seperated_rgb_hist.py 
Input:
arguments_file – path to the arguments file.
Hist – if True generates histograms file. Defult – False, generate rgb file.
Index – index in the arguments file (0 – all instances training, 1 all instances testing. 2,3 same for every second view. 4,5 every third view. 6,7 every fifth view).

3. generate labels using generate_labels.py. 
Input:
arguments_file
labels_dict – path to a json file containing the mapping between objects and labels.
db_path – where to save the labels file.
Index – for wich dataset (training/testing, every _ view).

4. generate mean image binaryproto file using generate_mean_image_binaryprotos.py. 
Input:
arguments_file – path to the arguments file.
Hist – if True generates histograms file. Defult – False, generate rgb file.
Index – index in the arguments file (0 – all instances training, 1 all instances testing. 2,3 same for every second view. 4,5 every third view. 6,7 every fifth view).
results_path – where to save the mean file.

5. define a solver.prototxt and train_test.prototxt and load and train the network using the script simple_nework.py while directing the output to aa lig file.
