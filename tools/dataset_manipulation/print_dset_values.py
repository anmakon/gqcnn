import numpy as np
import argparse

parser = argparse.Argumentparser("Prints the values in the Cornell and DexNet dataset")
parser.add_argument("dset",
			type = str,
			default = 'Cornell',
			help = "Dataset to show")
parser.add_argument("file",
			type = str,
			help = "File to show")
parser.add_argument("--array",
			type = str,
			default = None,
			help = "Array to show")
filenum = parser.file
array = parser.array
dset = parser.dset
filecnt = {"0.05d"}.format(filenum)

if dset == 'Cornell' or dset == 'cornell':
	data_path = "./data/training/Cornell/tensors/"
elif dset == 'dexnet' or dset == 'Dexnet' or dset == 'DexNet':
	data_path = "./data/training/dexnet_2_tensor/tensors/"

#depth_im = np.load(data_path+'depth_ims_raw_table_'+filecnt+'.npz')
#print('depth_ims_raw_table shape:',depth_im.files)
#with np.printoptions(threshold=np.inf):
#print('depth_ims_raw_table arr_0', depth_im['arr_0'][0])

#depth_im_table = np.load(data_path+'depth_ims_tf_table_'+filecnt+'.npz')
#print('depth_ims_tf_table shape:',depth_im_table.files)
#with np.printoptions(threshold=np.inf):
#print('depth_ims_tf_table arr_0', depth_im_table['arr_0'][0])
#print('depht_ims_tf_table arr_0 shape',depth_im_table['arr_0'].shape)

#binary_im_table = np.load(data_path+'binary_ims_tf_table_'+filecnt+'.npz')
#print('binary_ims_tf_table shape:',binary_im_table.files)
#print('binary_ims_tf_table arr_0', binary_im_table['arr_0'])
#for i in range(0,17):
#	filecnt = ('{0:05d}').format(i)
#	hand_pose = np.load(data_path+'hand_poses_'+filecnt+'.npz')
#	#print('hand_pose shape:',hand_pose.files)
#	gripper = hand_pose['arr_0'][:,-1]
#	with np.printoptions(threshold=np.inf):
#		print('hand_pose arr_0',gripper)
#		print('Gripper width range:', np.min(gripper),np.max(gripper))

#grasp_metric = np.load(data_path+'robust_ferrari_canny_'+filecnt+'.npz')
#print('robust_ferrari_canny shape:',grasp_metric.files)
#print('robust_ferrari_canny arr_0:',grasp_metric['arr_0'])

object_labels = np.load(data_path+'object_labels_'+filecnt+'.npz')
print('object labels shape:',object_labels.files)
print('object labels arr_0:', object_labels['arr_0'][array])

pose_labels = np.load(data_path+'pose_labels_'+filecnt+'.npz')
print('pose labels shape:',pose_labels.files)
print('pose labels arr_0:', pose_labels['arr_0'][array])

#image_labels = np.load(data_path+'image_labels_'+filecnt+'.npz')
#print('image labels shape:',image_labels.files)
#print('image labels arr_0:', image_labels['arr_0'])
