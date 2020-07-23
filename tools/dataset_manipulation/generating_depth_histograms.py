import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


class Generate_Depth_Maps():
	def __init__(self,data_path,output_path,selection,tensor=None,array=None):
		self.data_path = data_path
		self.output_path = output_path

		if 'dexnet_2_tensor' in data_path:
			self.dset = 'DexNet'
		else:
			self.dset = 'Cornell'
		
		if tensor is not None and array is not None:
			self.tensor = tensor
			self.array = array
			self._create_depthmap() 
			return None
		
	def _create_depthmap(self):
		filestring = ("{0:05d}").format(self.tensor)
		depth_ims = np.load(self.data_path+"depth_ims_tf_table_"+filestring+".npz")['arr_0'][self.array].flatten()
		binwidth = 0.001
		plt.hist(depth_ims,bins = np.arange(min(depth_ims),max(depth_ims)+binwidth,binwidth))
		plt.title("Depths in file "+filestring+", array "+ str(array)+' '+self.dset+ ' dataset')
		plt.xlabel("Depth [m]")
#		plt.ylim((0,1030))
#		plt.xlim((0.6,0.8))
		plt.savefig(self.output_path+filestring+'_'+str(array))
		return None

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("dset",
				type = str,
				default = 'Cornell',
				help = "Dataset to take depth maps from")
	parser.add_argument("--file",
				type = int,
				default = None,
				help="File number of depth map")
	parser.add_argument("--array",
				type = int,
				default = None,
				help = "Array number of depth map")
	parser.add_argument("--selection",
				type = str,
				default = 'Manual',
				help = "Method to select the images")
	args = parser.parse_args()
	tensor = args.file
	array = args.array
	dset = args.dset
	selection = args.selection
	if dset == 'Cornell' or dset == 'cornell':
		data_path = './data/training/Cornell/tensors/'
		output_path = './analysis/SingleFiles/Cornell_DepthMaps/'
	elif dset == 'DexNet' or dset=='Dexnet' or dset == 'dexnet':
		data_path = './data/training/dexnet_2_tensor/tensors/'
		output_path = './analysis/SingleFiles/DexNet_DepthMaps/'
	else:
		raise ValueError("No recognisable dataset name given.")

	if not os.path.exists(output_path):
		os.mkdir(output_path)
	print("Writing to",output_path)
	if not os.path.exists(data_path):
		raise NameError("Path to data does not exist.")
	if tensor is not None and array is not None:
		Generate_Depth_Maps(data_path,output_path,selection,tensor,array)
	else:
		Generate_Depth_Maps(data_path,output_path,selection)
