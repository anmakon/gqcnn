import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import csv

"""
This is a script to analyse the normalisation values in the DexNet and Cornell dataset.
The normalisation values can be calculated with --calc True and the normalisation applied
to the datasets with --visu True.
The script uses the original datasets in "./data/training" as default. The depth and pose
values are stored and can be used instead of accessing the original dataset with --buffer True.
The normalisation values are the mean and the standarad deviation of a gaussian curve that
is fitted to the depth and pose values.

"""

class Normalisation():
	def __init__(self,calc,visu):
		
		cornell_path = "./data/training/Cornell/"
		self.output_path = "./analysis/Normalisation/"
		if not os.path.exists(self.output_path):
			os.mkdir(self.output_path)
		dexnet_path = "./data/training/dexnet_2_tensor/"

		self.num_random_files = 10000
		#Calculate the normalisation values

		if calc:
			self.cornell = True
			print("Cornell.")
			self._compute_mean_and_std(cornell_path)
			self.cornell = False
			print("DexNet.")
			self._compute_mean_and_std(dexnet_path)
		# Apply the normalisation values to the data

		if visu:
			self.cornell = True
			self._visualise_normalised_data(cornell_path)
			self.cornell = False
			self._visualise_normalised_data(dexnet_path)
		return None

	def _get_normalisation_data(self,data):
		# Used when applying the normalisation to the data.
		self.im_mean = np.load(self.output_path+data+'_image_mean.npy')
		self.im_std = np.load(self.output_path+data+'_image_std.npy')
		self.pos_mean = np.load(self.output_path+data+'_pos_mean.npy')
		self.pos_std = np.load(self.output_path+data+'_pos_std.npy')
		return None

	def _compute_mean_and_std(self,path):
		self.im_mean = 0
		self.im_std = 0
		self.pos_mean = 0
		self.pos_std = 0
		self.pos = []
		self.depths = []

		self._compute_mean(path)
		print("Mean is:",self.im_mean)
		self._compute_std(path)

		print("Std is:",self.im_std)

		self._plt_single_hist(self.depths,'image',mean=self.im_mean,std=self.im_std)
		self._plt_single_hist(self.pos,'pos',mean=self.pos_mean,std=self.pos_std)

	def _visualise_normalised_data(self,path):
		self._get_normalisation_data('Cornell')
		self._normalise_depth_and_pose(path)

		#plot histograms
		self._plt_single_hist(self.depths,'image',norm_data='Cornell')
		self._plt_single_hist(self.pos,'pos',norm_data='Cornell')

		self._get_normalisation_data('DexNet')
		self._normalise_depth_and_pose(path)
		#plot histograms

		self._plt_single_hist(self.depths,'image',norm_data='DexNet')
		self._plt_single_hist(self.pos,'pos',norm_data='DexNet')

	def _plt_single_hist(self,data,mode,mean=None,std=None,norm_data=None):
		# Plot histograms. Different modes specify the text and saving path of the image
		if self.cornell:
			dset = 'Cornell'
			unit = '[m]'
		else:
			dset= 'DexNet'
			unit = '[units]'
		if mode == 'image':
			data_type = 'Depth'
			y_limit = (0,250000)
			y_pose = 230000
		elif mode == 'pos':
			data_type = 'Pose'
			y_limit= (0,250)
			y_pose = 230
		path = self.output_path + data_type+"s_in_"+dset+'_'+mode
		if mean is None and std is None:
			path += '_normalised_on_'+norm_data
			binwidth = 0.05
			title = data_type+" values in "+dset+" dataset normalised on "+norm_data+" data"
			unit = ', normalised'
		else:
			binwidth = 0.001
			title = data_type+" values in "+dset+ " dataset"

			#Save Mean and Std to .npy files
			np.save(self.output_path+dset+'_'+mode+"_mean",mean)
			np.save(self.output_path+dset+'_'+mode+"_std",std)
			# Save data to file in order to use it for the buffer-method.
			# This is done every time that you calculate the normalisation values
			# directly from the dataset.
			np.savez(self.output_path+dset+'_'+mode,data)
		hist_color = (0.2,0.6,1,0.8)
		
		# Create and save histogram of Depths/Poses
		n,bins,_ =plt.hist(data,bins=np.arange(min(data),max(data)+binwidth,binwidth),color=hist_color)
		plt.title(title)
		plt.xlabel(data_type+" "+unit)
		if mean is not None and std is not None:
			plt.vlines(mean,0,max(n),colors='r')
			plt.text(bins[1],max(n)*7/8,"Mean: "+str(round(mean,3)))
			plt.vlines(mean-std,0,max(n),colors='r')
			plt.vlines(mean+std,0,max(n),colors='r')	
			plt.text(bins[1],max(n)*6/8,"Std: "+str(round(std,3)))
		plt.savefig(path+"_all")

		# Save a 2nd image with a limited x-axis	
		plt.close()
		n,bins,_ = plt.hist(data,bins = np.arange(min(data),max(data)+binwidth,binwidth),color=hist_color)
		plt.title(title)
		plt.xlabel(data_type+" "+unit)
		if mean is not None and std is not None:
			plt.text(mean-2.5*std,max(n)*7/8,"Mean: "+str(round(mean,3)))
			plt.text(mean-2.5*std,max(n)*6/8,"Std: "+str(round(std,3)))
			plt.xlim((mean-3*std,mean+3*std))
		else: 
			data_range = "Data range: ["+("{:02.1f}").format(min(data))+";"+("{:02.1f}").format(max(data))+"]"
			plt.xlim((-6,3))
			plt.ylim(y_limit)
			plt.text(-5.5,y_pose,data_range)
		plt.savefig(path)	
		plt.close()
		return None

class Normalisation_from_dset(Normalisation):
	def __init__(self,calc,visu):
		super().__init__(calc,visu)

	def _compute_mean(self,path):
		# Get indices from the training dataset
		random_files = self._get_random_file_indices(path)
		print("Amount of files: ",len(random_files))
		num_im = 0
		num_pos = 0
		# Browse through data and calculate the mean
		for tensor,array in random_files:
			im_data,pos = self._load_training_data(path,int(tensor),int(array))
			num_im += im_data.shape[0]*im_data.shape[1]*im_data.shape[2]
			num_pos += 1
			self.im_mean += np.sum(im_data)
			self.pos_mean += pos
		self.im_mean = self.im_mean / num_im
		self.pos_mean = self.pos_mean / num_pos
		return None

	def _compute_std(self,path):
		# Get indices from the training dataset
		random_files = self._get_random_file_indices(path)
		print("Amount of files: ",len(random_files))
		num_im = 0
		outliers = []
		num_pos = 0
		# Browse through data and calculate the std
		for tensor,array in random_files:
			im_data,pos = self._load_training_data(path,int(tensor),int(array))
			data = im_data.flatten().tolist()
			num_im += im_data.shape[0]*im_data.shape[1]*im_data.shape[2]
			num_pos += 1
			self.depths.extend(data)
			self.pos.append(pos)
			self.im_std += np.sum((im_data - self.im_mean)**2)
			self.pos_std += np.sum((pos - self.pos_mean)**2)
		self.im_std = np.sqrt(self.im_std / num_im)
		self.pos_std = np.sqrt(self.pos_std / num_pos)
		return None

	def _load_training_data(self,path,tensor,array):
		# Load one single training grasp
		depth_path = path +"tensors/depth_ims_tf_table_"
		label_path = path +"tensors/hand_poses_"
		hand_pose = np.load(label_path+("{0:05d}").format(tensor)+".npz")['arr_0'][array]
		depth = np.load(depth_path+("{0:05d}").format(tensor)+".npz")['arr_0'][array]
		return depth,hand_pose[2]

	def _normalise_depth_and_pose(self,path):
		random_files = self._get_random_file_indices(path)
		self.depths = []
		self.pos = []
		# Browse through data and apply normalisation to pose and depth
		for tensor,array in random_files:
			im_data,pos = self._load_training_data(path,int(tensor),int(array))
			self.pos.append((pos-self.pos_mean)/self.pos_std)
			data = ((im_data-self.im_mean)/self.im_std).flatten().tolist()
			self.depths.extend(data)
		return None
	
	def _get_random_file_indices(self,path):
		# Get random file indices from training dataset.
		# Amount of samples is either 10000 or all of the
		# training samples (in case there are less than 10000)
		index = self._read_training_indices(path)
		num_random_files = min(self.num_random_files,len(index))
		random_index = np.random.choice(index,
							size=num_random_files,
							replace=False)
		# ToDo: make this dependent on the config file
		if self.cornell:
			random_file_indices = [[pos//500,pos%500] for pos in random_index]
		else:
			random_file_indices = [[pos//1000,pos%1000] for pos in random_index]
		return random_file_indices

	def _read_training_indices(self,path):
		path += "splits/image_wise/train_indices.npz"
		return np.load(path)['arr_0'] 
	
class Normalisation_from_buffer(Normalisation):
	def __init__(self,calc,visu):
		super().__init__(calc,visu)
		return None

	def _load_buffer_data(self):
		# Load the depth and pose data from the buffer
		if self.cornell:
			data='Cornell'
		else:
			data='DexNet'
		depth = np.load(self.output_path+data+'_image.npz')['arr_0']
		pose = np.load(self.output_path+data+'_pos.npz')['arr_0']
		return depth,pose	

	def _compute_mean(self,path):
		depth,pose = self._load_buffer_data()
		print("Amount of files: ",len(pose))
		self.im_mean = sum(depth) / len(depth)
		self.pos_mean = sum(pose) / len(pose)
		return None

	def _compute_std(self,path):
		depth,pose = self._load_buffer_data()
		print("Amount of files: ",len(pose))
		self.im_std = np.sqrt(np.sum((depth-self.im_mean)**2)/len(depth))
		self.pos_std = np.sqrt(np.sum((pose-self.pos_mean)**2)/len(pose))
		self.depths = depth
		self.pos = pose
		return None

	def _normalise_depth_and_pose(self,path):
		depth,pose = self._load_buffer_data()
		self.depths = (depth-self.im_mean)/self.im_std
		self.pos = (pose-self.pos_mean)/self.pos_std
		return None

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=("Analyse the normalisation parameters of DexNet and Cornell"))
	parser.add_argument("--calc",
				type = bool,
				default = False,
				help="Calculate the mean and std")
	parser.add_argument("--visu",
				type = bool,
				default = False,
				help="Visualise the normalised depths/poses")
	parser.add_argument("--buffer",
				type = bool,
				default = False,
				help="Use buffer as data origin ")

	args = parser.parse_args()
	calc = args.calc
	visu = args.visu
	if args.buffer:
		Normalisation_from_buffer(calc,visu)
	else:
		Normalisation_from_dset(calc,visu)
