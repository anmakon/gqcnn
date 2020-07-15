import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import csv

class Analyse_Normalisation():
	def __init__(self,calc,visu):
		cornell_path = "./data/training/Cornell/"
		self.output_path = "./analysis/Normalisation/"
		if not os.path.exists(self.output_path):
			os.mkdir(self.output_path)
		dexnet_path = "./data/training/dexnet_2_tensor/"

		self.num_random_files = 10000

		if calc:
			self.cornell = True
			print("Cornell.")
			self._compute_mean_and_std(cornell_path)
			self.cornell = False
			print("DexNet.")
			self._compute_mean_and_std(dexnet_path)
		if visu:
			self.cornell = True
			self._visualise_normalised_data(cornell_path)
			self.cornell = False
			self._visualise_normalised_data(dexnet_path)
		return None

	def _visualise_normalised_data(self,path):
		random_file_indices = self._get_random_file_indices(path)
		self._get_normalisation_data('Cornell')
		self._normalise_depth_and_pose(path,random_file_indices)
		#plot histograms
		self._plt_single_hist(self.depths,'image',norm_data='Cornell')
		self._plt_single_hist(self.pos,'pos',norm_data='Cornell')

		self._get_normalisation_data('DexNet')
		self._normalise_depth_and_pose(path,random_file_indices)
		#plot histograms
		self._plt_single_hist(self.depths,'image',norm_data='DexNet')
		self._plt_single_hist(self.pos,'pos',norm_data='DexNet')

	def _normalise_depth_and_pose(self,path,random_files):
		self.depths = []
		self.pos = []
		for tensor,array in random_files:
			im_data,pos = self._load_training_data(path,int(tensor),int(array))
			self.pos.append((pos-self.pos_mean)/self.pos_std)
			data = ((im_data-self.im_mean)/self.im_std).flatten().tolist()
			self.depths.extend(data)
		return None


	def _get_normalisation_data(self,data):
		self.im_mean = np.load(self.output_path+data+'_image_mean.npy')
		self.im_std = np.load(self.output_path+data+'_image_std.npy')
		self.pos_mean = np.load(self.output_path+data+'_pos_mean.npy')
		self.pos_std = np.load(self.output_path+data+'_pos_std.npy')
		return None


	def _compute_mean(self,path,random_files):
		num_im = 0
		num_pos = 0
		for tensor,array in random_files:
			im_data,pos = self._load_training_data(path,int(tensor),int(array))
			num_im += im_data.shape[0]*im_data.shape[1]*im_data.shape[2]
			num_pos += 1
			self.im_mean += np.sum(im_data)
			self.pos_mean += pos
		self.im_mean = self.im_mean / num_im
		self.pos_mean = self.pos_mean / num_pos
		return None

	def _compute_std(self,path,random_files):
		num_im = 0
		outliers = []
		num_pos = 0
		for tensor,array in random_files:
			im_data,pos = self._load_training_data(path,int(tensor),int(array))
			data = im_data.flatten().tolist()
			num_im += im_data.shape[0]*im_data.shape[1]*im_data.shape[2]
			num_pos += 1
			self.depths.extend(data)
			self.pos.append(pos)
			if min(data)< 0.2:
				print("Found depth image with outlier.")
				print("Dataset:",path.split('/')[-1])
				print("Tensor: ", tensor, "; Array: ",array)
				outliers.append([tensor,array,tensor,array])
			self.im_std += np.sum((im_data - self.im_mean)**2)
			self.pos_std += np.sum((pos - self.pos_mean)**2)
		self.im_std = np.sqrt(self.im_std / num_im)
		self.pos_std = np.sqrt(self.pos_std / num_pos)
		if len(outliers) >= 1:
			self._save_outlier_pointers(outliers)
		return 1
	
	def _get_random_file_indices(self,path):
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

	def _compute_mean_and_std(self,path):
		self.im_mean = 0
		self.im_std = 0
		self.pos_mean = 0
		self.pos_std = 0
		self.pos = []
		self.depths = []

		random_file_indices = self._get_random_file_indices(path)
		print("Amount of files: ",len(random_file_indices))
		self._compute_mean(path,random_file_indices)
		print("Mean is:",self.im_mean)
		self._compute_std(path,random_file_indices)

		print("Std is:",self.im_std)
		self.save_data()

	
	def _save_outlier_pointers(self,outliers):
		export_path = "./data/training/csv_files"
		if self.cornell:
			export_path += "/Cornell_outliers.csv"
		else:
			export_path += "/DexNet_outliers.csv"
		with open(export_path,'w',newline='') as csvfile:
			writer = csv.writer(csvfile,delimiter = ',')
			for row in outliers:
				writer.writerow(row)
		return None

	def save_data(self):
		self._plt_single_hist(self.depths,'image',mean=self.im_mean,std=self.im_std)
		self._plt_single_hist(self.pos,'pos',mean=self.pos_mean,std=self.pos_std)

	def _plt_single_hist(self,data,mode,mean=None,std=None,norm_data=None):
		if self.cornell:
			dset = 'Cornell'
			unit = '[m]'
		else:
			dset= 'DexNet'
			unit = '[units]'
		if mode == 'image':
			data_type = 'Depth'
		elif mode == 'pos':
			data_type = 'Pose'
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
		
		# Create and save histogram of Depths/Poses
		n,bins,_ =plt.hist(data,bins = np.arange(min(data),max(data)+binwidth,binwidth))
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
		if mean is not None and std is not None:
			plt.text(mean-2.5*std,max(n)*7/8,"Mean: "+str(round(mean,3)))
			plt.text(mean-2.5*std,max(n)*6/8,"Std: "+str(round(std,3)))
			plt.xlim((mean-3*std,mean+3*std))
		else:
			plt.xlim((-1.5,1.5))
		plt.savefig(path)	
		plt.close()
		return None


	def _load_training_data(self,path,tensor,array):
		depth_path = path +"tensors/depth_ims_tf_table_"
		label_path = path +"tensors/hand_poses_"
		hand_pose = np.load(label_path+("{0:05d}").format(tensor)+".npz")['arr_0'][array]
		depth = np.load(depth_path+("{0:05d}").format(tensor)+".npz")['arr_0'][array]
		return depth,hand_pose[2]

	def _read_training_indices(self,path):
		path += "splits/image_wise/train_indices.npz"
		return np.load(path)['arr_0'] 
	
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

	args = parser.parse_args()
	calc = args.calc
	visu = args.visu
				
	Analyse_Normalisation(calc,visu)
